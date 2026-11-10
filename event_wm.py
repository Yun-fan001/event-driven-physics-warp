"""
事件序列自回归世界模型 · 三路对照 (2026-11, 对应知识库 F076 / 预注册问题 Q6)

科学问题: 接触密集操作的"世界模型"应把 roll-out 状态表示成
  显式离散物理事件 (EventWM, 事件级监督),
  还是连续力值 (ForceWM, 连续回归 + 固定阈值, DreamerV3 式),
  还是隐式潜变量 (ImplicitWM, 无事件标签, PRISM-WM 式)?
指标:
  1. 首次失效时间 (free-run rollout 首次模式错误步数)
  2. rollout 事件 F1 (±5 帧)
  3. 跨摩擦泛化 (ID: 摩擦<=1.0, OOD: 摩擦>1.0)
  4. (连续模型) 力预测 MSE
  5. (干预, 实验2轻量版) 摩擦+50% 时滑移起始是否单调后移

输入: 力(3) + 推杆位置(1) + 摩擦条件(1) 的历史 K 帧
      (不含速度——速度即 GT 判据, 视觉已证明无增益, 故本实验用力+本体+摩擦)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_step2 import force_to_events

K = 64        # 输入历史 (128 ms @ 500 Hz)
H = 128       # roll-out 时域 (256 ms)
ENC_DIM = 128
BOTTLENECK = 8   # ImplicitWM 的隐式模式数


# ============================================================
# 数据: 窗口 (K 帧历史 → H 帧未来)
# ============================================================
def load_trajectories(data_dir):
    trajs = []
    for f in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(f)
        T = d["forces"].shape[0]
        if T < K + H:
            continue
        trajs.append({
            "force": d["forces"].astype(np.float32),          # (T,3)
            "pusher": d["pusher_x"].astype(np.float32),       # (T,)
            "mode": d["contact_states_gt"].astype(np.int64),  # (T,)
            "friction": float(d["friction_scale"]),
            "file": f.name,
        })
    return trajs


def make_windows(trajs, rng, n_per_traj=5):
    xs, y_modes, y_forces = [], [], []
    for t in trajs:
        T = len(t["mode"])
        starts = rng.integers(0, T - K - H, size=n_per_traj)
        feat = np.concatenate([
            t["force"],
            t["pusher"][:, None],
            np.full((T, 1), t["friction"], dtype=np.float32),
        ], axis=1)                                            # (T,5)
        for s in starts:
            xs.append(feat[s:s + K])                          # (K,5)
            y_modes.append(t["mode"][s + K:s + K + H])        # (H,)
            y_forces.append(t["force"][s + K:s + K + H])      # (H,3)
    return (np.stack(xs), np.stack(y_modes), np.stack(y_forces))


def split_by_friction(trajs, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    in_dist = [trajs[i] for i in idx if trajs[i]["friction"] <= 1.0]
    ood = [trajs[i] for i in idx if trajs[i]["friction"] > 1.0]
    rng2 = np.random.default_rng(seed + 1)
    p = rng2.permutation(len(in_dist))
    n_train = int(len(in_dist) * 0.8)
    train = [in_dist[i] for i in p[:n_train]]
    test_id = [in_dist[i] for i in p[n_train:]]
    return train, test_id, ood


# ============================================================
# 模型
# ============================================================
class Encoder(nn.Module):
    """K 帧历史 → 初始世界状态 h0"""
    def __init__(self, in_dim=5, hdim=ENC_DIM):
        super().__init__()
        self.gru = nn.GRU(in_dim, hdim, batch_first=True)

    def forward(self, x):        # (B,K,5)
        _, h = self.gru(x)
        return h[-1]             # (B,hdim)


class EventWM(nn.Module):
    """显式事件世界模型: 动力学作用在离散模式上 (事件级监督)"""
    def __init__(self, num_modes=3, hdim=ENC_DIM):
        super().__init__()
        self.enc = Encoder()
        self.head = nn.Linear(hdim, num_modes)
        self.emb = nn.Embedding(num_modes, hdim)
        self.cell = nn.GRUCell(hdim, hdim)
        self.num_modes = num_modes
        self.ss_prob = 0.0     # scheduled sampling: 用自身预测作下一步输入的概率

    def forward(self, x, gt_modes=None):
        h = self.enc(x)                          # (B,hdim)
        logits = []
        mode = None
        for k in range(H):
            l = self.head(h)                     # (B,C)
            logits.append(l)
            if self.training:
                if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                    mode = l.detach().argmax(-1)
                else:
                    mode = gt_modes[:, k]
            else:
                mode = l.argmax(-1)
            h = self.cell(self.emb(mode), h)
        return torch.stack(logits, dim=1)         # (B,H,C)


class ForceWM(nn.Module):
    """连续力世界模型: 动力学作用在连续力上 (MSE, 传统路径)"""
    def __init__(self, hdim=ENC_DIM):
        super().__init__()
        self.enc = Encoder()
        self.head = nn.Linear(hdim, 3)
        self.cell = nn.GRUCell(hdim + 3, hdim)
        self.ss_prob = 0.0

    def forward(self, x, gt_forces=None):
        h = self.enc(x)
        preds = []
        f = torch.zeros(x.shape[0], 3, device=x.device)
        for k in range(H):
            f = self.head(h)
            preds.append(f)
            if self.training and gt_forces is not None:
                if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                    f_in = f.detach()
                else:
                    f_in = gt_forces[:, k]
            else:
                f_in = f
            h = self.cell(torch.cat([h, f_in], -1), h)
        return torch.stack(preds, dim=1)         # (B,H,3)


class ImplicitWM(nn.Module):
    """隐式模式世界模型: 无事件标签, 8 类 Gumbel 潜变量 + 力重构 (PRISM-WM 式)"""
    def __init__(self, hdim=ENC_DIM, n_modes=BOTTLENECK):
        super().__init__()
        self.enc = Encoder()
        self.z_proj = nn.Linear(hdim, n_modes)
        self.dec = nn.Linear(n_modes, 3)
        self.cell = nn.GRUCell(hdim + n_modes, hdim)
        self.n_modes = n_modes

    def forward(self, x, gt_forces=None, hard=False):
        h = self.enc(x)
        preds = []
        z = torch.zeros(x.shape[0], self.n_modes, device=x.device)
        for k in range(H):
            logits = self.z_proj(h)
            z = F.gumbel_softmax(logits, tau=1.0, hard=(hard and not self.training))
            f = self.dec(z)
            preds.append(f)
            h = self.cell(torch.cat([h, z], -1), h)
        return torch.stack(preds, dim=1)         # (B,H,3)


def build_model(name):
    if name == "event_wm":
        return EventWM()
    if name == "force_wm":
        return ForceWM()
    if name == "implicit_wm":
        return ImplicitWM()
    raise ValueError(name)

# ============================================================
# 训练
# ============================================================
def make_anchored_train(trajs, rng, n_per_traj=6):
    """转移锚定训练窗口: 随机视界 ∈ [16,128], 让模型学习滑移时机"""
    xs, ym, yf = [], [], []
    for t in trajs:
        T = len(t["mode"])
        feat = np.concatenate([
            t["force"],
            t["pusher"][:, None],
            np.full((T, 1), t["friction"], dtype=np.float32),
        ], axis=1)
        trans = [k for k in range(1, T) if t["mode"][k - 1] == 1 and t["mode"][k] == 2]
        n_traj = 0
        for t_star in trans:
            horizon = int(rng.integers(16, 129))
            s = t_star - horizon - K
            if s < 0 or s + K + H > T:
                continue
            xs.append(feat[s:s + K])
            ym.append(t["mode"][s + K:s + K + H])
            yf.append(t["force"][s + K:s + K + H])
            n_traj += 1
            if n_traj >= n_per_traj:
                break
    if not xs:
        return (np.zeros((0, K, 5), np.float32), np.zeros((0, H), np.int64),
                np.zeros((0, H, 3), np.float32))
    return np.stack(xs), np.stack(ym), np.stack(yf)


def train_model(model, xs, y_modes, y_forces, epochs=40, lr=1e-3, batch=64,
                seed=0, class_weights=None, inv_reg=False, inv_strength=1.0):
    """训练; inv_reg=True 时给 EventWM 加训练中干预正则:
    高摩擦输入下 'k 帧内已滑移' 累积概率不得高于真实摩擦输入 (随机占优 hinge),
    在训练期就把 guard 条件的因果方向刻进编码器。"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(xs), torch.from_numpy(y_modes), torch.from_numpy(y_forces))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    is_event = isinstance(model, EventWM)
    max_ss = 1.0 if is_event else 0.5
    for ep in range(epochs):
        if hasattr(model, "ss_prob"):
            model.ss_prob = min(max_ss, (ep + 1) / epochs * max_ss)  # scheduled sampling 升温
        model.train()
        tot, n = 0.0, 0
        for xb, mb, fb in dl:
            opt.zero_grad()
            if is_event:
                out = model(xb, gt_modes=mb)
                if class_weights is not None:
                    loss = F.cross_entropy(out.reshape(-1, 3), mb.reshape(-1),
                                           weight=class_weights,
                                           label_smoothing=0.05)
                else:
                    loss = F.cross_entropy(out.reshape(-1, 3), mb.reshape(-1),
                                           label_smoothing=0.05)
                if inv_reg:
                    # 干预正则 v2: do(摩擦×1.5) → 滑移不得更早
                    # v1 负结果: 非对称 hinge 使"永不预测滑移"成为捷径 (evF1 全 0)。
                    # v2 修正: 仅在 GT 含转移的窗口施加 (CE 已强制这些窗口预测滑移,
                    #   hinge 只塑造摩擦依赖性, 无零代价逃逸)。
                    xb_cf = xb.clone()
                    xb_cf[:, :, 4] = xb_cf[:, :, 4] * 1.5
                    out_cf = model(xb_cf, gt_modes=mb)
                    p = F.softmax(out, -1)[:, :, 2]        # P(mode=SLIP)
                    p_cf = F.softmax(out_cf, -1)[:, :, 2]
                    cdf = 1.0 - torch.cumprod(1.0 - p, dim=1)
                    cdf_cf = 1.0 - torch.cumprod(1.0 - p_cf, dim=1)
                    viol = F.relu(cdf_cf - cdf)
                    mask = torch.zeros(len(mb), dtype=torch.bool)
                    for j in range(len(mb)):
                        g = mb[j]
                        mask[j] = any(g[k - 1] == 1 and g[k] == 2
                                      for k in range(1, len(g)))
                    if mask.any():
                        reg = viol[mask].mean()
                    else:
                        reg = torch.zeros(())
                    loss = loss + inv_strength * reg
            else:
                out = model(xb, gt_forces=fb)
                loss = F.mse_loss(out, fb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            n += xb.shape[0]
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  loss={tot / n:.4f}")
    return model


# ============================================================
# 评估
# ============================================================
def threshold_events(force):   # (H,3) numpy → (H,) 模式, 复用纯力规则
    with torch.no_grad():
        f = torch.as_tensor(force, dtype=torch.float32)
        m = force_to_events(f[None], task="stick_slip")[0]
    return m.numpy()


def match_events(pred_t, gt_t, tol=5):
    """事件级 F1: 匹配滑移起始 (1→2 转移), ±tol 帧贪心"""
    def trans(seq):
        out = []
        for k in range(1, len(seq)):
            if seq[k - 1] == 1 and seq[k] == 2:
                out.append(k)
        return out
    p, g = trans(pred_t), trans(gt_t)
    if not p and not g:
        return 1.0, 1.0, 1.0        # 无事件窗口: 双方都正确判定"无滑移"
    matched, used = 0, set()
    for gp in g:
        for i, pp in enumerate(p):
            if i not in used and abs(pp - gp) <= tol:
                matched += 1
                used.add(i)
                break
    prec = matched / len(p) if p else 0.0
    rec = matched / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def slip_onset(mode_seq):
    for k, m in enumerate(mode_seq):
        if m == 2:
            return k
    return len(mode_seq)


def has_transition(seq):
    """窗口内是否含滑移起始转移 (1→2)"""
    for k in range(1, len(seq)):
        if seq[k - 1] == 1 and seq[k] == 2:
            return True
    return False


def evaluate(model, xs, y_modes, y_forces, friction=None, batch=64):
    model.eval()
    failures, f1s, mses = [], [], []
    pred_onsets, gt_trans = [], []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = torch.from_numpy(xs[i:i + batch])
            if friction is not None:                 # 干预: 覆盖摩擦通道
                fr = np.asarray(friction, dtype=np.float32)[i:i + batch]
                xb = xb.clone()
                xb[:, :, 4] = torch.from_numpy(fr)[:, None]
            mb = y_modes[i:i + batch]
            if isinstance(model, EventWM):
                out = model(xb)                      # (B,H,C)
                pm = out.argmax(-1).numpy()
            elif isinstance(model, ForceWM):
                out = model(xb).numpy()
                pm = np.stack([threshold_events(f) for f in out])
                mses.append(np.mean((out - y_forces[i:i + batch]) ** 2))
            else:                                    # ImplicitWM
                out = model(xb, hard=True).numpy()
                pm = np.stack([threshold_events(f) for f in out])
                mses.append(np.mean((out - y_forces[i:i + batch]) ** 2))
            for j in range(len(xb)):
                gt = mb[j]
                k = 0
                while k < H and pm[j, k] == gt[k]:
                    k += 1
                failures.append(k)
                f1, _, _ = match_events(pm[j], gt)
                f1s.append(f1)
                pred_onsets.append(slip_onset(pm[j]))
                gt_trans.append(has_transition(gt))
    out = {
        "first_failure_mean": float(np.mean(failures)),
        "first_failure_median": float(np.median(failures)),
        "event_f1": float(np.mean(f1s)),
        "n_windows": int(len(failures)),
    }
    # 转移窗口限定指标 (GT 窗口内含 1→2 滑移起始转移才有判别力)
    slip_idx = [j for j, f in enumerate(gt_trans) if f]
    if slip_idx:
        out["n_slip_windows"] = int(len(slip_idx))
        out["first_failure_slip"] = float(np.mean([failures[j] for j in slip_idx]))
        out["event_f1_slip"] = float(np.mean([f1s[j] for j in slip_idx]))
    else:
        out["n_slip_windows"] = 0
    if mses:
        out["force_mse"] = float(np.mean(mses))
    return out, pred_onsets, gt_trans


def intervention(model, xs, y_modes, y_forces, f_true, f_counter, batch=64):
    """摩擦 +50% 干预: 滑移起始是否单调后移 (正确物理方向, 仅限 GT 含转移窗口)

    严谨口径: a==b (无位移) 计 neutral, 不算 correct; 并报告平均位移与方向分布。
    """
    _, o1, g1 = evaluate(model, xs, y_modes, y_forces, friction=f_true, batch=batch)
    _, o2, g2 = evaluate(model, xs, y_modes, y_forces, friction=f_counter, batch=batch)
    correct = wrong = neutral = 0
    shifts = []
    for a, b, g in zip(o1, o2, g1):
        if not g:                          # GT 本窗口无滑移转移: 不计
            continue
        if a >= len(y_modes[0]) and b >= len(y_modes[0]):
            neutral += 1
            continue
        shifts.append(b - a)
        if b > a:
            correct += 1
        elif b < a:
            wrong += 1
        else:
            neutral += 1                   # 无位移 = 摩擦未被使用, 不算正确
    total = correct + wrong
    return {
        "n_gt_slip_windows": correct + wrong + neutral,
        "direction_correct": correct,
        "direction_wrong": wrong,
        "direction_neutral": neutral,
        "direction_rate": float(correct / total) if total else None,
        "mean_shift_frames": float(np.mean(shifts)) if shifts else None,
        "n_positive_shift": int(sum(1 for s in shifts if s > 0)),
        "n_negative_shift": int(sum(1 for s in shifts if s < 0)),
        "n_zero_shift": int(sum(1 for s in shifts if s == 0)),
    }
# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["event_wm", "force_wm", "implicit_wm"])
    ap.add_argument("--data-dir", default="data/stick_slip")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints_wm")
    ap.add_argument("--anchored-train", action="store_true",
                    help="训练窗口混合转移锚定窗口 + SLIP 类加权")
    ap.add_argument("--inv-reg", action="store_true",
                    help="训练中干预正则 (do(摩擦×1.5) 随机占优 hinge)")
    ap.add_argument("--inv-strength", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    result_path = out_dir / f"{args.model}_results.json"

    trajs = load_trajectories(args.data_dir)
    train_t, test_id, test_ood = split_by_friction(trajs)
    print(f"[{args.model}] trajs: {len(trajs)} "
          f"(train={len(train_t)}, test_id={len(test_id)}, test_ood={len(test_ood)})")

    rng = np.random.default_rng(args.seed)
    xs, ym, yf = make_windows(train_t, rng, n_per_traj=5)
    class_weights = None
    if args.anchored_train:
        xa, ya, yfa = make_anchored_train(train_t, rng)
        print(f"  anchored train windows: {len(xa)}")
        xs = np.concatenate([xs, xa])
        ym = np.concatenate([ym, ya])
        yf = np.concatenate([yf, yfa])
        # SLIP 类加权 (逆频率归一化, 提高稀有滑移类权重)
        counts = np.bincount(ym.reshape(-1), minlength=3).astype(float)
        counts[counts == 0] = 1e-6
        w = counts.sum() / (3 * counts)
        class_weights = torch.from_numpy(w.astype(np.float32))
        print(f"  class counts={counts} weights={w}")
    rng2 = np.random.default_rng(args.seed + 10)
    xs_id, ym_id, yf_id = make_windows(test_id, rng2, n_per_traj=3)
    xs_ood, ym_ood, yf_ood = make_windows(test_ood, rng2, n_per_traj=3)
    print(f"  windows: train={len(xs)}, id={len(xs_id)}, ood={len(xs_ood)}")

    model = build_model(args.model)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_param}")
    train_model(model, xs, ym, yf, epochs=args.epochs, seed=args.seed,
                class_weights=class_weights, inv_reg=args.inv_reg,
                inv_strength=args.inv_strength)

    def friction_of(traj_list, n):
        vals = []
        for t in traj_list:
            vals += [t["friction"]] * n
        return np.array(vals, dtype=np.float32)

    f_id = friction_of(test_id, 3)
    f_ood = friction_of(test_ood, 3)

    res = {"model": args.model, "seed": args.seed, "params": n_param}
    for name, (xw, yw, yfw) in {
        "id": (xs_id, ym_id, yf_id),
        "ood": (xs_ood, ym_ood, yf_ood),
    }.items():
        r, _, _ = evaluate(model, xw, yw, yfw, friction=None)
        res[name] = r
        print(f"  [{name}] {r}")

    # 干预: 真实摩擦 → +50%
    res["intervention"] = {
        "id": intervention(model, xs_id, ym_id, yf_id, f_id, f_id * 1.5),
        "ood": intervention(model, xs_ood, ym_ood, yf_ood, f_ood, f_ood * 1.5),
    }
    print(f"  [intervention] {res['intervention']}")

    suffix = ("_anchored" if args.anchored_train else "")
    if args.inv_reg:
        suffix += "_inv2"    # v2: GT转移窗口限定的支配正则 (v1 负结果见 git 历史)
    suffix += f"_s{args.seed}"
    result_path = out_dir / f"{args.model}{suffix}_results.json"
    result_path.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    torch.save(model.state_dict(), out_dir / f"{args.model}{suffix}.pt")
    print(f"saved -> {result_path}")


if __name__ == "__main__":
    main()

