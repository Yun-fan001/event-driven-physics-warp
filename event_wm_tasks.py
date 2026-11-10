"""事件序列自回归世界模型 · 任务扩展版 (peg_in_hole / stack_block)

与 event_wm.py 相同三路对照 (EventWM / ForceWM / ImplicitWM), 任务参数化:
- peg_in_hole:  2 模式, 锚定事件 = 接触建立 (0→1); 干预 = do(peg_z + 2mm) → 接触更晚
- stack_block:  3 模式, 锚定事件 = 抓取 (0→1);   干预 = do(gripper_z + 2mm) → 抓取更晚
- (stick_slip 亦支持, 锚定事件 = 滑移 1→2; 干预 = do(摩擦×1.5))
评估: 转移锚定视界 (16/32/64/96) + 反事实干预 (严格口径: a==b 计 neutral)
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

K = 64
H = 128
ENC_DIM = 128
BOTTLENECK = 8

TASK_SPEC = {
    "stick_slip": {
        "num_modes": 3, "proprio_keys": ["pusher_x"],
        "transitions": [(1, 2)], "to_mode": 2,
        "intervene": {"channel": 4, "kind": "scale", "value": 1.5},
        "friction_split": True,
        "gt_name": "slip", "cond_shift_note": "do(friction x1.5)",
    },
    "peg_in_hole": {
        "num_modes": 2, "proprio_keys": ["peg_z"],
        "transitions": [(0, 1)], "to_mode": 1,
        "intervene": {"channel": 3, "kind": "shift", "value": 0.002},
        "friction_split": False,
        "gt_name": "contact", "cond_shift_note": "do(peg_z +2mm)",
    },
    "stack_block": {
        "num_modes": 3, "proprio_keys": ["gripper_x", "gripper_z"],
        "transitions": [(0, 1)], "to_mode": 1,
        "intervene": {"channel": 4, "kind": "shift", "value": 0.002},
        "friction_split": False,
        "gt_name": "grip", "cond_shift_note": "do(gripper_z +2mm)",
    },
}


def load_trajectories(data_dir, task):
    spec = TASK_SPEC[task]
    trajs = []
    for f in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(f)
        T = d["forces"].shape[0]
        if T < K + H:
            continue
        trajs.append({
            "force": d["forces"].astype(np.float32),
            "mode": d["contact_states_gt"].astype(np.int64),
            "friction": float(d["friction_scale"]),
            "file": f.name,
            **{k: d[k].astype(np.float32) for k in spec["proprio_keys"]},
        })
    return trajs


def make_feat(t, task):
    spec = TASK_SPEC[task]
    T = len(t["mode"])
    parts = [t["force"]]
    for k in spec["proprio_keys"]:
        parts.append(t[k][:, None])
    parts.append(np.full((T, 1), t["friction"], dtype=np.float32))
    return np.concatenate(parts, axis=1)          # (T, 3+len(proprio)+1)


def find_transitions(mode, fm, tm):
    out = []
    for t in range(1, len(mode)):
        if mode[t - 1] == fm and mode[t] == tm:
            out.append(t)
    return out


def has_transition(seq, fm, tm):
    for k in range(1, len(seq)):
        if seq[k - 1] == fm and seq[k] == tm:
            return True
    return False


def event_onset(mode_seq, to_mode):
    for k, m in enumerate(mode_seq):
        if m == to_mode:
            return k
    return len(mode_seq)


def make_windows(trajs, task, rng, n_per_traj=5):
    xs, y_modes, y_forces = [], [], []
    for t in trajs:
        T = len(t["mode"])
        feat = make_feat(t, task)
        starts = rng.integers(0, T - K - H, size=n_per_traj)
        for s in starts:
            xs.append(feat[s:s + K])
            y_modes.append(t["mode"][s + K:s + K + H])
            y_forces.append(t["force"][s + K:s + K + H])
    return (np.stack(xs), np.stack(y_modes), np.stack(y_forces))


def make_anchored_train(trajs, task, rng, n_per_traj=6):
    spec = TASK_SPEC[task]
    xs, ym, yf = [], [], []
    for t in trajs:
        T = len(t["mode"])
        feat = make_feat(t, task)
        trans = find_transitions(t["mode"], *spec["transitions"][0])
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
        return (np.zeros((0, K, 9), np.float32), np.zeros((0, H), np.int64),
                np.zeros((0, H, 3), np.float32))
    return np.stack(xs), np.stack(ym), np.stack(yf)


def split_trajectories(trajs, task, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    spec = TASK_SPEC[task]
    if spec["friction_split"]:
        in_dist = [trajs[i] for i in idx if trajs[i]["friction"] <= 1.0]
        ood = [trajs[i] for i in idx if trajs[i]["friction"] > 1.0]
        rng2 = np.random.default_rng(seed + 1)
        p = rng2.permutation(len(in_dist))
        n_train = int(len(in_dist) * 0.8)
        return [in_dist[i] for i in p[:n_train]], [in_dist[i] for i in p[n_train:]], ood
    n_train = int(len(trajs) * 0.8)
    return trajs[:n_train], trajs[n_train:], []


def make_anchored_eval(trajs, task, horizon, max_per_traj=4):
    spec = TASK_SPEC[task]
    xs, ym, yf, fric = [], [], [], []
    for t in trajs:
        T = len(t["mode"])
        feat = make_feat(t, task)
        n_traj = 0
        for t_star in find_transitions(t["mode"], *spec["transitions"][0]):
            s = t_star - horizon - K
            if s < 0 or s + K + H > T:
                continue
            xs.append(feat[s:s + K])
            ym.append(t["mode"][s + K:s + K + H])
            yf.append(t["force"][s + K:s + K + H])
            fric.append(t["friction"])
            n_traj += 1
            if n_traj >= max_per_traj:
                break
    if not xs:
        return (np.zeros((0, K, 9), np.float32), np.zeros((0, H), np.int64),
                np.zeros((0, H, 3), np.float32), np.zeros(0, np.float32))
    return (np.stack(xs), np.stack(ym), np.stack(yf),
            np.array(fric, dtype=np.float32))


# ============================================================
# 模型 (同 event_wm.py, EventWM 模式数参数化)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, in_dim, hdim=ENC_DIM):
        super().__init__()
        self.gru = nn.GRU(in_dim, hdim, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class EventWM(nn.Module):
    def __init__(self, num_modes=3, hdim=ENC_DIM, in_dim=5):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.head = nn.Linear(hdim, num_modes)
        self.emb = nn.Embedding(num_modes, hdim)
        self.cell = nn.GRUCell(hdim, hdim)
        self.num_modes = num_modes
        self.ss_prob = 0.0

    def forward(self, x, gt_modes=None):
        h = self.enc(x)
        logits = []
        mode = None
        for k in range(H):
            l = self.head(h)
            logits.append(l)
            if self.training:
                if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                    mode = l.detach().argmax(-1)
                else:
                    mode = gt_modes[:, k]
            else:
                mode = l.argmax(-1)
            h = self.cell(self.emb(mode), h)
        return torch.stack(logits, dim=1)


class ForceWM(nn.Module):
    def __init__(self, hdim=ENC_DIM, in_dim=5):
        super().__init__()
        self.enc = Encoder(in_dim)
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
        return torch.stack(preds, dim=1)


class ImplicitWM(nn.Module):
    def __init__(self, hdim=ENC_DIM, n_modes=BOTTLENECK, in_dim=5):
        super().__init__()
        self.enc = Encoder(in_dim)
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
        return torch.stack(preds, dim=1)


def build_model(name, num_modes, in_dim=5):
    if name == "event_wm":
        return EventWM(num_modes, in_dim=in_dim)
    if name == "force_wm":
        return ForceWM(in_dim=in_dim)
    if name == "implicit_wm":
        return ImplicitWM(in_dim=in_dim)
    raise ValueError(name)


# ============================================================
# 训练
# ============================================================
def train_model(model, xs, y_modes, y_forces, epochs=40, lr=1e-3, batch=64,
                seed=0, class_weights=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(xs), torch.from_numpy(y_modes), torch.from_numpy(y_forces))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    is_event = isinstance(model, EventWM)
    max_ss = 1.0 if is_event else 0.5
    for ep in range(epochs):
        if hasattr(model, "ss_prob"):
            model.ss_prob = min(max_ss, (ep + 1) / epochs * max_ss)
        model.train()
        tot, n = 0.0, 0
        for xb, mb, fb in dl:
            opt.zero_grad()
            if is_event:
                out = model(xb, gt_modes=mb)
                loss = F.cross_entropy(out.reshape(-1, model.num_modes),
                                       mb.reshape(-1),
                                       weight=class_weights,
                                       label_smoothing=0.05)
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
def threshold_events(force, task):
    with torch.no_grad():
        f = torch.as_tensor(force, dtype=torch.float32)
        m = force_to_events(f[None], task=task)[0]
    return m.numpy()


def match_events(pred_t, gt_t, fm, tm, tol=5):
    def trans(seq):
        return [k for k in range(1, len(seq))
                if seq[k - 1] == fm and seq[k] == tm]
    p, g = trans(pred_t), trans(gt_t)
    if not p and not g:
        return 1.0, 1.0, 1.0
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


def evaluate(model, xs, y_modes, y_forces, task, override=None, batch=64):
    spec = TASK_SPEC[task]
    fm, tm = spec["transitions"][0]
    model.eval()
    failures, f1s, mses = [], [], []
    pred_onsets, gt_trans = [], []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = torch.from_numpy(xs[i:i + batch])
            if override is not None:                 # 干预: 覆盖指定通道
                ch, kind, val = override
                xb = xb.clone()
                if kind == "scale":
                    xb[:, :, ch] = xb[:, :, ch] * val
                else:
                    xb[:, :, ch] = xb[:, :, ch] + val
            mb = y_modes[i:i + batch]
            if isinstance(model, EventWM):
                out = model(xb)
                pm = out.argmax(-1).numpy()
            elif isinstance(model, ForceWM):
                out = model(xb).numpy()
                pm = np.stack([threshold_events(f, task) for f in out])
                mses.append(np.mean((out - y_forces[i:i + batch]) ** 2))
            else:
                out = model(xb, hard=True).numpy()
                pm = np.stack([threshold_events(f, task) for f in out])
                mses.append(np.mean((out - y_forces[i:i + batch]) ** 2))
            for j in range(len(xb)):
                gt = mb[j]
                k = 0
                while k < H and pm[j, k] == gt[k]:
                    k += 1
                failures.append(k)
                f1, _, _ = match_events(pm[j], gt, fm, tm)
                f1s.append(f1)
                pred_onsets.append(event_onset(pm[j], tm))
                gt_trans.append(has_transition(gt, fm, tm))
    out = {
        "first_failure_mean": float(np.mean(failures)),
        "first_failure_median": float(np.median(failures)),
        "event_f1": float(np.mean(f1s)),
        "n_windows": int(len(failures)),
    }
    slip_idx = [j for j, f in enumerate(gt_trans) if f]
    if slip_idx:
        out["n_event_windows"] = int(len(slip_idx))
        out["first_failure_event"] = float(np.mean([failures[j] for j in slip_idx]))
        out["event_f1_event"] = float(np.mean([f1s[j] for j in slip_idx]))
    else:
        out["n_event_windows"] = 0
    if mses:
        out["force_mse"] = float(np.mean(mses))
    return out, pred_onsets, gt_trans


def intervention(model, xs, y_modes, y_forces, task, ov_true, ov_cf, batch=64):
    """干预 (严格口径): a==b 计 neutral; 报告 mean_shift 与 +/−/0 分布"""
    spec = TASK_SPEC[task]
    fm, tm = spec["transitions"][0]
    _, o1, g1 = evaluate(model, xs, y_modes, y_forces, task, override=ov_true, batch=batch)
    _, o2, g2 = evaluate(model, xs, y_modes, y_forces, task, override=ov_cf, batch=batch)
    correct = wrong = neutral = 0
    shifts = []
    for a, b, g in zip(o1, o2, g1):
        if not g:
            continue
        if a >= H and b >= H:
            neutral += 1
            continue
        shifts.append(b - a)
        if b > a:
            correct += 1
        elif b < a:
            wrong += 1
        else:
            neutral += 1
    total = correct + wrong
    return {
        "n_gt_event_windows": correct + wrong + neutral,
        "direction_correct": correct,
        "direction_wrong": wrong,
        "direction_neutral": neutral,
        "direction_rate": float(correct / total) if total else None,
        "mean_shift_frames": float(np.mean(shifts)) if shifts else None,
        "n_positive_shift": int(sum(1 for s in shifts if s > 0)),
        "n_negative_shift": int(sum(1 for s in shifts if s < 0)),
        "n_zero_shift": int(sum(1 for s in shifts if s == 0)),
    }

    return (np.stack(xs), np.stack(ym), np.stack(yf),
            np.array(fric, dtype=np.float32))
# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=list(TASK_SPEC.keys()))
    ap.add_argument("--model", required=True,
                    choices=["event_wm", "force_wm", "implicit_wm"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchored-train", action="store_true",
                    help="训练窗口混合转移锚定窗口 + 稀有类加权")
    ap.add_argument("--out-dir", default="checkpoints_wm_tasks")
    args = ap.parse_args()

    spec = TASK_SPEC[args.task]
    data_dir = Path("data") / args.task
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    suffix = ("_anchored" if args.anchored_train else "") + f"_s{args.seed}"
    tag = f"{args.model}_{args.task}{suffix}"

    trajs = load_trajectories(data_dir, args.task)
    train_t, test_t, test_ood = split_trajectories(trajs, args.task)
    print(f"[{args.task}/{args.model}] trajs={len(trajs)} "
          f"(train={len(train_t)}, test={len(test_t)}, ood={len(test_ood)})")

    rng = np.random.default_rng(args.seed)
    xs, ym, yf = make_windows(train_t, args.task, rng, n_per_traj=3)
    class_weights = None
    if args.anchored_train:
        xa, ya, yfa = make_anchored_train(train_t, args.task, rng)
        print(f"  anchored train windows: {len(xa)}")
        if len(xa):
            xs = np.concatenate([xs, xa])
            ym = np.concatenate([ym, ya])
            yf = np.concatenate([yf, yfa])
            counts = np.bincount(ym.reshape(-1),
                                 minlength=spec["num_modes"]).astype(float)
            counts[counts == 0] = 1e-6
            w = counts.sum() / (spec["num_modes"] * counts)
            class_weights = torch.from_numpy(w.astype(np.float32))
            print(f"  class counts={counts} weights={w}")
    print(f"  windows: train={len(xs)}")

    in_dim = 3 + len(spec["proprio_keys"]) + 1
    model = build_model(args.model, spec["num_modes"], in_dim=in_dim)
    print(f"  params: {sum(p.numel() for p in model.parameters())} "
          f"(in_dim={in_dim})")
    train_model(model, xs, ym, yf, epochs=args.epochs, seed=args.seed,
                class_weights=class_weights)

    # 干预参数
    iv = spec["intervene"]
    ov_true = (iv["channel"], iv["kind"], 1.0 if iv["kind"] == "scale" else 0.0)
    ov_cf = (iv["channel"], iv["kind"], iv["value"])

    res = {"model": args.model, "task": args.task, "seed": args.seed}

    # 随机窗口评估
    rng2 = np.random.default_rng(args.seed + 10)
    xs_t, ym_t, yf_t = make_windows(test_t, args.task, rng2, n_per_traj=3)
    r, _, _ = evaluate(model, xs_t, ym_t, yf_t, args.task)
    res["random_id"] = r
    print(f"  [random_id] {r}")
    if len(test_ood):
        xs_o, ym_o, yf_o = make_windows(test_ood, args.task, rng2, n_per_traj=3)
        r, _, _ = evaluate(model, xs_o, ym_o, yf_o, args.task)
        res["random_ood"] = r
        print(f"  [random_ood] {r}")

    # 转移锚定评估 + 干预
    for hz in [16, 32, 64, 96]:
        xa, ya, yfa, fr = make_anchored_eval(test_t, args.task, hz)
        if len(xa) == 0:
            res[f"id_h{hz}"] = {"n": 0}
            continue
        r, _, _ = evaluate(model, xa, ya, yfa, args.task)
        iv_res = intervention(model, xa, ya, yfa, args.task, ov_true, ov_cf)
        r["intervention"] = iv_res
        r["n"] = len(xa)
        res[f"id_h{hz}"] = r
        print(f"  [id h{hz}] n={len(xa)} FF={r['first_failure_mean']:.1f} "
              f"evF1={r['event_f1']:.3f} iv_rate={iv_res.get('direction_rate')}")
        if len(test_ood):
            xa_o, ya_o, yfa_o, fr_o = make_anchored_eval(test_ood, args.task, hz)
            if len(xa_o):
                r, _, _ = evaluate(model, xa_o, ya_o, yfa_o, args.task)
                iv_res = intervention(model, xa_o, ya_o, yfa_o, args.task,
                                      ov_true, ov_cf)
                r["intervention"] = iv_res
                r["n"] = len(xa_o)
                res[f"ood_h{hz}"] = r
                print(f"  [ood h{hz}] n={len(xa_o)} FF={r['first_failure_mean']:.1f} "
                      f"evF1={r['event_f1']:.3f} "
                      f"iv_rate={iv_res.get('direction_rate')}")

    out_path = out_dir / f"{tag}_results.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    torch.save(model.state_dict(), out_dir / f"{tag}.pt")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

