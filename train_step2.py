"""
Step 2 训练脚本
支持 3 个任务 × 6 个模型 的实验矩阵:
  force_mlp    — 基线A: 连续力回归 + 固定阈值判定事件
  vision_only  — 基线B: 纯视觉事件分类
  force_tcn    — 消融: 力觉-only TCN (无视觉)
  event_tcn    — 主模型: 多模态 TCN (视觉+力觉+本体)
  force_lstm   — 降级消融: 力觉-only Bi-LSTM (TCN 不稳定时)
  event_lstm   — 降级方案: 多模态 Bi-LSTM (TCN 不稳定时)

额外功能: 数据效率实验 (--data-fraction), 噪声鲁棒性实验 (--force-noise-std, --time-shift-ms)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ContactEventDataset
from models import ForceMLP, VisionOnly, EventTCN, EventLSTM

# ============================================================
# 任务配置
# ============================================================
TASK_CONFIG = {
    "stick_slip": {
        "num_classes": 3,
        "force_dim": 6,      # 力(3) + 速度(3)
        "proprio_dim": 1,    # pusher_x
    },
    "peg_in_hole": {
        "num_classes": 2,
        "force_dim": 6,
        "proprio_dim": 1,    # peg_z
    },
    "stack_block": {
        "num_classes": 3,
        "force_dim": 6,
        "proprio_dim": 2,    # gripper_x, gripper_z
    },
}

MODEL_CHOICES = ["force_mlp", "vision_only", "force_tcn", "event_tcn", "force_lstm", "event_lstm"]

# ============================================================
# Force-MLP 基线核心: 预测连续力 → 固定阈值 → 离散事件
# 严格模拟「先恢复物理量 → 再推理」的传统路径。
# 训练: 纯 MSE 力回归（不学分类器）。
# 评估: 通过确定性阈值规则将预测力转为事件类别。
# 阈值与各环境 GT 判定逻辑一致 (见 env_*.py)。
# ============================================================
def force_to_events(
    force_pred: torch.Tensor,    # (B, T, 3)  预测的三维力
    force_input: torch.Tensor,   # (B, T, 6)  原始力+速度 [fx,fy,fz,vx,vy,vz]
    task: str,
) -> torch.Tensor:
    """
    固定阈值转为离散事件标签。
    Task 1 (stick_slip):  |F_xy| > 0.5 → Contact; |v_x| < 0.01 → Stick else Slip
    Task 2 (peg_in_hole): |F| > 1.0 → Contact
    Task 3 (stack_block): |F| > 2.0 → Grip (简化版, 仅用力)
    Returns: (B, T) LongTensor
    """
    if task == "stick_slip":
        force_mag = torch.norm(force_pred[:, :, :2], dim=-1)
        vel_x = force_input[:, :, 3]
        in_contact = force_mag > 0.5
        is_stick = torch.abs(vel_x) < 0.01
        events = torch.where(
            ~in_contact,
            torch.zeros_like(force_mag, dtype=torch.long),                 # 0=NO_CONTACT
            torch.where(is_stick,
                        torch.ones_like(force_mag, dtype=torch.long),      # 1=STICK
                        2 * torch.ones_like(force_mag, dtype=torch.long)), # 2=SLIP
        )
    elif task == "peg_in_hole":
        force_mag_3d = torch.norm(force_pred, dim=-1)
        in_contact = force_mag_3d > 1.0
        events = in_contact.long()   # 0/1
    elif task == "stack_block":
        force_mag_3d = torch.norm(force_pred, dim=-1)
        events = torch.where(
            force_mag_3d > 2.0,
            torch.ones_like(force_mag_3d, dtype=torch.long),    # 1=GRIP
            torch.zeros_like(force_mag_3d, dtype=torch.long),   # 0=NO_CONTACT
        )
    else:
        raise ValueError(f"未知任务: {task}")
    return events


# ============================================================
# 噪声注入工具 (Step 3 噪声鲁棒性实验)
# ============================================================
def add_force_noise(force: torch.Tensor, noise_std: float) -> torch.Tensor:
    """力觉通道加高斯噪声 (模拟传感器零漂)"""
    if noise_std <= 0:
        return force
    noise = torch.randn_like(force) * noise_std
    return force + noise


def add_time_shift(
    vision: torch.Tensor,
    force: torch.Tensor,
    proprio: torch.Tensor,
    labels: torch.Tensor | None,
    shift_ms: float,
    seq_len: int,
    dt: float = 0.002,  # timestep = 0.002s
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    视觉与力觉之间加入时间偏移，模拟异步采集。
    shift_ms > 0: 视觉滞后力觉; shift_ms < 0: 视觉领先力觉。
    
    统一策略: 仅平移 vision 模态，force/proprio/labels 始终对齐。
    -- 因为事件标签由力觉物理量决定，labels 应与 force 保持配对不变 --
    
    正偏移: vision 向后平移 (roll +shift_frames) → vision(t) 看到旧帧
    负偏移: vision 向前平移 (roll -shift_frames) → vision(t) 看到新帧
    """
    if abs(shift_ms) < 1e-6:
        return vision, force, proprio, labels

    shift_frames = int(round(abs(shift_ms) * 0.001 / dt))
    if shift_frames == 0:
        return vision, force, proprio, labels

    B, T = vision.shape[:2]
    if shift_frames >= T:
        shift_frames = T - 1

    if shift_ms > 0:
        # 视觉滞后力觉: vision(t) 对应 force(t-shift_frames)
        # 将 vision 向后平移，前面用首帧填充（保持对齐语义统一）
        vision = torch.roll(vision, shifts=shift_frames, dims=1)
        vision[:, :shift_frames] = vision[:, shift_frames:shift_frames + 1]
    else:
        # 视觉领先力觉: vision(t) 对应 force(t+shift_frames)
        # 将 vision 向前平移，后面用尾帧填充
        vision = torch.roll(vision, shifts=-shift_frames, dims=1)
        vision[:, -shift_frames:] = vision[:, -shift_frames - 1: -shift_frames]

    # force/proprio/labels 始终对齐，无需调整
    # (事件由 force 物理量决定，labels 与 force 配对不变)
    return vision, force, proprio, labels


# ============================================================
# 模型包装器: 统一接口
# ============================================================
class ModelWrapper(nn.Module):
    """
    统一接口 (vision, force, proprio) → logits / force_pred

    force_mlp:  训练时纯 MSE 力回归（无分类头），eval 时阈值判定
    vision_only: LSTM 视觉分类
    force_tcn:   力觉-only TCN (消融实验)
    event_tcn:   多模态 TCN (完整版)
    force_lstm:  力觉-only Bi-LSTM (TCN 不稳定时的降级消融)
    event_lstm:  多模态 Bi-LSTM (TCN 不稳定时的降级方案)
    """
    def __init__(self, model_name: str, task: str):
        super().__init__()
        cfg = TASK_CONFIG[task]
        self.model_name = model_name
        self.task = task
        self.num_classes = cfg["num_classes"]

        if model_name == "force_mlp":
            # 纯力回归器: 只预测连续力，不学分类
            self.model = ForceMLP(input_dim=cfg["force_dim"])
            # 不添加 classifier —— eval 时用 force_to_events() 阈值判定
        elif model_name == "vision_only":
            self.model = VisionOnly(num_classes=cfg["num_classes"])
        elif model_name == "force_tcn":
            # 力觉-only TCN: EventTCN 去掉视觉编码器分支
            self.model = EventTCN(
                num_classes=cfg["num_classes"],
                vision_feat_dim=0,           # 0 → 跳过视觉编码
                force_dim=cfg["force_dim"],
                proprio_dim=cfg["proprio_dim"],
            )
        elif model_name == "event_tcn":
            self.model = EventTCN(
                num_classes=cfg["num_classes"],
                force_dim=cfg["force_dim"],
                proprio_dim=cfg["proprio_dim"],
            )
        elif model_name == "force_lstm":
            # TCN 不稳定时的降级方案: 力觉-only Bi-LSTM
            self.model = EventLSTM(
                num_classes=cfg["num_classes"],
                vision_feat_dim=0,           # 0 → 跳过视觉编码
                force_dim=cfg["force_dim"],
                proprio_dim=cfg["proprio_dim"],
            )
        elif model_name == "event_lstm":
            # TCN 不稳定时的降级方案: 多模态 Bi-LSTM
            self.model = EventLSTM(
                num_classes=cfg["num_classes"],
                force_dim=cfg["force_dim"],
                proprio_dim=cfg["proprio_dim"],
            )
        else:
            raise ValueError(f"未知模型: {model_name}")

    def forward(self, vision, force, proprio):
        """
        Returns:
            force_mlp:  (force_pred, None)  只有连续力预测
            其他模型:   (logits, None)       分类 logits
        """
        if self.model_name == "force_mlp":
            force_pred = self.model(force)  # (B, T, 3)
            return None, force_pred
        elif self.model_name == "vision_only":
            logits = self.model(vision)
            return logits, None
        else:
            # force_tcn / event_tcn / force_lstm / event_lstm
            # vision 传 None 时由模型内部处理（force-only 模式）
            logits = self.model(vision, force, proprio)
            return logits, None


# ============================================================
# 训练/评估
# ============================================================
def train_epoch(
    model: ModelWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ignore_index: int = -100,
    force_noise_std: float = 0.0,
    time_shift_ms: float = 0.0,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train"):
        vision = batch["vision"].to(device)
        force = batch["force"].to(device)
        proprio = batch["proprio"].to(device)
        labels = batch["label"].to(device)

        # 噪声注入
        force = add_force_noise(force, force_noise_std)
        vision, force, proprio, labels = add_time_shift(
            vision, force, proprio, labels, time_shift_ms,
            seq_len=vision.shape[1],
        )

        optimizer.zero_grad()
        logits, force_pred = model(vision, force, proprio)

        if model.model_name == "force_mlp":
            # 纯 MSE 力回归 —— 不学分类
            force_target = force[:, :, :3]
            loss = F.mse_loss(force_pred, force_target)

            # 训练时的 acc: 通过阈值判定计算
            with torch.no_grad():
                events = force_to_events(force_pred, force, model.task)
                mask = labels != ignore_index
                correct += ((events == labels) & mask).sum().item()
                total += mask.sum().item()
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                labels.reshape(B * T),
                ignore_index=ignore_index,
            )
            mask = labels != ignore_index
            preds = logits.argmax(dim=-1)
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_epoch(
    model: ModelWrapper,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int = -100,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Eval"):
        vision = batch["vision"].to(device)
        force = batch["force"].to(device)
        proprio = batch["proprio"].to(device)
        labels = batch["label"].to(device)

        logits, force_pred = model(vision, force, proprio)

        if model.model_name == "force_mlp":
            # force_mlp: 预测力 → 阈值 → 离散事件 (不加分类头)
            force_target = force[:, :, :3]
            loss = F.mse_loss(force_pred, force_target)
            events = force_to_events(force_pred, force, model.task)
            preds = events
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                labels.reshape(B * T),
                ignore_index=ignore_index,
            )
            preds = logits.argmax(dim=-1)

        total_loss += loss.item()

        mask = labels != ignore_index
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())

    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total if total > 0 else 0.0,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 2 模型训练")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_CHOICES)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="使用数据比例 (用于数据效率实验)")
    # 噪声鲁棒性
    parser.add_argument("--force-noise-std", type=float, default=0.0,
                        help="力觉高斯噪声标准差 (0=不注入)")
    parser.add_argument("--time-shift-ms", type=float, default=0.0,
                        help="视觉-力觉时间偏移 (ms, ±50/±100/±200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    dataset = ContactEventDataset(args.data_dir, task=args.task,
                                   seq_len=args.seq_len)

    # 固定划分: 70% train / 15% val / 15% test
    n_total = len(dataset)
    n_train_full = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train_full - n_val

    if n_val < 1:
        raise RuntimeError(f"数据集太小 (n={n_total})，至少需要 7 条轨迹保证 val 非空")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_idx_full = indices[:n_train_full]
    val_idx = indices[n_train_full:n_train_full + n_val]
    test_idx = indices[n_train_full + n_val:]

    # 数据效率实验
    n_train_used = max(1, int(len(train_idx_full) * args.data_fraction))
    train_idx = train_idx_full[:n_train_used]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # 模型
    model = ModelWrapper(args.model, args.task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    output_dir = Path(args.output_dir) / args.task / args.model
    # 若开启噪声实验则写入子目录
    if args.force_noise_std > 0 or abs(args.time_shift_ms) > 1e-6:
        output_dir = output_dir / f"noise_f{args.force_noise_std}_t{args.time_shift_ms:.0f}ms"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            force_noise_std=args.force_noise_std,
            time_shift_ms=args.time_shift_ms,
        )
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} "
              f"| R: {val_metrics['recall']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best.pt")
            print(f"✅ 保存最佳模型 (F1={best_f1:.4f})")

    # 最终测试
    checkpoint = torch.load(output_dir / "best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = eval_epoch(model, test_loader, device)
    print(f"\n🎯 测试集结果: Acc={test_metrics['accuracy']:.4f}, "
          f"F1={test_metrics['f1']:.4f}, "
          f"P={test_metrics['precision']:.4f}, "
          f"R={test_metrics['recall']:.4f}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n✅ 训练完成，结果保存到: {output_dir}")


if __name__ == "__main__":
    main()