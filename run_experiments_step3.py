"""
Step 3 批量实验运行脚本
目标: 自动化运行完整的实验矩阵，产出 Step 3 所需的所有数据点。

实验矩阵:
  数据效率实验: 3 tasks × 4 models × 3 data_fractions (10%, 50%, 100%)
  噪声鲁棒性实验: 3 tasks × 4 models × 4 noise conditions
  消融实验: 已由 model 选项覆盖 (force_mlp / vision_only / force_tcn / event_tcn)

用法:
  # 数据效率实验 (所有任务+模型, 10%/50%/100%)
  python run_experiments_step3.py --exp data_efficiency --tasks all --models all

  # 噪声鲁棒性实验
  python run_experiments_step3.py --exp noise_robustness --tasks stick_slip --models event_tcn,force_mlp

  # 单组快速测试
  python run_experiments_step3.py --exp data_efficiency --tasks stick_slip --models event_tcn --fractions 0.5

输出结构:
  checkpoints/
    {task}/{model}/                          ← 标准训练 (100% 数据, 无噪声)
    {task}/{model}/frac_{fraction}/           ← 数据效率实验
    {task}/{model}/noise_f{std}_t{ms}ms/     ← 噪声鲁棒性实验
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# 实验配置
# ============================================================
TASKS = ["stick_slip", "peg_in_hole", "stack_block"]
MODELS = ["force_mlp", "vision_only", "force_tcn", "event_tcn", "force_lstm", "event_lstm"]

DATA_FRACTIONS = [0.1, 0.5, 1.0]  # 数据效率实验: 10%/50%/100%

# 噪声鲁棒性实验条件
NOISE_CONDITIONS = [
    # (force_noise_std, time_shift_ms, label)
    (0.0,   0,   "clean"),
    (0.1,   0,   "force_noise_0.1"),
    (0.3,   0,   "force_noise_0.3"),
    (0.0,  50,   "time_shift_+50ms"),
    (0.0, 100,   "time_shift_+100ms"),
    (0.0, 200,   "time_shift_+200ms"),
    (0.0, -50,   "time_shift_-50ms"),
    (0.3,  50,   "combined_noise"),
]

# 训练基础参数 (所有实验共用，除非指定覆盖)
BASE_TRAIN_ARGS = {
    "epochs": 50,
    "batch_size": 16,
    "lr": 1e-3,
    "seq_len": 128,
    "seed": 42,
}


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    """运行命令，返回 exit code"""
    print(f"  {'[DRY RUN] ' if dry_run else ''}$ {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def run_data_efficiency(
    tasks: list[str],
    models: list[str],
    fractions: list[float],
    dry_run: bool = False,
) -> None:
    """数据效率实验: 遍历 tasks × models × data_fractions"""
    total = len(tasks) * len(models) * len(fractions)
    count = 0
    failed = []

    for task, model, frac in itertools.product(tasks, models, fractions):
        count += 1
        data_dir = f"data/{task}"
        output_dir = f"checkpoints/{task}/{model}/frac_{frac}"

        print(f"\n{'='*60}")
        print(f"[{count}/{total}] 数据效率: {task}/{model} @ {frac*100:.0f}% 数据")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train_step2.py",
            "--task", task,
            "--model", model,
            "--data-dir", data_dir,
            "--output-dir", "checkpoints",
            "--data-fraction", str(frac),
            "--epochs", str(BASE_TRAIN_ARGS["epochs"]),
            "--batch-size", str(BASE_TRAIN_ARGS["batch_size"]),
            "--lr", str(BASE_TRAIN_ARGS["lr"]),
            "--seq-len", str(BASE_TRAIN_ARGS["seq_len"]),
            "--seed", str(BASE_TRAIN_ARGS["seed"]),
        ]

        # force_mlp 对 stack_block 效果有限（只用阈值为简化版），降低 epochs
        if model == "force_mlp" and task == "stack_block":
            cmd.extend(["--epochs", "30"])

        rc = run_command(cmd, dry_run=dry_run)
        if rc != 0:
            failed.append(f"{task}/{model} @ {frac*100:.0f}%")

    print("\n" + "="*60)
    print(f"数据效率实验完成: {count - len(failed)}/{count} 成功")
    if failed:
        print(f"失败项目 ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


def run_noise_robustness(
    tasks: list[str],
    models: list[str],
    conditions: list[tuple[float, float, str]] | None = None,
    dry_run: bool = False,
) -> None:
    """噪声鲁棒性实验: 遍历 tasks × models × noise_conditions"""
    if conditions is None:
        conditions = NOISE_CONDITIONS

    total = len(tasks) * len(models) * len(conditions)
    count = 0
    failed = []

    for task, model, (noise_std, shift_ms, label) in itertools.product(
        tasks, models, conditions
    ):
        count += 1
        data_dir = f"data/{task}"
        suffix = f"noise_f{noise_std}_t{shift_ms:.0f}ms"

        print(f"\n{'='*60}")
        print(f"[{count}/{total}] 噪声鲁棒性: {task}/{model} | {label}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train_step2.py",
            "--task", task,
            "--model", model,
            "--data-dir", data_dir,
            "--output-dir", "checkpoints",
            "--force-noise-std", str(noise_std),
            "--time-shift-ms", str(shift_ms),
            "--epochs", str(BASE_TRAIN_ARGS["epochs"]),
            "--batch-size", str(BASE_TRAIN_ARGS["batch_size"]),
            "--lr", str(BASE_TRAIN_ARGS["lr"]),
            "--seq-len", str(BASE_TRAIN_ARGS["seq_len"]),
            "--seed", str(BASE_TRAIN_ARGS["seed"]),
        ]

        rc = run_command(cmd, dry_run=dry_run)
        if rc != 0:
            failed.append(f"{task}/{model} | {label}")

    print("\n" + "="*60)
    print(f"噪声鲁棒性实验完成: {count - len(failed)}/{count} 成功")
    if failed:
        print(f"失败项目 ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


def collect_results(output_path: str) -> None:
    """
    汇总所有实验结果为单个 JSON，供 plot_results_step3.py 使用。
    遍历 checkpoints/ 目录，读取每个子目录的 test_results.json。
    """
    results = []
    ckpt_root = Path("checkpoints")

    if not ckpt_root.exists():
        print("⚠️  checkpoints/ 目录不存在，请先运行实验")
        return

    for test_json in ckpt_root.rglob("test_results.json"):
        rel_path = test_json.parent.relative_to(ckpt_root)
        parts = rel_path.parts  # e.g., ('stick_slip', 'event_tcn') 或 ('stick_slip', 'event_tcn', 'frac_0.1')

        try:
            with open(test_json, "r") as f:
                metrics = json.load(f)
        except Exception:
            continue

        entry = {
            "task": parts[0] if len(parts) >= 1 else "unknown",
            "model": parts[1] if len(parts) >= 2 else "unknown",
            "condition": "/".join(parts[2:]) if len(parts) >= 3 else "standard",
            "metrics": metrics,
        }
        results.append(entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 汇总 {len(results)} 个实验结果到 {output_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 3 批量实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览数据效率实验 (不实际运行)
  python run_experiments_step3.py --exp data_efficiency --dry-run

  # 运行单个任务的数据效率实验
  python run_experiments_step3.py --exp data_efficiency --tasks stick_slip

  # 运行全部噪声鲁棒性实验
  python run_experiments_step3.py --exp noise_robustness

  # 汇总已有结果 (不训练)
  python run_experiments_step3.py --exp collect --output results_step3.json
        """,
    )
    parser.add_argument(
        "--exp", type=str, required=True,
        choices=["data_efficiency", "noise_robustness", "all", "collect"],
        help="实验类型: data_efficiency / noise_robustness / all / collect(仅汇总)",
    )
    parser.add_argument(
        "--tasks", type=str, default="all",
        help="逗号分隔的任务列表, 如 stick_slip,peg_in_hole (默认 all)",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="逗号分隔的模型列表, 如 event_tcn,force_mlp (默认 all)",
    )
    parser.add_argument(
        "--fractions", type=str, default="0.1,0.5,1.0",
        help="数据效率比例 (默认 0.1,0.5,1.0)",
    )
    parser.add_argument(
        "--output", type=str, default="results_step3.json",
        help="汇总结果输出路径",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印命令，不实际运行",
    )
    args = parser.parse_args()

    # 解析任务和模型
    tasks = TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    models = MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    fractions = [float(f.strip()) for f in args.fractions.split(",")]

    # 验证
    for t in tasks:
        if t not in TASKS:
            print(f"❌ 未知任务: {t}. 可选: {TASKS}")
            sys.exit(1)
    for m in models:
        if m not in MODELS:
            print(f"❌ 未知模型: {m}. 可选: {MODELS}")
            sys.exit(1)

    print(f"🚀 Step 3 实验启动: {args.exp}")
    print(f"   任务: {tasks}")
    print(f"   模型: {models}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.exp in ("data_efficiency", "all"):
        run_data_efficiency(tasks, models, fractions, dry_run=args.dry_run)

    if args.exp in ("noise_robustness", "all"):
        run_noise_robustness(tasks, models, dry_run=args.dry_run)

    # 汇总结果
    if not args.dry_run:
        collect_results(args.output)
        print(f"\n📊 结果已汇总到 {args.output}")
        print(f"   下一步: python plot_results_step3.py --results {args.output}")


if __name__ == "__main__":
    main()