"""
Step 3 实验结果绘图脚本
目标: 从汇总的 JSON 结果文件生成论文级图表。

输出图表:
  1. 数据效率曲线: F1-score vs. 数据量 (3 tasks × 1 图，每条曲线一个模型)
  2. 噪声鲁棒性柱状图: F1 下降率 vs. 噪声条件
  3. 消融对比: vision-only / force-only / multimodal 的 F1 柱状图
  4. 综合实验结果表格 (LaTeX 格式输出到终端)

用法:
  # 先汇总结果
  python run_experiments_step3.py --exp collect --output results_step3.json

  # 绘图
  python plot_results_step3.py --results results_step3.json --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，适合服务器/远程运行
import matplotlib.pyplot as plt
import numpy as np

# 中文字体 (需要系统安装中文字体如 Noto Sans CJK / SimHei)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 模型/任务 可视化映射
# ============================================================
MODEL_LABELS = {
    "force_mlp":   "Force-MLP (连续力回归)",
    "vision_only": "Vision-Only (纯视觉)",
    "force_tcn":   "Force-TCN (力觉-only)",
    "event_tcn":   "Event-TCN (多模态)",
    "force_lstm":  "Force-LSTM (力觉-only Bi-LSTM)",
    "event_lstm":  "Event-LSTM (多模态 Bi-LSTM)",
    "rule_detector": "RuleDetector (零训练手调阈值)",
}
MODEL_COLORS = {
    "force_mlp":   "#e74c3c",  # 红
    "vision_only": "#3498db",  # 蓝
    "force_tcn":   "#f39c12",  # 橙
    "event_tcn":   "#2ecc71",  # 绿
    "force_lstm":  "#9b59b6",  # 紫
    "event_lstm":  "#1abc9c",  # 青
    "rule_detector": "#7f8c8d",  # 灰
}
MODEL_MARKERS = {
    "force_mlp":   "s",
    "vision_only": "o",
    "force_tcn":   "D",
    "event_tcn":   "^",
    "force_lstm":  "v",
    "event_lstm":  "p",
    "rule_detector": "x",
}
TASK_LABELS = {
    "stick_slip":  "Task 1: Stick-Slip",
    "peg_in_hole": "Task 2: Peg-in-Hole",
    "stack_block": "Task 3: Stack-Block",
}


def load_results(path: str) -> list[dict]:
    """加载 JSON 汇总结果"""
    with open(path, "r") as f:
        return json.load(f)


def filter_data_efficiency(results: list[dict]) -> list[dict]:
    """筛选数据效率实验条目 (condition 以 'frac_' 开头)"""
    return [r for r in results if r["condition"].startswith("frac_")]


def filter_noise(results: list[dict]) -> list[dict]:
    """筛选噪声实验条目 (condition 以 'noise_' 开头)"""
    return [r for r in results if r["condition"].startswith("noise_")]


def filter_standard(results: list[dict]) -> list[dict]:
    """筛选标准训练条目 (condition == 'standard')"""
    return [r for r in results if r["condition"] == "standard"]


# ============================================================
# 图表 1: 数据效率曲线 (F1-score vs. 数据量)
# ============================================================
def plot_data_efficiency(results: list[dict], output_dir: Path) -> None:
    data_eff = filter_data_efficiency(results)
    if not data_eff:
        print("⚠️  无数据效率实验结果，跳过图表 1")
        return

    # 按任务组织: {task: {model: [(frac, f1), ...]}}
    by_task = defaultdict(lambda: defaultdict(list))
    for entry in data_eff:
        task = entry["task"]
        model = entry["model"]
        # 从 condition 字段解析 fraction 值
        cond = entry["condition"]  # e.g., "frac_0.1" 或 "noise_f0.1_t50ms"
        try:
            frac_str = cond.replace("frac_", "").split("_")[0]
            frac = float(frac_str)
        except (ValueError, IndexError):
            continue
        f1 = entry["metrics"].get("f1", 0.0)
        by_task[task][model].append((frac * 100, f1))  # 转百分比

    # 排序每个模型的数据点
    for task in by_task:
        for model in by_task[task]:
            by_task[task][model].sort(key=lambda x: x[0])

    n_tasks = len(by_task)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, task in enumerate(sorted(by_task.keys())):
        ax = axes[ax_idx]
        for model in ["force_mlp", "vision_only", "force_tcn", "event_tcn"]:
            points = by_task[task].get(model, [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker=MODEL_MARKERS.get(model, "o"),
                    color=MODEL_COLORS.get(model, "#333"),
                    label=MODEL_LABELS.get(model, model),
                    linewidth=1.8, markersize=7)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=13, fontweight="bold")
        ax.set_xlabel("训练数据比例 (%)")
        ax.set_ylabel("F1-score (macro)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("数据效率实验: F1-score vs. 训练数据量", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "data_efficiency.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✅ 数据效率曲线 -> {out}")


# ============================================================
# 图表 2: 噪声鲁棒性柱状图 (F1 下降率)
# ============================================================
def plot_noise_robustness(results: list[dict], output_dir: Path) -> None:
    noise_data = filter_noise(results)
    standard_data = filter_standard(results)

    if not noise_data or not standard_data:
        print("⚠️  无噪声鲁棒性实验结果，跳过图表 2")
        return

    # 建立标准结果的基准: {(task, model): baseline_f1}
    baseline = {}
    for entry in standard_data:
        key = (entry["task"], entry["model"])
        baseline[key] = entry["metrics"].get("f1", 0.0)

    # 噪声条件分组 (只取几个代表性条件)
    key_conditions = [
        ("clean",           "无噪声\n(基准)"),
        ("noise_f0.1_t0ms", "力噪声\nσ=0.1"),
        ("noise_f0.3_t0ms", "力噪声\nσ=0.3"),
        ("noise_f0.0_t50ms","时偏\n+50ms"),
        ("noise_f0.0_t200ms","时偏\n+200ms"),
    ]

    # 只对 event_tcn 和 force_mlp 绘制噪声对比
    focus_models = ["event_tcn", "force_mlp"]
    tasks = sorted(set(e["task"] for e in noise_data))

    n_conds = len(key_conditions)
    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(12, 4 * n_tasks), squeeze=False)
    axes = axes[:, 0] if n_tasks > 1 else [axes[0, 0]]

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        x = np.arange(n_conds)
        bar_width = 0.35

        for i, model in enumerate(focus_models):
            baseline_f1 = baseline.get((task, model), 0.5)
            values = []
            for cond_prefix, _ in key_conditions:
                # 查找匹配的噪声实验
                matched = [e for e in noise_data
                           if e["task"] == task
                           and e["model"] == model
                           and e["condition"].startswith(cond_prefix)]
                if matched:
                    values.append(matched[0]["metrics"].get("f1", baseline_f1))
                else:
                    values.append(baseline_f1)  # 回退到基准值

            bars = ax.bar(x + i * bar_width, values, bar_width,
                          color=MODEL_COLORS.get(model, "#666"),
                          alpha=0.85, label=MODEL_LABELS.get(model, model))

            # 标注值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([l for _, l in key_conditions], fontsize=9)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=12, fontweight="bold")
        ax.set_ylabel("F1-score (macro)")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2, axis="y")
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("噪声鲁棒性: Event-TCN vs Force-MLP", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "noise_robustness.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✅ 噪声鲁棒性柱状图 -> {out}")


# ============================================================
# 图表 3: 消融柱状图 (vision-only / force-only / multimodal)
# ============================================================
def plot_ablation(results: list[dict], output_dir: Path) -> None:
    standard = filter_standard(results)
    if not standard:
        # 如果没有标准训练结果，尝试用数据效率 100% 的结果代替
        data_eff = filter_data_efficiency(results)
        standard = [e for e in data_eff if "frac_1.0" in e["condition"]]

    if not standard:
        print("⚠️  无消融实验结果，跳过图表 3")
        return

    # 按任务组织: {task: {model: f1}}
    by_task = defaultdict(dict)
    for entry in standard:
        task = entry["task"]
        model = entry["model"]
        f1 = entry["metrics"].get("f1", 0.0)
        by_task[task][model] = f1

    ablation_models = ["vision_only", "force_tcn", "event_tcn"]
    tasks = sorted(by_task.keys())

    x = np.arange(len(tasks))
    bar_width = 0.25
    colors_ablation = {
        "vision_only": "#3498db",
        "force_tcn":   "#f39c12",
        "event_tcn":   "#2ecc71",
    }
    labels_ablation = {
        "vision_only": "Vision-Only",
        "force_tcn":   "Force-TCN (力觉-only)",
        "event_tcn":   "Event-TCN (多模态)",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(ablation_models):
        values = [by_task[t].get(model, 0.0) for t in tasks]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      color=colors_ablation.get(model, "#999"),
                      alpha=0.85, label=labels_ablation.get(model, model))
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=11)
    ax.set_ylabel("F1-score (macro)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_title("消融实验: 模态贡献分析", fontsize=14, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "ablation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✅ 消融柱状图 -> {out}")


# ============================================================
# LaTeX 表格输出
# ============================================================
def print_latex_table(results: list[dict]) -> None:
    """打印实验结果 LaTeX 表格到终端"""
    standard = filter_standard(results)
    if not standard:
        data_eff = filter_data_efficiency(results)
        standard = [e for e in data_eff if "frac_1.0" in e["condition"]]

    if not standard:
        print("\n⚠️  无实验数据，无法生成表格")
        return

    print("\n" + "="*80)
    print("实验结果汇总 (100% 训练数据)")
    print("="*80)

    tasks = sorted(set(e["task"] for e in standard))
    models_order = ["force_mlp", "vision_only", "force_tcn", "event_tcn", "rule_detector"]

    # 表头
    header = f"{'Task':<16} | {'Model':<16} | {'F1':>6} | {'Precision':>9} | {'Recall':>7} | {'Acc':>6}"
    sep = "-" * len(header)
    print(sep)
    print("LaTeX 表格源码:")
    print(sep)
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{实验结果: 3 任务 × 4 模型的 F1-score / Precision / Recall}")
    print(r"  \label{tab:results}")
    print(r"  \begin{tabular}{lcccc}")
    print(r"    \toprule")
    print(r"    任务 & 模型 & F1-score & Precision & Recall \\")
    print(r"    \midrule")

    for task in tasks:
        first_row = True
        for model in models_order:
            matched = [e for e in standard if e["task"] == task and e["model"] == model]
            if not matched:
                continue
            m = matched[0]["metrics"]
            task_label = TASK_LABELS.get(task, task) if first_row else ""
            print(f"    {task_label:<20} & {MODEL_LABELS.get(model, model):<20} & "
                  f"{m.get('f1', 0):.3f} & {m.get('precision', 0):.3f} & "
                  f"{m.get('recall', 0):.3f} \\\\")
            first_row = False
        print(r"    \midrule" if task != tasks[-1] else r"    \bottomrule")

    print(r"  \end{tabular}")
    print(r"\end{table}")
    print(sep)

    # 同时打印人类可读版本
    print(f"\n{'Task':<16} | {'Model':<16} | {'F1':>6} | {'Precision':>9} | {'Recall':>7} | {'Acc':>6}")
    print(sep)
    for task in tasks:
        for model in models_order:
            matched = [e for e in standard if e["task"] == task and e["model"] == model]
            if not matched:
                continue
            m = matched[0]["metrics"]
            print(f"{task:<16} | {model:<16} | {m.get('f1',0):6.3f} | "
                  f"{m.get('precision',0):9.3f} | {m.get('recall',0):7.3f} | "
                  f"{m.get('accuracy',0):6.3f}")
    print(sep)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 3 实验结果绘图")
    parser.add_argument("--results", type=str, required=True,
                        help="汇总结果 JSON 文件路径 (由 run_experiments_step3.py --exp collect 生成)")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="输出图片目录")
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"✅ 加载 {len(results)} 条实验结果")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据效率曲线
    print("\n📈 图表 1: 数据效率曲线...")
    plot_data_efficiency(results, output_dir)

    # 2. 噪声鲁棒性柱状图
    print("\n📈 图表 2: 噪声鲁棒性柱状图...")
    plot_noise_robustness(results, output_dir)

    # 3. 消融柱状图
    print("\n📈 图表 3: 消融对比柱状图...")
    plot_ablation(results, output_dir)

    # 4. LaTeX 表格
    print_latex_table(results)

    print(f"\n🎉 所有图表已保存到: {output_dir}/")


if __name__ == "__main__":
    main()