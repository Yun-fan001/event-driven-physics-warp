# Event-Driven Physics Warp - World Model

基于离散物理事件 (Contact / Stick / Slip / Release) 的接触密集操作
感知与建模研究。验证核心命题: **离散物理事件作为表示/监督单元,
在数据效率、噪声鲁棒性与时间异步条件下是否优于连续力回归。**

> 对应文档: 《实验规划(重规划版)》《数据获取规划》《课题评估与实施路线图-立项版》
> (见桌面原项目目录)。本目录为纯英文路径工作区 (修复缺陷6 中文路径 bug)。

## 目录结构

```
assets/                MuJoCo 场景 XML (3 任务, 已修复)
models/                模型: EventTCN / EventLSTM / ForceMLP / VisionOnly
env_stick_slip.py      Task 1 Slip-Onset 环境 (重设计: 慢推三角波)
env_peg_in_hole.py     Task 2 Peg-in-Hole 环境 (修复: GT 运动学化 + 侧向偏移)
env_stack_block.py     Task 3 Stack-Block 环境 (重设计: 滑块+吸力执行器)
event_annotation_v2.py 自动标注器 (力+速度迟滞带; Task1/2/3)
annotation_report.py   标注质量报告 (事件 F1/P/R ±5帧 + 时间误差分布)
collect_data_step2.py  统一采集脚本 (3 任务)
train_step2.py         训练脚本 (6 模型 + RuleDetector 零训练基线)
run_experiments_step3.py 实验矩阵 (数据效率/噪声/时间偏移)
plot_results_step3.py  论文图表 + LaTeX 表
smoke_env.py           三任务物理冒烟测试
check_env_v2.py        环境检查
guide.txt              快速指南
params_record.md       参数速查 + 修复记录
real_robot_adaptation_guide_ros2.txt  实机 ROS2 适配指南
```

## 快速开始

```powershell
Set-Location D:\projects\event-driven-physics-warp
.venv\Scripts\python.exe check_env_v2.py          # 环境检查
.venv\Scripts\python.exe smoke_env.py             # 三任务物理冒烟
.venv\Scripts\python.exe collect_data_step2.py --task stick_slip --num-episodes 200 --output-dir data/stick_slip --render-interval 5
.venv\Scripts\python.exe annotation_report.py --task all --data-dir data
.venv\Scripts\python.exe train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --epochs 30
```

详见 guide.txt。

## 数据集 (已采集, 2026-11-05)

- 3 任务 × 200 条 = 600 条轨迹, 全部通过完整性扫描
- render-interval=5: 视觉等效 100 Hz 帧保持 (力 500 Hz), 采集提速 ~15x
- 标注质量报告 (`data/annotation_report_all.json`):
  - stick_slip: 帧准确率 0.966, 事件 F1=0.819 (P=0.803/R=0.835), 时间误差中位数 1 帧
  - peg_in_hole: 帧准确率 0.991, 事件 F1=1.000, 时间误差中位数 3 帧
  - stack_block: 帧准确率 0.983, 事件 F1=0.516 (R=1.000), 时间误差中位数 1 帧

## 实验结果汇总 (200 轨迹/任务, 30 epochs)

### 三任务全模型对比 (F1, seed 42)
| 模型 | stick_slip | peg_in_hole | stack_block |
|---|---|---|---|
| Force-MLP @100% | 0.404 | 0.980 | 0.459 |
| Vision-Only @100% | 0.237 | 0.995 | 0.965 |
| Force-TCN @100% | **0.963** | 0.980 | 0.963 |
| Event-TCN @100% | 0.929 | **0.996** | **0.986** |
| RuleDetector (零训练) | 0.874 | 0.985 | 0.433 |

### 主线验收 (Event-TCN@50% ≥ Force-MLP@100%) — 3 种子复现 ✓
| seed | Event-TCN@50% | Force-MLP@100% | 达标 |
|---|---|---|---|
| 42 | 0.941 | 0.404 | ✓ |
| 43 | 0.957 | 0.433 | ✓ |
| 44 | 0.921 | 0.410 | ✓ |
| 均值±std | 0.940±0.018 | 0.416±0.015 | ✓ 2.3× |

### 数据效率 (Event-TCN F1)
| 数据量 | stick_slip | peg_in_hole | stack_block |
|---|---|---|---|
| 10% | 0.636 | 0.986 | 0.513 |
| 50% | 0.941 | 0.984 | 0.969 |
| 100% | 0.929 | 0.996 | 0.986 |

### 噪声鲁棒性 (力觉高斯噪声, F1) — H2 ✓
| σ (N) | Event-TCN | Force-MLP | RuleDetector |
|---|---|---|---|
| 0 | 0.929 | 0.404 | 0.874 |
| 0.1 | 0.935 | 0.404 | 0.874 |
| 0.3 | 0.907 | 0.404 | 0.702 |
| 0.5 | 0.914 | 0.404 | **0.534** |

### 时间偏移 (Event-TCN F1)
| shift | stick_slip | stack_block (H3, 运行中) |
|---|---|---|
| 0 | 0.929 | 0.986 |
| ±50 | 0.934 / 0.930 | ... |
| ±100 | 0.960 / 0.909 | ... |
| ±200 | 0.953 / 0.951 | ... |

stick_slip 上视觉-力觉偏移 ±200ms 无显著退化 — 结构性零结果
(该任务视觉分支贡献小, 力觉主导); H3 严格检验在视觉主导的
stack_block 上进行 (`run_h3_stack_shift.ps1` 后台运行中)。

## 核心结论 (论文要点)
1. **H1 (表示假设) ✓**: 事件分类学习 (0.93+) 远超连续力回归 (0.40), 3 种子复现;
   stack_block 上 Event-TCN (0.986) > 力觉-only (0.963) ≈ 视觉-only (0.965)
2. **数据效率 ✓**: 50% 数据即可达标, stick/stack 均有清晰曲线
3. **H2 (噪声鲁棒) ✓**: Event-TCN σ=0.5N 仅退化 1.6%, 手调阈值基线崩至 0.534 (-39%)
4. **模态消融**: stick_slip 力觉主导; stack_block 视觉贡献大 (Grip/Release 运动可见)
5. **任务难度梯度**: peg (饱和) < stack < stick_slip (最难的滑移起始检测)
6. 图表: `figures/data_efficiency.png` / `noise_robustness.png` / `ablation.png`

## 2026-11 修复摘要 (7 项缺陷)

1. **力传感器污染** → env 改用 data.contact + mj_contactForce 提取真实接触力
2. **Task3 崩溃** → gripper_base 补惯性
3. **非粘滑物理** → 慢推 0.02 m/s + 三角波变幅轨迹 (kp↑/kv↓/质量↓)
4. **循环论证** → GT 全部运动学化 (几何间距/速度阈值/物块位姿), 与标注信号解耦
5. **标签≈输入** → 模型输入去除速度通道 (仅 RuleDetector 基线用速度)
6. **中文路径** → 迁移至本目录
7. **次要缺陷** → 标签 padding、Force-MLP 纯力规则、emoji、坐标 frame
   (Task2 peg 悬浮从未接触 socket、Task3 物块从未被提起——均为坐标/机制级 bug)

## 验收标准 (重规划修订版)

- 事件 macro-F1 / Precision / Recall (±5 帧容差)
- 事件时刻误差分布: 中位数 ≤ 2 帧
- 3 随机种子; 主线: Event-TCN@50% ≥ Force-MLP@100%
- RuleDetector (零训练手调阈值) 必须被主模型明确超越
- 诚实条款: 未达预期转原因分析写入 Limitation, 绝不改数据
