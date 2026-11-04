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

## 实验结果 (stick_slip, 200 条轨迹, 30 epochs, seed 42)

### 主线对比 (F1)
| 模型 | F1 | Precision | Recall |
|---|---|---|---|
| Force-MLP @100% (连续力回归基线) | 0.404 | 0.363 | 0.546 |
| Vision-Only @100% | 0.237 | 0.184 | 0.333 |
| Force-TCN @100% (力觉-only 消融) | **0.963** | 0.978 | 0.949 |
| Event-TCN @100% (多模态主模型) | 0.929 | 0.979 | 0.897 |
| Event-TCN @50% | 0.941 | 0.943 | 0.939 |
| Event-TCN @10% | 0.636 | 0.680 | 0.619 |
| RuleDetector (零训练手调阈值) | 0.874 | 0.948 | 0.835 |

**主线验收标准: Event-TCN@50% (0.941) ≥ Force-MLP@100% (0.404) ✓ (2.3× 余量)**

### 噪声鲁棒性 (力觉高斯噪声, F1) — H2 支持
| σ (N) | Event-TCN | Force-MLP | RuleDetector |
|---|---|---|---|
| 0 (干净) | 0.929 | 0.404 | 0.874 |
| 0.1 | 0.935 | 0.404 | 0.874 |
| 0.3 | 0.907 | 0.404 | 0.702 |
| 0.5 | 0.914 | 0.404 | **0.534** |

Event-TCN σ=0.5 仅退化 1.6%, RuleDetector 手调阈值崩至 0.534 (-39%) —
学习式事件检测对传感器噪声天然鲁棒, 验证假设 H2。
(Force-MLP 地板效应: 纯力规则在干净数据下已无法区分 Stick/Slip)

### 时间偏移 / Task2-3 矩阵 / 种子43-44
后台链 `run_exp_phase2.ps1` 运行中 (完成标记: done_marker.txt = PHASE2_DONE),
完成后: `python run_experiments_step3.py --exp collect` → `plot_results_step3.py`

核心发现 (H1 支持):
- 事件分类学习远超连续力回归 (0.93+ vs 0.40)
- 数据效率: Event-TCN@50% ≈ @100%, 均超 RuleDetector
- 力觉是主导模态 (Force-TCN 0.963 ≈ Event-TCN 0.929);
  128×128 侧视图+帧保持下视觉未增益 (论文消融讨论点)
- 图表: `figures/data_efficiency.png`, `figures/ablation.png`; LaTeX 表见 plot 输出

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
