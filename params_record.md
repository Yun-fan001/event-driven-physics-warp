| 参数                        | 文件                    | 默认值           | 什么情况下修改                                                    
| ------------------------- | --------------------- | ------------- | ---------------------------------------------------------- 
| `stick_vel_threshold`     | `env_stick_slip.py`   | 0.01 m/s      | **区分 Stick/Slip 的核心阈值**。物块总被判定为 Slip → 调大；Stick 阶段被切分 → 调小 
| `contact_force_threshold` | `env_stick_slip.py`   | 0.5 N         | 传感器有零漂 → 调大；轻接触检测不到 → 调小                                   
| `push_speed`              | `env_stick_slip.py`   | 0.0008 m/step | Stick 阶段太短 → 调小；仿真太慢 → 调大                                  
| `force_delta_threshold`   | `event_annotation.py` | 2.0 N         | 漏检 Slip 事件 → 调小；误检太多 → 调大                                  
| `slip_vel_threshold`      | `event_annotation.py` | 0.02 m/s      | 应 **略大于** `stick_vel_threshold`，形成滞后带防抖动                   
| `max_steps`               | `env_stick_slip.py`   | 500           | 轨迹太短事件没发生 → 调大；数据文件太大 → 调小                                 
| `insert_speed`            | `env_peg_in_hole.py`            | 0.0008 m/step | 插入太快导致跳过 NoContact 阶段 → 调小；单条轨迹过长、数据文件太大 → 调大                                                              
| `peg_contact_force_threshold` | `event_annotation_v2.py` | 1.0 N | **Task 2 自动标注阈值**。必须与 `env_peg_in_hole.py` 中的 `contact_force_threshold` 保持数值一致（当前均为1.0）。        
| `stick_contact_force_threshold` | `event_annotation_v2.py` | 0.5 N | **Task 1 自动标注接触阈值**。必须与 `env_stick_slip.py` 中的 `contact_force_threshold` 保持数值一致（当前均为0.5）。        
| `grip_force_threshold`    | `env_stack_block.py`            | 2.0 N         | **Task 3 核心阈值**。方块总被判定 NO_CONTACT（吸不住）→ 调小；未接触就判定GRIP → 调大。**注意：需与 `event_annotation_v2.py` 中的同名参数同步修改** 
| `release_height`          | `env_stack_block.py`            | 0.12 m        | 平移时方块蹭到地面 → 调大；轨迹过长、仿真太慢 → 调小                                                                              
| `move_speed`              | `env_stack_block.py`            | 0.0008 m/step | 移动太快导致方块甩飞/吸盘失效 → 调小；数据文件太大 → 调大                                                                           
| `release_pos_threshold`   | `event_annotation_v2.py`        | 0.05 m        | Task 3 Release 判定容差。迟迟不判 Release → 调大；提前判 Release → 调小                                                     
| `seq_len`                 | `dataset.py` / `train_step2.py` | 128 / None    | 显存不足 → 调小(64)；任务需要更长时序上下文 → 调大(256)                                                                        
| `data_fraction`           | `train_step2.py`                | 1.0           | **数据效率实验专用**。10% 数据训练 → 0.1；50% 数据 → 0.5                                                                   
| `batch_size`              | `train_step2.py`                | 16            | 显存 OOM → 调小(8)；训练震荡不稳定 → 调大(32)                                                                            
| `lr`                      | `train_step2.py`                | 1e-3          | loss 长期不下降 → 调大；loss 震荡不收敛 → 调小                                                                            
| `hidden_dim`              | `models/event_tcn.py` 等         | 128           | 模型欠拟合 → 调大(256)；过拟合严重或显存不足 → 调小(64)                                                                        
| `tcn_layers`              | `models/event_tcn.py`           | 4             | 需要更大时序感受野 → 调大；训练不稳定/梯度异常 → 调小(2~3)                                                                        
| `force_noise_std`         | `train_step2.py`                | 0.0           | **Step 3 噪声实验专用**。模拟力传感器零漂: 0.1 轻微 / 0.3 中等 / 0.5 重度。预期 Force-MLP 对此敏感，Event-TCN 鲁棒                 
| `time_shift_ms`           | `train_step2.py`                | 0.0           | **Step 3 噪声实验专用**。模拟视觉-力觉异步采集: ±50/±100/±200ms。正=视觉滞后，负=视觉领先。支撑"时间同步比力值精度更重要"论点           
| `vision_feat_dim`         | `models/event_tcn.py`           | 64            | **消融实验**: 设为 0 时跳过视觉编码器 → 退化为 Force-TCN (力觉-only)，用于消融验证多模态贡献                                    
| `num_layers` (LSTM)       | `models/event_lstm.py`          | 2             | **Event-LSTM 降级方案**: LSTM 层数。训练不稳定 → 减小(1)；欠拟合 → 增大(3)。与 `models/vision_only.py` 的 num_layers 默认值对齐 
| `bidirectional` (LSTM)    | `models/event_lstm.py`          | True          | **Event-LSTM 使用双向 LSTM**。注意: 双向引未来帧，不保证因果性，不适合在线部署。论文需讨论此 trade-off (因果性 vs 准确率) 
| `data_fraction`           | `train_step2.py` / `run_experiments_step3.py` | 1.0 | **Step 3 数据效率实验核心参数**。10% 数据训练 → 0.1；50% → 0.5；100% → 1.0。验收标准: Event-TCN @ 50% F1 > Force-MLP @ 100% F1          

---

## 修复记录 (2026-11 · 重规划版)

### 缺陷修复
- **缺陷1 (力传感器污染)**: 原 `<force>` 传感器测的是 body 与父级约束力 (= 自重 + 位置伺服弹簧力: Task1 重置 25N, Task2 260N)。
  修复: env 改用 `data.contact` + `mj_contactForce` 提取**真实接触力**。
- **缺陷2 (Task3 崩溃)**: stack_block.xml 的 gripper_base 无惯性 → MuJoCo 3.15 mjMINVAL。
  修复: XML 补 `<inertial mass="0.1">`。
- **缺陷3 (非粘滑物理)**: 0.4 m/s 快推、单 μ 摩擦、500 帧仅 1 次转换。
  修复: 0.02 m/s 慢推 + 三角波变幅轨迹 (push_amplitude 0.03-0.05 m), 每轨迹 4-6 次 Stick↔Slip。
- **缺陷4 (循环论证)**: GT 与自动标注共用阈值。修复: GT 全部运动学化 (Task1: 几何间距+速度阈值; Task2: tip 几何间距; Task3: 物块位姿), 标注器用 力+速度 传感器信号独立推导。
- **缺陷5 (标签≈输入)**: 速度拼在模型输入。修复: dataset.py 模型输入仅力(3维), 速度仅存 npz。
- **缺陷6 (中文路径)**: 迁至 `D:\projects\event-driven-physics-warp` (纯英文)。
- **缺陷7 (次要)**:
  - dataset.py 标签 padding: mode="constant", constant_values=-100 (原 edge 冲突)。
  - Force-MLP: 纯力回归 + 力窗口下降规则 (不再使用真值速度)。
  - 采集脚本 emoji → ASCII。
  - **Task 2 坐标 bug**: peg body pos=0.15 叠加 slide 关节 → peg 实际悬浮 z=0.18~0.30, 从未接触 socket (原"接触"全部是伺服弹簧力误判)。body pos 归零。
  - **Task 3 坐标 bug**: 同类型, gripper/block body pos 归零。
  - **Task 3 抓取机制**: 原高摩擦圆柱无法产生法向吸附 (物块从未被提起)。改为 slide 滑块 + 吸力执行器 (suction general actuator, 真空吸盘模型)。

### 关键参数 (修复版)
| 参数 | 文件 | 默认值 | 说明 |
|---|---|---|---|
| push_speed | env_stick_slip.py | 0.02 m/s (随机 0.015/0.02/0.03) | 慢推速度, 3 档随机 |
| push_amplitude | env_stick_slip.py | 0.04 m (随机 0.03-0.05) | 三角波推程, 变幅随机 |
| stick_vel_threshold (GT) | env_stick_slip.py | 0.01 m/s | GT 运动学判据, 仅作标签 |
| stick/slip_vel_threshold (标注) | event_annotation_v2.py | 0.01 / 0.02 | 标注迟滞带, 与 GT 阈值不同 |
| stick_contact_force_threshold | event_annotation_v2.py | 0.5 N | 标注接触阈值 (真实接触力) |
| peg_contact_force_threshold | event_annotation_v2.py | 1.0 N | Task 2 标注接触阈值 |
| insert_speed / z_min | env_peg_in_hole.py | 0.0004 m/step / 0.05 | 插入速度 / 目标下限 |
| lateral_offset_max | env_peg_in_hole.py | 0.002 m | 侧向偏移 0-2 mm 随机化 |
| grip_force_threshold (标注) | event_annotation_v2.py | 1.5 N | Task 3 标注抓取阈值 |
| suction_force / place_height | env_stack_block.py | 1.5 N / 0.098 | 吸力 / 放置停止高度 |
| move_speed / max_steps | env_stack_block.py | 0.0004 / 1200 | 移动速度 / 轨迹长度 |

