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
| `data_fraction`           | `train_step2.py` / `run_experiments_step3.py` | 1.0 | **Step 3 数据效率实验核心参数**。10% 数据训练 → 0.1；50% → 0.5；100% → 1.0。验收标准: Event-TCN @ 50% F1 > Force-MLP @ 100% F1          
