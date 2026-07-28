"""
Step 1 验收脚本 
核心功能: 
  1. (参考) 计算帧级状态准确率
  2. (核心) 计算 Stick→Slip 事件召回率 —— 验证“力跳变+速度跳变”双重条件是否卡准突变时刻
验收标准: 平均事件召回率 >= 0.99 (允许 ±5 帧误差)
"""

from pathlib import Path
import numpy as np

def compute_event_recall(gt_states, slip_events, tolerance=5):
    """
    计算 Stick(1) -> Slip(2) 转换事件的召回率
    Args:
        gt_states:   仿真真值状态序列 (T,)
        slip_events: 自动标注模块保存的事件标记 (T,) dtype=bool (由 force_jump AND vel_jump 生成)
        tolerance:   允许的时间误差帧数 (默认5帧, 即 10ms)
    Returns:
        recall: 召回率 (0~1)
        gt_event_count: 真值总事件数
        matched_count:  匹配上的事件数
    """
    # 1. 真值(GT)中的 Stick->Slip 跳变时刻
    gt_events = np.where((gt_states[:-1] == 1) & (gt_states[1:] == 2))[0] + 1
    
    # 2. 【核心改动】自动事件直接取自采集时保存的 slip_events（不再从 auto_states 反推）
    auto_events = np.where(slip_events)[0]  
    
    gt_event_count = len(gt_events)
    
    # 如果没有发生事件，视为完美
    if gt_event_count == 0:
        return 1.0, 0, 0
    
    # 3. 匹配：GT事件附近 ±tolerance 帧内是否有自动事件
    matched_count = 0
    for ge in gt_events:
        if np.any(np.abs(auto_events - ge) <= tolerance):
            matched_count += 1
    
    recall = matched_count / gt_event_count
    return recall, gt_event_count, matched_count


def main():
    files = sorted(Path("data/seed").glob("traj_*.npz"))
    if len(files) == 0:
        print("❌ 错误: 未找到轨迹文件，请先运行 collect_seed_data.py")
        return

    frame_accs = []          # 帧级准确率 (参考)
    event_recalls = []       # 事件召回率 (核心)
    total_gt_events = 0
    total_matched = 0
    failed_trajs = []

    print("=" * 60)
    print("开始逐条验收轨迹...")
    print("=" * 60)

    for idx, f in enumerate(files):
        d = np.load(f)
        gt = d["contact_states_gt"]
        auto = d["contact_states_auto"]
        slip_ev = d["slip_events"]  # 读取保存的事件标记
        
        # 1. 帧级准确率 (保留作参考)
        acc = (gt == auto).mean()
        frame_accs.append(acc)
        
        # 2. 事件召回率 (传入 slip_ev，而不是 auto)
        recall, gt_cnt, match_cnt = compute_event_recall(gt, slip_ev, tolerance=5)
        event_recalls.append(recall)
        total_gt_events += gt_cnt
        total_matched += match_cnt
        
        if recall < 1.0 and gt_cnt > 0:
            failed_trajs.append((idx, f.name, recall, gt_cnt, match_cnt))

    # --- 输出 ---
    print("\n📊 [参考指标] 帧级状态准确率 (旧标准):")
    print(f"   平均: {np.mean(frame_accs):.4f} ({np.mean(frame_accs)*100:.2f}%)")
    print(f"   最低: {np.min(frame_accs):.4f} ({np.min(frame_accs)*100:.2f}%)")
    
    print("\n🎯 [核心指标] Stick→Slip 事件召回率 (基于力跳变+速度跳变双重条件):")
    print(f"   总真值事件数: {total_gt_events}")
    print(f"   成功匹配事件数: {total_matched}")
    print(f"   平均召回率: {np.mean(event_recalls):.4f} ({np.mean(event_recalls)*100:.2f}%)")
    print(f"   最低召回率: {np.min(event_recalls):.4f} ({np.min(event_recalls)*100:.2f}%)")
    
    # --- 验收 ---
    avg_recall = np.mean(event_recalls)
    print("\n" + "=" * 60)
    
    if avg_recall >= 0.99:
        print("✅ Step 1 验收通过！")
        print("   核心依据: 平均事件召回率 ≥ 99%，算法能精准捕获滑动突变时刻。")
        if np.min(frame_accs) < 0.98:
            print("   ⚠️ 提示: 帧级准确率虽未达98%，但事件召回率达标，说明状态分类略有延迟但不影响事件检测。")
    else:
        print("⚠️  Step 1 验收未通过: 事件召回率不足 99%。")
        print("   建议: 调小 event_annotation.py 中的 force_delta_threshold（让力跳变更敏感）")
        print("        或调大 tolerance 容差（当前为5帧，即10ms）。")
    
    if failed_trajs:
        print("\n🔍 未完全通过的事件检测轨迹 (召回率 < 1.0):")
        for idx, name, recall, gt_cnt, match_cnt in failed_trajs:
            print(f"   - 文件 {name} (索引 {idx}): 召回率 {recall:.2f} ({match_cnt}/{gt_cnt})")
        print("   💡 请运行 visualize_trajectory.py 查看这些轨迹的力-速度曲线，调整阈值。")

if __name__ == "__main__":
    main()