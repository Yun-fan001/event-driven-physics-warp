# Cleanup experiments (2026-11): clean GT re-collection + re-run + peak sweep
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'
$C = @('--output-dir','checkpoints','--epochs','30','--batch-size','16','--seq-len','128','--seed','42')

# 1. 清理版 GT 重新采集 peg + stack (覆盖旧数据)
& $py collect_data_step2.py --task peg_in_hole --num-episodes 200 --output-dir data/peg_in_hole --render-interval 5 *> cleanup_collect_peg.log
& $py collect_data_step2.py --task stack_block --num-episodes 200 --output-dir data/stack_block --render-interval 5 *> cleanup_collect_stack.log

# 2. peg/stack 全模型矩阵 (清理版 GT)
foreach ($task in @('peg_in_hole','stack_block')) {
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $C --data-fraction 1.0 *> "cleanup_${task}_eventtcn.log"
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $C --data-fraction 0.5 *> "cleanup_${task}_eventtcn_frac0.5.log"
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $C --data-fraction 0.1 *> "cleanup_${task}_eventtcn_frac0.1.log"
    & $py train_step2.py --task $task --model force_mlp --data-dir data/$task $C *> "cleanup_${task}_forcemlp.log"
    & $py train_step2.py --task $task --model force_tcn --data-dir data/$task $C *> "cleanup_${task}_forcetcn.log"
    & $py train_step2.py --task $task --model vision_only --data-dir data/$task $C *> "cleanup_${task}_visiononly.log"
    & $py train_step2.py --task $task --model rule_detector --data-dir data/$task $C --epochs 0 *> "cleanup_${task}_rule.log"
}

# 3. stick_slip 峰值搜索 (更大模型/更长时序)
& $py train_step2.py --task stick_slip --model force_tcn --data-dir data/stick_slip --output-dir checkpoints_sweep --epochs 40 --batch-size 16 --seq-len 128 --hidden-dim 256 --tcn-layers 6 --seed 42 *> cleanup_force_tcn_hd256_l6.log
& $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --output-dir checkpoints_sweep --epochs 40 --batch-size 8 --seq-len 256 --hidden-dim 256 --tcn-layers 6 --seed 42 *> cleanup_event_tcn_hd256_l6_s256.log
& $py train_step2.py --task stick_slip --model force_tcn --data-dir data/stick_slip --output-dir checkpoints_sweep --epochs 40 --batch-size 16 --seq-len 256 --hidden-dim 256 --tcn-layers 6 --seed 42 *> cleanup_force_tcn_hd256_l6_s256.log

Set-Content -LiteralPath 'done_marker.txt' -Value 'CLEANUP_DONE'
Write-Output "[$(Get-Date -Format o)] CLEANUP ALL DONE"
