# Phase 2 experiments: noise robustness + time shift + task matrices + multi-seed
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'
$common = @('--output-dir','checkpoints','--epochs','30','--batch-size','16','--seq-len','128','--seed','42')

# --- A. 噪声鲁棒性 (stick_slip): 力觉高斯噪声 sigma 0.1/0.3/0.5 ---
foreach ($std in @('0.1','0.3','0.5')) {
    & $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip $common --force-noise-std $std *> "exp2_eventtcn_noise_$std.log"
    & $py train_step2.py --task stick_slip --model force_mlp --data-dir data/stick_slip $common --force-noise-std $std *> "exp2_forcemlp_noise_$std.log"
    & $py train_step2.py --task stick_slip --model rule_detector --data-dir data/stick_slip $common --epochs 0 --force-noise-std $std *> "exp2_rule_noise_$std.log"
}

# --- B. 时间偏移 (stick_slip, Event-TCN): +/-50/100/200 ms ---
foreach ($ms in @('50','-50','100','-100','200','-200')) {
    & $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip $common --time-shift-ms $ms *> "exp2_eventtcn_shift_$ms.log"
}

# --- C. Task 2/3 全模型矩阵 ---
foreach ($task in @('peg_in_hole','stack_block')) {
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $common --data-fraction 1.0 *> "exp2_${task}_eventtcn.log"
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $common --data-fraction 0.5 *> "exp2_${task}_eventtcn_frac0.5.log"
    & $py train_step2.py --task $task --model event_tcn --data-dir data/$task $common --data-fraction 0.1 *> "exp2_${task}_eventtcn_frac0.1.log"
    & $py train_step2.py --task $task --model force_mlp --data-dir data/$task $common *> "exp2_${task}_forcemlp.log"
    & $py train_step2.py --task $task --model force_tcn --data-dir data/$task $common *> "exp2_${task}_forcetcn.log"
    & $py train_step2.py --task $task --model vision_only --data-dir data/$task $common *> "exp2_${task}_visiononly.log"
    & $py train_step2.py --task $task --model rule_detector --data-dir data/$task $common --epochs 0 *> "exp2_${task}_rule.log"
}

# --- D. 多种子复现 (stick_slip 主线验收: seed 43/44) ---
foreach ($seed in @('43','44')) {
    & $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip $common --seed $seed --data-fraction 0.5 *> "exp2_eventtcn_frac0.5_seed$seed.log"
    & $py train_step2.py --task stick_slip --model force_mlp --data-dir data/stick_slip $common --seed $seed *> "exp2_forcemlp_seed$seed.log"
}

& $py run_experiments_step3.py --exp collect --output results_step3.json *> collect_results.log
Set-Content -LiteralPath 'done_marker.txt' -Value 'PHASE2_DONE'
Write-Output "[$(Get-Date -Format o)] PHASE2 ALL DONE"
