# Core experiment: stick_slip data-efficiency matrix (paper's main result)
# 7 runs (rule_detector 已完成), 30 epochs each, cached dataset on GPU
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'

& $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --output-dir checkpoints --data-fraction 1.0 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_eventtcn_f1.0.log
& $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --output-dir checkpoints --data-fraction 0.5 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_eventtcn_f0.5.log
& $py train_step2.py --task stick_slip --model event_tcn --data-dir data/stick_slip --output-dir checkpoints --data-fraction 0.1 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_eventtcn_f0.1.log
& $py train_step2.py --task stick_slip --model force_mlp --data-dir data/stick_slip --output-dir checkpoints --data-fraction 1.0 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_forcemlp_f1.0.log
& $py train_step2.py --task stick_slip --model force_mlp --data-dir data/stick_slip --output-dir checkpoints --data-fraction 0.5 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_forcemlp_f0.5.log
& $py train_step2.py --task stick_slip --model vision_only --data-dir data/stick_slip --output-dir checkpoints --data-fraction 1.0 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_visiononly_f1.0.log
& $py train_step2.py --task stick_slip --model force_tcn --data-dir data/stick_slip --output-dir checkpoints --data-fraction 1.0 --epochs 30 --batch-size 16 --seq-len 128 --seed 42 *> train_forcetcn_f1.0.log

& $py run_experiments_step3.py --exp collect --output results_step3.json *> collect_results.log
Set-Content -LiteralPath 'done_marker.txt' -Value 'CORE_EXP_DONE'
Write-Output "[$(Get-Date -Format o)] CORE EXP ALL DONE"
