# Core experiment: stick_slip data-efficiency matrix (paper's main result)
# 8 runs, ~30 epochs each, cached dataset on GPU
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'

function Run-Train($model, $frac, $epochs = 30) {
    Write-Output "[$(Get-Date -Format o)] TRAIN $model frac=$frac"
    & $py train_step2.py --task stick_slip --model $model --data-dir data/stick_slip `
        --output-dir checkpoints --data-fraction $frac --epochs $epochs `
        --batch-size 16 --seq-len 128 --seed 42 *> "train_${model}_f${frac}.log"
}

Run-Train rule_detector 1.0 0
Run-Train event_tcn 1.0
Run-Train event_tcn 0.5
Run-Train event_tcn 0.1
Run-Train force_mlp 1.0
Run-Train force_mlp 0.5
Run-Train vision_only 1.0
Run-Train force_tcn 1.0

Write-Output "[$(Get-Date -Format o)] collect results"
& $py run_experiments_step3.py --exp collect --output results_step3.json *> collect_results.log
Set-Content -LiteralPath 'done_marker.txt' -Value 'CORE_EXP_DONE'
Write-Output "[$(Get-Date -Format o)] CORE EXP ALL DONE"
