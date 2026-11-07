# H3 follow-up: time-shift on stack_block (vision-dominant task)
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'

foreach ($ms in @('50','-50','100','-100','200','-200')) {
    & $py train_step2.py --task stack_block --model event_tcn --data-dir data/stack_block --output-dir checkpoints --epochs 30 --batch-size 16 --seq-len 128 --seed 42 --time-shift-ms $ms *> "exp3_stack_shift_$ms.log"
}

Set-Content -LiteralPath 'done_marker.txt' -Value 'H3_STACK_SHIFT_DONE'
