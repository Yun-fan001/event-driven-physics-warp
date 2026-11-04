# Full dataset collection (background job)
# 3 tasks x 200 trajectories + final annotation report
# Estimated: ~2.5-3 h total with render-interval 5
$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'

Write-Output "[$(Get-Date -Format o)] start stick_slip collection (200 eps)"
& $py collect_data_step2.py --task stick_slip --num-episodes 200 --output-dir data/stick_slip --render-interval 5 *> collect_stick_slip.log
Write-Output "[$(Get-Date -Format o)] start peg_in_hole collection (200 eps)"
& $py collect_data_step2.py --task peg_in_hole --num-episodes 200 --output-dir data/peg_in_hole --render-interval 5 *> collect_peg.log
Write-Output "[$(Get-Date -Format o)] start stack_block collection (200 eps)"
& $py collect_data_step2.py --task stack_block --num-episodes 200 --output-dir data/stack_block --render-interval 5 *> collect_stack.log
Write-Output "[$(Get-Date -Format o)] annotation quality report"
& $py annotation_report.py --task all --data-dir data --output data/annotation_report_all.json *> report_all.log
Set-Content -LiteralPath 'done_marker.txt' -Value 'BG_COLLECT_DONE'
Write-Output "[$(Get-Date -Format o)] ALL DONE"
