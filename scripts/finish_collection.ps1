$ErrorActionPreference = "Continue"
Set-Location 'D:\projects\event-driven-physics-warp'
$py = '.venv\Scripts\python.exe'
& $py collect_data_step2.py --task stick_slip --num-episodes 30 --output-dir data/stick_slip --render-interval 5 --seed-start 170 *> collect_finish.log
& $py annotation_report.py --task all --data-dir data --output data/annotation_report_all.json *> report_all.log
& $py scan_data.py *> scan_data.log
Set-Content -LiteralPath 'done_marker.txt' -Value 'FINISH_DONE'
