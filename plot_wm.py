"""汇总 event_wm / force_wm / implicit_wm 三路世界模型对照结果。"""
import json
from pathlib import Path

ROOT = Path(__file__).parent / "checkpoints_wm"
MODELS = ["event_wm", "force_wm", "implicit_wm"]
SPLITS = ["id", "ood"]

rows = []
for m in MODELS:
    p = ROOT / f"{m}_results.json"
    if not p.exists():
        print(f"[missing] {p}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.append(d)

if not rows:
    raise SystemExit("no results")

print("\n=== 事件序列自回归世界模型 · 三路对照 (stick_slip) ===")
print("FF=首次失效帧数(128=全对), evF1=事件F1(±5帧), fMSE=力重构MSE")
hdr = (f"{'model':<14}{'split':<6}{'n_win':>7}{'n_slip':>7}{'FF_mean':>9}"
       f"{'FF_slip':>9}{'evF1':>7}{'evF1_slip':>10}{'fMSE':>8}")
print(hdr)
print("-" * len(hdr))
for d in rows:
    for s in SPLITS:
        r = d[s]
        fmse = f"{r.get('force_mse', float('nan')):8.3f}" if "force_mse" in r else "     n/a"
        ff_slip = f"{r.get('first_failure_slip', float('nan')):9.1f}"
        f1_slip = f"{r.get('event_f1_slip', float('nan')):10.3f}"
        print(f"{d['model']:<14}{s:<6}{r['n_windows']:>7}{r['n_slip_windows']:>7}"
              f"{r['first_failure_mean']:>9.1f}{ff_slip}{r['event_f1']:>7.3f}"
              f"{f1_slip}{fmse}")

print("\n=== 跨摩擦泛化 (OOD - ID) ===")
print(f"{'model':<14}{'dFF_mean':>10}{'dFF_slip':>10}{'dEvF1':>8}{'dEvF1_slip':>12}")
for d in rows:
    ood, idd = d["ood"], d["id"]
    d1 = ood["first_failure_mean"] - idd["first_failure_mean"]
    d2 = ood.get("first_failure_slip", 0) - idd.get("first_failure_slip", 0)
    d3 = ood["event_f1"] - idd["event_f1"]
    d4 = ood.get("event_f1_slip", 0) - idd.get("event_f1_slip", 0)
    print(f"{d['model']:<14}{d1:>10.2f}{d2:>10.2f}{d3:>8.3f}{d4:>12.3f}")

print("\n=== 干预: 摩擦+50% 滑移起始单调后移 (仅GT含转移窗口) ===")
for d in rows:
    iv = d.get("intervention", {})
    for s in SPLITS:
        r = iv.get(s, {})
        print(f"{d['model']:<14}{s:<6}n={r.get('n_gt_slip_windows')} "
              f"correct={r.get('direction_correct')} wrong={r.get('direction_wrong')} "
              f"neutral={r.get('direction_neutral')} rate={r.get('direction_rate')}")

