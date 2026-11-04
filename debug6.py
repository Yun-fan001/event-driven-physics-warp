"""Debug 6: measure velocity ramp at slip onset vs annotation thresholds."""
from __future__ import annotations
import numpy as np
from event_annotation_v2 import annotate_stick_slip, AnnotationConfig

for fname in ("traj_0000", "traj_0002", "traj_0004"):
    d = np.load(f"data/smoke/stick_slip/{fname}.npz")
    gt = d["contact_states_gt"]
    vx = np.abs(d["velocities"][:, 0])
    force_mag = np.linalg.norm(d["forces"], axis=1)
    gt_ev = np.where((gt[:-1] == 1) & (gt[1:] == 2))[0] + 1
    if len(gt_ev) == 0:
        print(f"{fname}: no GT slip event")
        continue
    g = int(gt_ev[0])
    print(f"\n{fname}: GT slip at frame {g}")
    print(f"  vx around event: " + " ".join(
        f"{t}:{vx[g + t]:.4f}" for t in range(-6, 10)))
    print(f"  force around:   " + " ".join(
        f"{t}:{force_mag[g + t]:.2f}" for t in range(-6, 10)))
    for th in (0.010, 0.011, 0.012, 0.013, 0.015):
        cross = np.where(vx[g - 10:g + 20] > th)[0]
        first = g - 10 + cross[0] if len(cross) else None
        print(f"  vx first > {th}: frame {first} (offset {first - g if first else None})")

    pred, ev = annotate_stick_slip(
        d["forces"][:, :2], d["velocities"][:, 0], AnnotationConfig())
    ev_frames = np.where(ev)[0]
    print(f"  annotation slip events: {ev_frames.tolist()}")
