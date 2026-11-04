"""Debug 4: inspect fresh npz, compare GT vs auto directly."""
from __future__ import annotations
import numpy as np
from event_annotation_v2 import annotate_stick_slip, AnnotationConfig

d = np.load("data/test2/traj_0000.npz")
print("keys:", list(d.keys()))
gt = d["contact_states_gt"]
auto = d["contact_states_auto"]
forces = d["forces"]
vels = d["velocities"]
print("gt unique/counts:", {int(c): int((gt == c).sum()) for c in set(gt)})
print("auto unique/counts:", {int(c): int((auto == c).sum()) for c in set(auto)})
print("frame acc:", float((gt == auto).mean()))

# recompute annotation here
pred, ev = annotate_stick_slip(forces[:, :2], vels[:, 0], AnnotationConfig())
print("recomputed unique/counts:", {int(c): int((pred == c).sum()) for c in set(pred)})
print("recomputed frame acc:", float((gt == pred).mean()))
print("recomputed == stored auto:", bool((pred == auto).all()))
print("recomputed slip events:", np.where(ev)[0])
print("gt stick->slip transitions:", np.where((gt[:-1] == 1) & (gt[1:] == 2))[0] + 1)
print("auto stick->slip transitions:", np.where((auto[:-1] == 1) & (auto[1:] == 2))[0] + 1)

# force signal check
print("force mag min/max:", float(np.linalg.norm(forces, axis=1).min()),
      float(np.linalg.norm(forces, axis=1).max()))
print("vel_x min/max:", float(vels[:, 0].min()), float(vels[:, 0].max()))
