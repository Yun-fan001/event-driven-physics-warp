"""Debug 5: stack_block annotation vs GT direct comparison."""
from __future__ import annotations
import numpy as np
from event_annotation_v2 import annotate_stack_block, AnnotationConfig

d = np.load("data/test2/stack/traj_0000.npz")
print("keys:", list(d.keys()))
gt = d["contact_states_gt"]
auto = d["contact_states_auto"]
forces = d["forces"]
gz = d["gripper_z"]
bp = d["block_pos"]
print("gt  counts:", {int(c): int((gt == c).sum()) for c in set(gt)})
print("auto counts:", {int(c): int((auto == c).sum()) for c in set(auto)})
print("frame acc:", float((gt == auto).mean()))
print("force mag min/max:", float(np.linalg.norm(forces, axis=1).min()),
      float(np.linalg.norm(forces, axis=1).max()))
print("gripper_z min/max:", float(gz.min()), float(gz.max()))
print("block_z min/max:", float(bp[:, 2].min()), float(bp[:, 2].max()))
print("block_x min/max:", float(bp[:, 0].min()), float(bp[:, 0].max()))

pred, g_ev, r_ev = annotate_stack_block(forces, gz, bp, AnnotationConfig())
print("recomputed counts:", {int(c): int((pred == c).sum()) for c in set(pred)})
print("recomputed acc:", float((gt == pred).mean()))
print("recomputed == stored:", bool((pred == auto).all()))

# where do gt/auto disagree
diff = np.where(gt != auto)[0]
print("first/last mismatch frames:", diff[:5], diff[-5:])
# gt transition frames
print("gt NoC->Grip:", np.where((gt[:-1] == 0) & (gt[1:] == 1))[0] + 1)
print("gt Grip->Release:", np.where((gt[:-1] == 1) & (gt[1:] == 2))[0] + 1)
print("auto NoC->Grip:", np.where((auto[:-1] == 0) & (auto[1:] == 1))[0] + 1)
print("auto Grip->Release:", np.where((auto[:-1] == 1) & (auto[1:] == 2))[0] + 1)
# print a few frames around gt transitions
for t in (250, 400, 550, 700, 850, 1000, 1150):
    if t < len(gt):
        print(f"t={t}: gt={gt[t]} auto={auto[t]} fz={np.linalg.norm(forces[t]):.2f} "
              f"gz={gz[t]:.3f} bz={bp[t,2]:.3f} bx={bp[t,0]:.3f}")
