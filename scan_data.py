"""Scan all collected npz files for validity (corruption check)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

BAD = []
for task in ("stick_slip", "peg_in_hole", "stack_block"):
    files = sorted(Path("data") / task / "*.npz".replace("*", "")) if False else sorted((Path("data") / task).glob("*.npz"))
    n = len(files)
    for f in files:
        try:
            d = np.load(f)
            t = d["forces"].shape[0]
            ok = (d["frames"].shape[0] > 0 and d["contact_states_gt"].shape[0] == t)
            ri = int(d.get("render_interval", 1))
            if not ok:
                BAD.append((str(f), "shape mismatch"))
        except Exception as e:
            BAD.append((str(f), str(e)[:80]))
    print(f"{task}: {n} files")

print(f"\nBAD files: {len(BAD)}")
for f, why in BAD[:20]:
    print(f"  {f}: {why}")
sys.exit(1 if BAD else 0)
