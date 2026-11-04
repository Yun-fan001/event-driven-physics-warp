"""
Stage 0 smoke test: load all 3 MuJoCo XMLs and instantiate the 3 envs.
Pure-ASCII output to avoid GBK console encoding issues.
Records known defects (esp. Task 3 gripper_base mass crash).
"""

from __future__ import annotations

import sys
import traceback

import mujoco
import numpy as np


def load_xml(xml_path: str) -> str:
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return f"OK nbody={model.nbody} nq={model.nq} nv={model.nv}"
    except Exception:
        return "FAIL: " + traceback.format_exc(limit=3)


def test_env(env_name: str, make_fn, n_steps: int = 2500) -> str:
    try:
        env = make_fn()
        obs, info = env.reset()
        forces = []
        states = []
        vels = []
        touches = []
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
            forces.append(float(np.linalg.norm(info["force"])))
            states.append(int(info["contact_state"]))
            vels.append(float(np.linalg.norm(info["velocity"])))
            touches.append(float(info.get("touch", 0.0)))
            if terminated or truncated:
                break
        env.close()
        fmin, fmax = min(forces), max(forces)
        vmax = max(vels)
        st = np.array(states)
        trans = int(((st[:-1] == 1) & (st[1:] == 2)).sum())  # Stick->Slip
        trans_nc = int(((st[:-1] == 0) & (st[1:] == 1)).sum())  # NoContact->Stick
        counts = {int(c): int((st == c).sum()) for c in sorted(set(states))}
        touch_frac = float(np.mean(touches))
        return (f"OK steps={len(states)} force[min={fmin:.2f},max={fmax:.2f}]N vmax={vmax:.3f} "
                f"state_counts={counts} trans Stick->Slip={trans} NoC->Stick={trans_nc} "
                f"touch_frac={touch_frac:.2f}")
    except Exception:
        return "FAIL: " + traceback.format_exc(limit=3)


def main() -> None:
    print("=" * 60)
    print("Stage 0 smoke test: XML loading + env instantiation")
    print("=" * 60)

    print("\n[1] XML load tests:")
    for name in ("stick_slip", "peg_in_hole", "stack_block"):
        path = f"assets/{name}.xml"
        print(f"  {name:12s} -> {load_xml(path)}")

    print("\n[2] Env physics tests (full episodes):")
    from env_stick_slip import StickSlipEnv, EpisodeConfig as SSCfg
    from env_peg_in_hole import PegInHoleEnv
    from env_stack_block import StackBlockEnv

    for name, fn in (
        ("stick_slip", lambda: StickSlipEnv(config=SSCfg())),
        ("peg_in_hole", lambda: PegInHoleEnv()),
        ("stack_block", lambda: StackBlockEnv()),
    ):
        print(f"  {name:12s} -> {test_env(name, fn)}")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
