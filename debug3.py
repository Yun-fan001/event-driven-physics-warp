"""Debug 3: stick_slip touch sensor during contact."""
from __future__ import annotations
import numpy as np
import mujoco
from env_stick_slip import StickSlipEnv

env = StickSlipEnv()
env.reset(options={"friction_scale": 1.0, "push_speed": 0.02, "push_amplitude": 0.03})
print(f"push_base={env._push_base:.4f} push_top={env._push_top:.4f}")
for step in range(1500):
    obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    if step % 150 == 0 or info["in_contact"]:
        pass
    if step in (99, 199, 299, 399, 499, 749, 999, 1249, 1499):
        ncon = env.data.ncon
        pairs = []
        for i in range(ncon):
            c = env.data.contact[i]
            g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            pairs.append(f"{g1}-{g2}")
        print(f"step={step} touch={info['touch']} force={np.round(info['force'],2)} "
              f"block_pos={np.round(info['block_pos'],3)} "
              f"pusher_x={info['pusher_x']:.4f} target={info['pusher_target']:.4f} "
              f"state={info['contact_state']} ncon={ncon} pairs={pairs}")
env.close()
