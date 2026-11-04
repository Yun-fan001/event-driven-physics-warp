"""Debug sensors: force vs touch vs framelinvel, and pusher geometry."""
from __future__ import annotations
import numpy as np
import mujoco
from env_peg_in_hole import PegInHoleEnv
from env_stick_slip import StickSlipEnv

print("=== peg sensor names/adrs ===")
m = mujoco.MjModel.from_xml_path("assets/peg_in_hole.xml")
for i in range(m.nsensor):
    print(f"  sensor[{i}] name={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)} "
          f"adr={m.sensor_adr[i]} dim={m.sensor_dim[i]}")

print("\n=== peg descent: qpos/qvel/sensordata/touch ===")
env = PegInHoleEnv()
env.reset(options={"init_z": 0.15})
for step in range(0, 120, 20):
    for _ in range(20):
        env.step(np.array([1.0], dtype=np.float32))
    print(f"step={step+20} peg_z={env.data.qpos[env._peg_z_qpos_adr]:.4f} "
          f"qvel={env.data.qvel[0]:.4f} "
          f"sensordata_vel={np.round(env.data.sensordata[env._vel_adr:env._vel_adr+3], 4)} "
          f"touch={env.data.sensordata[env._touch_adr]:.1f} "
          f"force_sensor={np.round(env.data.sensordata[env._force_adr:env._force_adr+3], 2)}")

print("\n=== peg ncon / contact pairs at end ===")
print("ncon =", env.data.ncon)
for i in range(env.data.ncon):
    c = env.data.contact[i]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    tmp = np.zeros(6)
    mujoco.mj_contactForce(env.model, env.data, i, tmp)
    print(f"  contact[{i}]: {g1} <-> {g2} force={np.round(tmp, 3)}")
env.close()

print("\n=== stick_slip geometry ===")
ss = StickSlipEnv()
ss.reset()
pusher_body_id = ss.model.body("pusher").id
print(f"pusher body pos = {ss.model.body_pos[pusher_body_id]}")
print(f"pusher tip site pos = {ss.model.site('pusher_tip').pos}")
print(f"push_base={ss._push_base:.4f} push_top={ss._push_top:.4f} pusher_qpos={ss.data.qpos[ss._pusher_qpos_adr]:.4f}")
print(f"block_x0={ss.data.qpos[ss._slider_qpos_adr]:.4f}")
ss.close()
