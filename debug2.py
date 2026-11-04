"""Debug 2: touch sensor semantics, peg contact, stack release."""
from __future__ import annotations
import numpy as np
import mujoco

print("=== 1. touch sensor: direct contact test ===")
xml = """
<mujoco>
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="floor" pos="0 0 0"><geom name="fgeom" type="plane" size="1 1 0.1"/></body>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom name="ballgeom" type="sphere" size="0.05" mass="0.1"/>
      <site name="balltip" pos="0 0 -0.05" size="0.005"/>
    </body>
  </worldbody>
  <sensor><touch name="t1" site="balltip"/></sensor>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
print("  t=0  touch =", d.sensordata[0], " ball_z =", round(d.qpos[2], 4))
for i in range(250):
    mujoco.mj_step(m, d)
print("  t=250 touch =", d.sensordata[0], " ball_z =", round(d.qpos[2], 4))

print("\n=== 2. peg contact force at full insertion ===")
from env_peg_in_hole import PegInHoleEnv
env = PegInHoleEnv()
env.reset(options={"init_z": 0.15})
for step in range(400):
    obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
print(f"  final peg_z={env.data.qpos[env._peg_z_qpos_adr]:.4f} ncon={env.data.ncon}")
for i in range(env.data.ncon):
    c = env.data.contact[i]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    tmp = np.zeros(6)
    mujoco.mj_contactForce(env.model, env.data, i, tmp)
    print(f"  contact[{i}]: {g1} <-> {g2} force={np.round(tmp, 3)}")
print(f"  info force={info['force']} touch={info['touch']} state={info['contact_state']}")
env.close()

print("\n=== 3. stack block: phase timeline ===")
from env_stack_block import StackBlockEnv
env = StackBlockEnv()
env.reset(options={"friction_scale": 1.0})
last_phase = -1
for step in range(600):
    obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    if info["phase"] != last_phase:
        last_phase = info["phase"]
        print(f"  step={step} phase={last_phase} state={info['contact_state']} "
              f"gripper_z={info['gripper_z']:.4f} block_pos={np.round(info['block_pos'], 4)}")
print(f"  final state={info['contact_state']} gripper_z={info['gripper_z']:.4f} "
      f"block_pos={np.round(info['block_pos'], 4)}")
env.close()
