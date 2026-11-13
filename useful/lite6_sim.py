# TASK 2

# lite6_sim.py - play my joint_trajectory on the Lite 6 in MuJoCo
# shows the official Lite 6 model, the planned club tip path (green), the ball (gold) and the waypoints P0..P3
# the first run downloads the robot model to ~/.cache/lite6_model (needs internet once)
# run from Robotics-Lab-2 :  python -m useful.lite6_sim      (on macOS: mjpython -m useful.lite6_sim)

import os
import sys
import time
import urllib.request
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")  # main_swing calls plt.show(), this stops it from blocking

# so "import useful.xxx" works when this file is inside useful/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from useful.main_swing import joint_trajectory, tq, path_xyz, ball_xyz, tips_clamped

# --- download the robot model once (official UFACTORY meshes from MuJoCo Menagerie) ---
URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/ufactory_lite6/"
FOLDER = os.path.expanduser("~/.cache/lite6_model")
FILES = ["lite6.xml"] + [f"assets/{kind}/{name}.stl" for kind in ("visual", "collision")
                         for name in ("link_base", "link1", "link2", "link3", "link4", "link5", "link6")]
for f in FILES:
    path = os.path.join(FOLDER, f)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", f)
        urllib.request.urlretrieve(URL + f, path)

# --- scene: robot + floor + light ---
SCENE = """
<mujoco>
  <include file="lite6.xml"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.6 0.66" rgb2="0.2 0.22 0.26" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.42 0.45 0.5" rgb2="0.36 0.39 0.44" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0.6 -0.6 1.6" dir="-0.35 0.35 -1" directional="true"/>
    <geom type="plane" size="1 1 0.01" material="grid" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""
with open(os.path.join(FOLDER, "scene.xml"), "w") as f:
    f.write(SCENE)

model = mujoco.MjModel.from_xml_path(os.path.join(FOLDER, "scene.xml"))
data = mujoco.MjData(model)


def add_sphere(scn, pos, radius, rgba):
    '''
    goal: draw a sphere in the viewer (only drawn, no physics)
    PARAMS:
        scn: scene of the viewer (viewer.user_scn)
        pos: 3D position of the centre (m)
        radius: radius of the sphere (m)
        rgba: colour [red, green, blue, transparency], each between 0 and 1
    '''
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0],
                        pos, np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def add_line(scn, p0, p1, rgba):
    '''
    goal: draw a thin line (capsule) between two points in the viewer
    PARAMS:
        scn: scene of the viewer (viewer.user_scn)
        p0, p1: 3D positions of the two ends (m)
        rgba: colour [red, green, blue, transparency], each between 0 and 1
    '''
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p0, p1)
    scn.ngeom += 1


dt = tq[1] - tq[0]
with mujoco.viewer.launch_passive(model, data) as viewer:
    # draw the planned club-tip path (green) and the ball (gold) once
    for i in range(len(path_xyz) - 1):
        add_line(viewer.user_scn, path_xyz[i], path_xyz[i + 1], [0.1, 0.9, 0.2, 1])
    add_sphere(viewer.user_scn, ball_xyz, 0.0214, [1, 0.84, 0, 0.4])   # see-through, so P2 shows inside it

    # draw the 4 waypoints P0..P3 (same colours as the matplotlib 3D plot)
    waypoint_colors = [[0.12, 0.47, 0.71, 1],   # P0 blue   - address
                       [1.00, 0.50, 0.05, 1],   # P1 orange - top of backswing
                       [0.17, 0.63, 0.17, 1],   # P2 green  - impact (ball)
                       [0.84, 0.15, 0.16, 1]]   # P3 red    - follow-through
    for P, color in zip(tips_clamped, waypoint_colors):
        add_sphere(viewer.user_scn, P, 0.012, color)

    while viewer.is_running():
        for q in joint_trajectory:          # play the swing
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
        time.sleep(1.0)                     # pause, then repeat