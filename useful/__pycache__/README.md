# Robotics Lab 2 – UFACTORY Lite 6 golf swing

Code for Tasks 1, 2 and 4 of Robotics Lab Assignment 2. The robot holds a (virtual) club and swings it through a ball at **B = (0.25, 0, 0.15) m**.

---

## 1. Folder layout

```
Robotics-Lab-2/                  <- run the commands from HERE
├── useful/
│   ├── kinematics_2.py              Task 1 – DH table + symbolic forward kinematics (SymPy)
│   ├── robot_math.py                Task 2 – look-at matrix, IK, clamping, waypoints, timing
│   ├── main_swing.py                Task 2 – builds the final swing (spline + IK + velocity clamp)
│   ├── lite6_sim.py                 MuJoCo 3D simulation of the swing
│   ├── validation.py                Task 2 – recomputes every number of the results + safety checks
│   ├── report_figures.py            Task 2 – makes all figures of the report
│   ├── task_4_v1.py                 Task 4 – accuracy of the xArm built-in IK
│   ├── trajectory_following.py      sends the swing to the real robot (velocity control)
│   └── trajectory_following_mock.py same, with a fake robot (no hardware needed)
└── figures/                     created by report_figures.py
```

### How the files depend on each other

```
kinematics_2.py  ──►  robot_math.py  ──►  main_swing.py  ──►  lite6_sim.py
   (FK, Task 1)        (IK, waypoints,     (final swing)       validation.py
                        timing, clamps)                        report_figures.py
                                                               task_4_v1.py
                                                               trajectory_following(_mock).py
```

Every script after `main_swing.py` **imports it**. Importing it runs the whole swing (waypoints → timing → spline → IK → velocity clamp), so all scripts always use the same trajectory. If you change a parameter in `robot_math.py` or `main_swing.py`, every result, figure and simulation updates automatically.

---

## 2. Installation

Python 3.9 or newer.

```bash
pip install numpy sympy scipy matplotlib     # needed by everything
pip install mujoco                           # only for lite6_sim.py (and section 11 of validation.py)
pip install xarm-python-sdk                  # only for the real robot (task_4_v1 MODE="robot", trajectory_following.py)
```

---

## 3. What to run, in order

All commands are run **from the `Robotics-Lab-2` folder** (the folder that contains `useful/`). The files use `import useful.xxx`, so they must be run as modules (`python -m useful.xxx`).

| # | Command | What it does | Needs |
|---|---------|--------------|-------|
| 1 | `python -m useful.main_swing` | Builds the swing, prints the waypoints and the max joint speed, opens the 3 plots (3D path, joint speeds, tip speed) | – |
| 2 | `python -m useful.lite6_sim` | Plays the swing on the real Lite 6 model in 3D | `mujoco`, internet the 1st time |
| 3 | `python -m useful.validation` | Prints every number of the Results / Safety sections + PASS/FAIL table | – |
| 4 | `python -m useful.report_figures` | Saves all report figures into `figures/` | – |
| 5 | `python -m useful.task_4_v1` | Task 4 – built-in IK vs our FK | robot for `MODE="robot"` |
| 6 | `python -m useful.trajectory_following_mock` | Dry run of sending the swing to the robot | – |
| 7 | `python -m useful.trajectory_following` | **Moves the real robot** | robot + SDK |

`kinematics_2.py` and `robot_math.py` are libraries: they are not run on their own (`kinematics_2.py` only prints "Starting kinematics calculation script..." when imported, and the SymPy simplification takes a few seconds).

---

## 4. Details per file

### `kinematics_2.py` – Task 1
- Standard DH table of the Lite 6 (lengths in m): `a = [0, 0.2, 0.087, 0, 0, 0]`, `d = [0.2433, 0, 0, 0.2276, 0, 0.0615]`, offsets −π/2 on J2 and J3.
- Gives `T_0_6` (flange pose) and `X_tip, Y_tip, Z_tip` (club tip, club length `L_CLUB`).
- Check: all joints at 0 → flange at (0.087, 0, 0.1542) m = the robot's home pose (87, 0, 154.2) mm.

### `robot_math.py` – Task 2 toolbox
Parameters you might want to change:

| Name | Value | Meaning |
|------|-------|---------|
| `L_CLUB_VAL` | 0.0 m | club length (0 → the flange is the club tip) |
| `MAX_JOINT_VELOCITY` | π rad/s | joint speed limit |
| `MAX_REACH_SAFETY` | 0.95 | use only 95 % of the reach (measured from the shoulder) |
| `find_all_waypoints(..., lean_deg=60, radius=0.15)` | | swing pivot c: 0.15 m from the ball, leaning 60° towards the robot |
| `find_time_steps_from_waypoints(v_slow=π/4, v_max=3π/4)` | | nominal speeds for backswing / downswing |

Main functions: `get_matrix_look_at` (club orientation from the pivot), `solve_ik_with_bounds` (closed-form IK), `clamp_waypoint` (move unreachable waypoints), `find_all_waypoints` (P0…P3), `find_time_steps_from_waypoints` + `find_waypoint_velocities` (timing, fastest at P2), `clamp_velocity` (slows everything down if a joint goes faster than π rad/s).

### `main_swing.py` – the final swing
- Ball: `ball_xyz = [0.25, 0, 0.15]` (change it here).
- Variables used by the other scripts: `tips_clamped` (P0…P3), `t_corners` (times at P0…P3), `tq` (100 time samples), `path_xyz` (club tip path), `joint_trajectory` (100 × 6 joint angles, rad), `joint_velocities`, `tip_speed`.
- Expected output: P0 = (0.25, −0.05, 0.16), P1 = (0.13, −0.15, 0.25), P2 = (0.25, 0, 0.15), P3 = (0.13, 0.15, 0.25); velocity clamp ×3.53; max joint speed 2.99 rad/s → `Feasible: True`.
- The plot window **blocks** the script: close it to finish.

### `lite6_sim.py` – MuJoCo simulation
- The first run downloads the official Lite 6 model (MuJoCo Menagerie) into `~/.cache/lite6_model/`; after that it works offline.
- Shows: green = planned club-tip path, see-through gold sphere = ball, waypoints P0 blue, P1 orange, P2 green, P3 red. The swing repeats until you close the window.
- **macOS:** the MuJoCo viewer must be started with `mjpython -m useful.lite6_sim` instead of `python`.
- It only *plays* `joint_trajectory` (no physics), so it shows exactly what the IK gives.

### `validation.py` – results + safety (report sections 2.6–2.7)
Prints, in 12 sections: design parameters, pivot c, waypoints (clamped or not), nominal timing, the impact pose step by step (worked example of the IK), velocity clamping (α = 3.53), feasibility (speed, reach, joint limits, θ5, IK→FK error, lowest tip, peak tip speed), IK round trip on 3000 random points, the two clamping examples, the lean-angle table, our DH model vs the official MuJoCo model, and a **PASS/FAIL summary** (all 7 should PASS).
- Section 11 needs `mujoco` and needs `lite6_sim.py` to have been run once (for the model); otherwise it prints "skipped".

### `report_figures.py` – report figures
Saves into `Robotics-Lab-2/figures/`: `Pipeline.png`, `Clamping.png`, `Waypoint_construction.png`, `Visualise_pivot.png`, `IK_step3.png`, `Spline_timing.png`, `3d_path.png`, `joint_vel.png`, `tip_vel.png` (same names as in the LaTeX). Only numpy, matplotlib and scipy.
`mujoco_swing.png` is a screenshot taken from `lite6_sim.py`.

### `task_4_v1.py` – Task 4 (built-in IK accuracy)
For each of the 100 points of the swing: desired flange pose → **xArm built-in IK** → joint angles → **our FK (Task 1)** and the xArm FK → position error (mm) and orientation error (°).

Set `MODE` at the top of the file:

| MODE | When | What happens |
|------|------|--------------|
| `"test"` | at home, no robot | our own IK replaces the xArm IK → errors ≈ 0. **Only checks that the code works; this is NOT the Task 4 result.** |
| `"robot"` | at the lab | connects to `XARM_IP = "192.168.1.158"`, asks the controller for IK/FK (**the arm does not move**), saves everything to `useful/task4_joints.npy` |
| `"load"` | at home, after the lab | redoes the analysis from `task4_joints.npy` without the robot |

Outputs: error statistics (max / mean / RMS) in the terminal and the plot `useful/task4_errors.png`.
At the lab: check the robot's IP (in xArm Studio) and that the computer is on the same network. MuJoCo **cannot** replace the robot here: the built-in IK runs on the robot's controller.

### `trajectory_following.py` / `trajectory_following_mock.py` – send the swing to the robot
1. Moves slowly (20°/s) to the first pose of the swing, in position mode.
2. Switches to joint-velocity mode (mode 4) and streams `joint_velocities` every `dt`.
3. Always stops (zero velocity, back to mode 0) at the end, even after an error.

`speed_scale = 0.5` plays the swing at half speed (twice as long). Start with the **mock** to check the stream, then the real robot with someone at the emergency stop.

---

## 5. Known issues to fix before running

1. **`trajectory_following.py` and `trajectory_following_mock.py` import `main_swing_0`**, which is not in the folder. Change line 4 to:
   ```python
   from useful.main_swing import joint_trajectory, joint_velocities, tq
   ```
   and run them with `python -m useful.trajectory_following_mock`.
2. **`task_4_v1.py`**: the docstring says `python -m useful.task_4`, but the file is called `task_4_v1.py`. Use `python -m useful.task_4_v1` (or rename the file to `task_4.py`). In `MODE="test"` the plot legend still says "xArm IK", although the angles come from our IK.
3. `main_swing.py` opens its plot window when **imported**. The other scripts switch matplotlib to `Agg` before importing it, so nothing pops up. If you write a new script, do the same:
   ```python
   import matplotlib
   matplotlib.use("Agg")
   ```

---

## 6. Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'useful'` | run from `Robotics-Lab-2` with `python -m useful.<file>` (no `.py`) |
| `ValueError: OUT OF REACH` / `JOINT LIMIT` | a point is unreachable; move the ball closer to (0.25, 0, 0.15) |
| `waypoint ... unreachable` | clamping could not fix a waypoint; reduce the offsets in `find_all_waypoints` |
| MuJoCo window does not open on macOS | use `mjpython` instead of `python` |
| `urllib.error.URLError` in `lite6_sim.py` | no internet on the first run (the model is downloaded once) |
| Task 4: `Connection failed with xArm` | robot off, wrong IP, or not on the robot's network |
