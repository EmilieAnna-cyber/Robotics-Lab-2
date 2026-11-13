# TASK 2 - RESULTS AND SAFETY CHECKS (sections 2.6 and 2.7 of the report)

# validation.py
# recomputes every number written in the results of the report and checks the safety
# run from inside useful/ :  python validation.py     or from Robotics-Lab-2 :  python -m useful.validation
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # main_swing calls plt.show(), this stops it from blocking
from scipy.interpolate import CubicHermiteSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "import useful.xxx" works
import useful.robot_math as rm
import useful.main_swing as ms  # runs the whole swing (waypoints, timing, IK, velocity clamp)

np.set_printoptions(suppress=True)

# DH lengths (m), same as kinematics_2.py
a2 = 0.2
a3 = 0.087
d1 = 0.2433
d4 = 0.2276
d6 = 0.0615
R = np.sqrt(a3**2 + d4**2)        # straight line elbow -> wrist centre
alpha_R = np.arctan2(a3, d4)      # fixed angle of R (called alpha in the report)

ball = ms.ball_xyz
pivot = rm.SWING_PIVOT.copy()     # set by find_all_waypoints() inside main_swing
waypoints = ms.tips_clamped
reach_allowed = rm.MAX_REACH * rm.MAX_REACH_SAFETY   # 0.95 * (a2 + R)


# =============================================================================
# 1. DESIGN PARAMETERS
# =============================================================================
print("\n=== 1. DESIGN PARAMETERS ===")
print("ball B             =", ball)
print("club length        =", rm.L_CLUB_VAL, "m")
print("joint speed limit  = %.4f rad/s (pi)" % rm.MAX_JOINT_VELOCITY)
print("max reach a2 + R   = %.4f m,  allowed (95%%) = %.4f m" % (rm.MAX_REACH, reach_allowed))


# =============================================================================
# 2. STEP A - SWING PIVOT   c = B + r (sin(lambda) i + cos(lambda) u)
# =============================================================================
print("\n=== 2. STEP A - SWING PIVOT ===")
r = 0.15                     # radius in find_all_waypoints
lam = np.radians(60)         # lean_deg in find_all_waypoints
towards_robot = r * np.sin(lam) * np.array([-1, 0, 0])
up = r * np.cos(lam) * np.array([0, 0, 1])
pivot_by_hand = ball + towards_robot + up
print("r sin(lambda) towards robot =", towards_robot.round(4))
print("r cos(lambda) up            =", up.round(4))
print("c by hand                   =", pivot_by_hand.round(4))
print("c from robot_math           =", pivot.round(4))
print("distance ball - pivot       = %.4f m (must be r = %.2f)" % (np.linalg.norm(ball - pivot), r))


# =============================================================================
# 3. STEP B - WAYPOINTS (and was one of them clamped?)
# =============================================================================
print("\n=== 3. STEP B - WAYPOINTS ===")
s_dir = np.array([0, 1, 0])    # hitting direction
i_dir = np.array([-1, 0, 0])   # towards the robot
u_dir = np.array([0, 0, 1])    # up
# waypoints straight from the formulas, before clamping
P0 = ball - 0.05*s_dir + 0.01*u_dir
P1 = ball - 0.15*s_dir + 0.12*i_dir + 0.10*u_dir
P2 = ball
P3 = ball + 0.15*s_dir + 0.12*i_dir + 0.10*u_dir
formulas = [P0, P1, P2, P3]
names = ["address", "top of backswing", "impact", "follow-through"]
for k in range(4):
    clamped = not np.allclose(formulas[k], waypoints[k])   # different -> clamp_waypoint moved it
    print("P%d (%s) = %s   clamped: %s" % (k, names[k], waypoints[k].round(3), clamped))


# =============================================================================
# 4. STEP C - TIMING (before the velocity clamp)
# =============================================================================
print("\n=== 4. STEP C - TIMING (nominal) ===")
dist01 = np.linalg.norm(waypoints[1] - waypoints[0])   # P0 -> P1
dist12 = np.linalg.norm(waypoints[2] - waypoints[1])   # P1 -> P2
dist23 = np.linalg.norm(waypoints[3] - waypoints[2])   # P2 -> P3
t_nominal = rm.find_time_steps_from_waypoints(waypoints)
v_imp_nominal = 2 * dist12 / (t_nominal[2] - t_nominal[1])
print("distances d0, d1, d2 = %.4f, %.4f, %.4f m" % (dist01, dist12, dist23))
print("nominal times        =", t_nominal.round(3), "s")
print("nominal v_imp        = %.3f m/s  (3pi/4 = %.3f)" % (v_imp_nominal, 3*np.pi/4))


# =============================================================================
# 5. STEP D - POSE AT IMPACT, BY HAND (worked example of the IK)
# =============================================================================
print("\n=== 5. STEP D - POSE AT IMPACT (P2 = ball) ===")
T_tip = rm.get_matrix_look_at(ball[0], ball[1], ball[2])
x_tool = T_tip[:3, 0]
y_tool = T_tip[:3, 1]
z_tool = T_tip[:3, 2]
print("p - c  =", (ball - pivot).round(4))
print("z_tool =", z_tool.round(3))
print("x_tool =", x_tool.round(3))
print("y_tool =", y_tool.round(3))
print("T_tip(B) =\n", T_tip.round(3))
print("club angle from the vertical = %.1f deg (= lean angle)" % np.degrees(np.arccos(-z_tool[2])))

# step 1: wrist centre (flange = ball because the club length is 0)
wrist = rm.get_gripper_pos(ball) - d6 * z_tool
print("\nwrist centre =", wrist.round(4))

# step 2: base angle, then kx and kz in the arm plane
theta1 = np.arctan2(wrist[1], wrist[0])
kx = wrist[0]*np.cos(theta1) + wrist[1]*np.sin(theta1)
kz = wrist[2] - d1
print("theta1 = %.2f deg,  kx = %.4f m,  kz = %.4f m" % (np.degrees(theta1), kx, kz))

# step 3a: elbow with the law of cosines
D2 = kx**2 + kz**2
inner_angle = np.arccos((a2**2 + R**2 - D2) / (2*a2*R))
theta3 = inner_angle - alpha_R
theta3_code = np.arctan2(d4, a3) + np.arcsin((D2 - a2**2 - R**2) / (2*a2*R))   # form used in robot_math
print("R = %.4f m, alpha = %.1f deg, D^2 = %.4f m^2" % (R, np.degrees(alpha_R), D2))
print("theta3 = %.1f - %.1f = %.1f deg  (code form: %.1f deg)" % (np.degrees(inner_angle), np.degrees(alpha_R), np.degrees(theta3), np.degrees(theta3_code)))

# step 3b: shoulder with A and B
A = a2 + a3*np.sin(theta3) - d4*np.cos(theta3)
B = a3*np.cos(theta3) + d4*np.sin(theta3)
theta2 = np.arctan2(A*kx - B*kz, A*kz + B*kx)
print("A = %.4f, B = %.4f, atan2(%.4f, %.4f) -> theta2 = %.1f deg" % (A, B, A*kx - B*kz, A*kz + B*kx, np.degrees(theta2)))

# full IK from robot_math (also gives the wrist angles) and check with the FK of task 1
q_impact = rm.solve_ik_with_bounds(rm.get_gripper_pos(ball), tip_pos=ball)
print("\nIK at impact (deg) =", np.degrees(q_impact).round(1))
print("FK of these angles =", rm.get_ee_position(q_impact).round(5), " (should be the ball)")


# =============================================================================
# 6. STEP E - VELOCITY CLAMPING
# =============================================================================
print("\n=== 6. STEP E - VELOCITY CLAMPING ===")
# rebuild the swing with the nominal times, same steps as main_swing
vel = rm.find_waypoint_velocities(waypoints, t_nominal)
spline_x = CubicHermiteSpline(t_nominal, waypoints[:, 0], vel[:, 0])
spline_y = CubicHermiteSpline(t_nominal, waypoints[:, 1], vel[:, 1])
spline_z = CubicHermiteSpline(t_nominal, waypoints[:, 2], vel[:, 2])
tq_nominal = np.linspace(0, t_nominal[-1], 100)
path_nominal = np.column_stack([spline_x(tq_nominal), spline_y(tq_nominal), spline_z(tq_nominal)])
joints_nominal = []
for target in path_nominal:
    joints_nominal.append(rm.solve_ik_with_bounds(rm.get_gripper_pos(target), tip_pos=target))
joints_nominal = np.unwrap(np.array(joints_nominal), axis=0)
dq_nominal = np.abs(np.gradient(joints_nominal, tq_nominal, axis=0))

peak_nominal = dq_nominal.max()
fastest_joint = np.argmax(dq_nominal.max(axis=0)) + 1
alpha = 1.05 * peak_nominal / np.pi
v_imp = 2 * dist12 / (ms.t_corners[2] - ms.t_corners[1])
print("nominal fastest joint = %.2f rad/s (joint J%d)" % (peak_nominal, fastest_joint))
print("alpha = 1.05 * %.2f / pi = %.2f" % (peak_nominal, alpha))
print("new times             =", ms.t_corners.round(3), "s  (nominal * alpha =", (t_nominal*alpha).round(3), ")")
print("new v_imp             = %.3f m/s  (= %.3f / %.2f)" % (v_imp, v_imp_nominal, alpha))
print("new fastest joint     = %.2f rad/s  (= %.2f / %.2f)" % (np.abs(ms.dq).max(), peak_nominal, alpha))


# =============================================================================
# 7. FEASIBILITY OF THE FINAL TRAJECTORY
# =============================================================================
print("\n=== 7. FEASIBILITY OF THE FINAL TRAJECTORY ===")
q = ms.joint_trajectory         # final joint angles (already unwrapped)
dq = np.abs(ms.dq)              # joint speeds, same as in main_swing

# (i) velocity
max_speed = dq.max()
sample, joint = np.unravel_index(np.argmax(dq), dq.shape)
print("(i)   max joint speed      = %.2f rad/s, joint J%d at t = %.2f s" % (max_speed, joint + 1, ms.tq[sample]))

# (ii) reach: distance of the flange from the shoulder (0, 0, d1)
distances = []
for target in ms.path_xyz:
    flange = rm.get_gripper_pos(target)
    distances.append(np.linalg.norm(flange - np.array([0, 0, d1])))
max_distance = max(distances)
print("(ii)  max dist. from shoulder = %.3f m (allowed %.4f m)" % (max_distance, reach_allowed))

# (iii) joint limits
print("(iii) joint ranges (deg):")
joints_ok = True
for j in range(6):
    low, high = rm.JOINT_BOUNDS[j]
    q_min = q[:, j].min()
    q_max = q[:, j].max()
    inside = q_min >= low - 1e-9 and q_max <= high + 1e-9   # J4/J6 are unwrapped, their limit is +-360 deg
    if not inside:
        joints_ok = False
    print("      J%d: min %7.1f  max %7.1f   limit [%6.1f, %6.1f]  ok: %s" % (j + 1, np.degrees(q_min), np.degrees(q_max), np.degrees(low), np.degrees(high), inside))

# wrist singularity, accuracy of the IK, lowest point, tip speed
min_theta5 = np.degrees(np.min(np.abs(q[:, 4])))
fk_error = 0
for i in range(len(q)):
    error = np.linalg.norm(rm.get_ee_position(q[i]) - rm.get_gripper_pos(ms.path_xyz[i]))
    fk_error = max(fk_error, error)
lowest_tip = ms.path_xyz[:, 2].min()
fastest_sample = np.argmax(ms.tip_speed)
print("      min |theta5|         = %.1f deg (distance from wrist singularity)" % min_theta5)
print("      IK -> FK error       = %.2e m" % fk_error)
print("      lowest club tip      = %.3f m" % lowest_tip)
print("      peak tip speed       = %.3f m/s at t = %.2f s (impact at %.2f s)" % (ms.tip_speed.max(), ms.t_vel[fastest_sample], ms.t_corners[2]))
print("      tip speed first/last = %.3f / %.3f m/s" % (ms.tip_speed[0], ms.tip_speed[-1]))


# =============================================================================
# 8. IK ROUND TRIP ON 3000 RANDOM TARGETS
# =============================================================================
print("\n=== 8. IK ROUND TRIP ON 3000 RANDOM TARGETS ===")
rng = np.random.default_rng(0)
reachable = 0
worst_error = 0
for n in range(3000):
    tip = rng.uniform([-0.35, -0.35, 0.05], [0.35, 0.35, 0.5])   # random point in a box around the robot
    try:
        q_test = rm.solve_ik_with_bounds(rm.get_gripper_pos(tip), tip_pos=tip)
    except ValueError:
        continue   # not reachable -> skip it
    reachable += 1
    error = np.linalg.norm(rm.get_ee_position(q_test) - rm.get_gripper_pos(tip))
    worst_error = max(worst_error, error)
print("reachable targets: %d / 3000, worst FK(IK) error = %.2e m" % (reachable, worst_error))


# =============================================================================
# 9. CLAMPING EXAMPLES (figure of the clamping)
# =============================================================================
print("\n=== 9. CLAMPING EXAMPLES ===")
# same steps as clamp_waypoint, but we also print which stage fixed the point

# example 1: too far -> stage 1 (radial shrink)
p = np.array([0.45, 0.10, 0.35])
distance = np.linalg.norm(rm.get_gripper_pos(p) - np.array([0, 0, d1]))
for s in rm.RADIUS_STEPS:
    if rm._tip_is_feasible(p * s):
        print("p = %s: dist. from shoulder %.3f m -> stage 1, s = %.2f, p_new = %s" % (p, distance, s, (p*s).round(3)))
        break

# example 2: too close -> stage 1 fails, stage 2 (azimuth rotation) works
p = np.array([-0.092, -0.022, 0.283])
distance = np.linalg.norm(rm.get_gripper_pos(p) - np.array([0, 0, d1]))
stage1_works = False
for s in rm.RADIUS_STEPS:
    if rm._tip_is_feasible(p * s):
        stage1_works = True
rho = np.sqrt(p[0]**2 + p[1]**2)
phi = np.arctan2(p[1], p[0])
found = False
for dphi in rm.ANGLE_STEPS:
    for sign in [1, -1]:
        p_new = np.array([rho*np.cos(phi + sign*dphi), rho*np.sin(phi + sign*dphi), p[2]])
        if rm._tip_is_feasible(p_new):
            print("p = %s: dist. from shoulder %.3f m -> stage 1 works: %s, stage 2, dphi = %+.2f deg, p_new = %s"
                  % (p, distance, stage1_works, np.degrees(sign*dphi), p_new.round(3)))
            found = True
            break
    if found:
        break


# =============================================================================
# 10. CHOICE OF THE LEAN ANGLE
# =============================================================================
print("\n=== 10. CHOICE OF THE LEAN ANGLE ===")
print(" lean   min theta5   slowdown   swing time")
for lean in [0, 15, 30, 45, 60, 75]:
    wp = rm.find_all_waypoints(ball, lean_deg=lean)    # NOTE: this changes SWING_PIVOT
    t = rm.find_time_steps_from_waypoints(wp)
    vel = rm.find_waypoint_velocities(wp, t)
    spline_x = CubicHermiteSpline(t, wp[:, 0], vel[:, 0])
    spline_y = CubicHermiteSpline(t, wp[:, 1], vel[:, 1])
    spline_z = CubicHermiteSpline(t, wp[:, 2], vel[:, 2])
    tq = np.linspace(0, t[-1], 100)
    path = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
    joints = []
    for target in path:
        joints.append(rm.solve_ik_with_bounds(rm.get_gripper_pos(target), tip_pos=target))
    joints = np.unwrap(np.array(joints), axis=0)
    slowdown = 1.05 * np.abs(np.gradient(joints, tq, axis=0)).max() / np.pi
    print(" %3d    %6.1f      x%5.1f     %5.1f s" % (lean, np.degrees(np.min(np.abs(joints[:, 4]))), slowdown, slowdown * t[-1]))
rm.find_all_waypoints(ball)   # put the pivot back to the default lean (60 deg)


# =============================================================================
# 11. OUR DH MODEL vs THE OFFICIAL MUJOCO MODEL (optional)
# =============================================================================
print("\n=== 11. DH MODEL vs OFFICIAL LITE 6 MODEL (MuJoCo) ===")
# only works if mujoco is installed and lite6_sim.py was run once (it downloads the model)
try:
    import mujoco
    import sympy as sp
    from useful.kinematics_2 import T_0_6, theta1, theta2, theta3, theta4, theta5, theta6
    dh_rotation = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), T_0_6[:3, :3], "numpy")
    model = mujoco.MjModel.from_xml_path(os.path.expanduser("~/.cache/lite6_model/lite6.xml"))
    data = mujoco.MjData(model)
    flange_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rng = np.random.default_rng(0)
    max_pos_diff = 0
    max_rot_diff = 0
    for n in range(500):
        q_test = rng.uniform(-1.5, 1.5, 6)
        q_test[2] = abs(q_test[2])   # J3 must be >= -3.5 deg
        data.qpos[:] = q_test
        mujoco.mj_kinematics(model, data)
        pos_diff = np.linalg.norm(data.site_xpos[flange_site] - rm.get_ee_position(q_test))
        rot_diff = np.abs(data.site_xmat[flange_site].reshape(3, 3) - np.array(dh_rotation(*q_test), dtype=float)).max()
        max_pos_diff = max(max_pos_diff, pos_diff)
        max_rot_diff = max(max_rot_diff, rot_diff)
    print("500 random poses: max position difference = %.2f mm, max orientation difference = %.1e" % (max_pos_diff*1000, max_rot_diff))
except Exception as e:
    print("skipped (%s). Run lite6_sim.py once to download the model." % e)


# =============================================================================
# 12. SAFETY SUMMARY (section 2.7)
# =============================================================================
print("\n=== 12. SAFETY SUMMARY ===")
dt = ms.tq[1] - ms.tq[0]

def check(ok, text, value):
    '''
    goal: print one line of the safety summary
    PARAMS:
        ok: True if the check passes
        text: what is checked
        value: the value obtained (text)
    '''
    if ok:
        print("PASS  %-42s %s" % (text, value))
    else:
        print("FAIL  %-42s %s" % (text, value))

check(max_speed < np.pi,                   "(i)   joint speed below pi rad/s",          "%.2f rad/s" % max_speed)
check(max_distance <= reach_allowed,       "(ii)  inside the allowed reach",            "%.3f m" % max_distance)
check(joints_ok,                           "(iii) all joints inside their limits",      "see section 7")
check(min_theta5 > 30,                     "      far from wrist singularity (>30 deg)", "%.1f deg" % min_theta5)
check(fk_error < 1e-9,                     "      IK reproduces the path (<1e-9 m)",    "%.1e m" % fk_error)
check(lowest_tip >= ball[2] - 1e-9,        "      club tip never below the ball",       "%.3f m" % lowest_tip)
check(abs(ms.t_vel[fastest_sample] - ms.t_corners[2]) < 2*dt, "      fastest exactly at impact", "t = %.2f s" % ms.t_vel[fastest_sample])
print("\ncollisions with the robot itself are checked visually in lite6_sim.py (no formal collision check)")