# TASK 2
 
# main_swing.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline   
import useful.robot_math as rm
 
# =============================================================================
# 1. DEFINE THE BALL AND BUILD WAYPOINTS
# =============================================================================
ball_xyz = np.array([0.25, 0, 0.15])
 
tips_clamped = rm.find_all_waypoints(ball_xyz)
print("Clamped waypoints:")
for i, P in enumerate(tips_clamped):
    print(f"  P{i}: {P.round(3)}")
 
# =============================================================================
# 2. BUILD TRAJECTORY & CLAMP VELOCITY
# =============================================================================
t_corners = rm.find_time_steps_from_waypoints(tips_clamped)
t_corners = rm.clamp_velocity(tips_clamped, t_corners)
 
vel = rm.find_waypoint_velocities(tips_clamped, t_corners)   # FIX: 0 at P0/P1/P3, max at P2
spline_x = CubicHermiteSpline(t_corners, tips_clamped[:, 0], vel[:, 0])
spline_y = CubicHermiteSpline(t_corners, tips_clamped[:, 1], vel[:, 1])
spline_z = CubicHermiteSpline(t_corners, tips_clamped[:, 2], vel[:, 2])
 
tq = np.linspace(0.0, t_corners[-1], 100)
path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
 
# =============================================================================
# 3. SOLVE IK
# =============================================================================
joint_trajectory = []
for target in path_xyz:
    gripper_pos = rm.get_gripper_pos(target)
    q_sol = rm.solve_ik_with_bounds(gripper_pos)
    joint_trajectory.append(q_sol) 
joint_trajectory = np.unwrap(np.array(joint_trajectory), axis=0) #added unwrap to remove invalid velocity
 
dq = np.gradient(joint_trajectory, tq, axis=0) 
vmax = np.max(np.abs(dq))
print(f"Max joint velocity: {vmax:.2f} rad/s (limit {rm.MAX_JOINT_VELOCITY})")
print(f"Feasible: {vmax <= rm.MAX_JOINT_VELOCITY}")
 
# =============================================================================
# 4. COMPUTE VELOCITIES FOR PLOTTING
# =============================================================================
dt = tq[1] - tq[0]
joint_velocities = np.diff(joint_trajectory, axis=0) / dt
t_vel = tq[:-1]
tip_displacements = np.diff(path_xyz, axis=0)
tip_speed = np.linalg.norm(tip_displacements / dt, axis=1) 
 
# =============================================================================
# 5. PLOTS
# =============================================================================
fig = plt.figure(figsize=(20, 5))
 
# --- Left: 3D tip path ---
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], 'k-', lw=1.5, label='tip path')
for i, P in enumerate(tips_clamped):
    ax1.scatter(*P, s=80, label=f'P{i}')
ax1.scatter(*ball_xyz, color='gold', s=250, edgecolor='k', label='ball')
ax1.scatter(0, 0, 0, color='k', marker='^', s=100, label='base')
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_zlabel('z [m]')
ax1.set_title('3D tip path')
ax1.legend(fontsize=8)
 
# --- Middle: Joint velocities vs time ---
ax2 = fig.add_subplot(1, 3, 2)
for j in range(6):
    ax2.plot(t_vel, np.abs(joint_velocities[:, j]), label=f'θ{j+1}')
ax2.axhline(rm.MAX_JOINT_VELOCITY, color='r', linestyle='--', label=f'limit = {rm.MAX_JOINT_VELOCITY:.2f} rad/s')
for i, t_c in enumerate(t_corners):
    ax2.axvline(t_c, color='gray', linestyle=':', alpha=0.7)
    ax2.text(t_c, ax2.get_ylim()[1]*0.9, f'P{i}', ha='center', color='black')
ax2.set_xlabel('t [s]'); ax2.set_ylabel('|dq/dt| [rad/s]')
ax2.set_title('Joint velocities vs time'); ax2.legend(fontsize=8); ax2.grid(True)
 
# --- Right: Club tip Cartesian velocity vs time ---
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t_vel, tip_speed, color='purple', lw=2, label='Tip speed')
for i, t_c in enumerate(t_corners):
    ax3.axvline(t_c, color='gray', linestyle=':', alpha=0.7)
    ax3.text(t_c, ax3.get_ylim()[1]*0.9, f'P{i}', ha='center', color='black')
ax3.set_xlabel('t [s]'); ax3.set_ylabel('Speed [m/s]')
ax3.set_title('Club tip velocity vs time'); ax3.legend(fontsize=8); ax3.grid(True)
 
plt.tight_layout()
plt.show()
 
# =============================================================================
# 6. ACCURACY VERIFICATION (Task 4)
# =============================================================================
# This requires the robot or the xArm Studio simulator to be running.
# Uncomment the lines below when I install the xArm SDK 
 
'''
from xarm.wrapper import XArmAPI
 
# Connect to robot (change IP to your robot/simulator)
arm = XArmAPI('192.168.1.XXX', is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
 
recovered_positions = []
recovered_orientations = []
 
for i, target in enumerate(path_xyz):
    # 1. Convert the club-tip target to a flange target (using our helper)
    flange_target = rm.get_gripper_pos(target)
 
    # 2. Use the BUILT-IN IK solver
    # Note: we need a starting guess for the SDK, use the previous joint angles
    code, joints = arm.get_inverse_kinematics(
        [flange_target[0]*1000, flange_target[1]*1000, flange_target[2]*1000, 180, 0, 0], 
        # position in mm, orientation roll=180 pitch=0 yaw=0
        input_is_radian=False
    )
 
    if code != 0:
        print(f"IK failed at point {i}")
        continue
 
    # 3. Use the BUILT-IN FK solver to verify
    code, pose = arm.get_forward_kinematics(joints, input_is_radian=True)
    
    if code == 0:
        # pose is [x, y, z, roll, pitch, yaw] — x,y,z in mm
        recovered_positions.append(np.array(pose[:3]) / 1000.0)  # back to meters
        recovered_orientations.append(np.array(pose[3:]))
 
recovered_positions = np.array(recovered_positions)
 
# 4. Compare target vs. recovered
errors = np.linalg.norm(path_xyz - recovered_positions, axis=1) * 1000  # mm
print(f"\n--- BUILT-IN IK/FK ACCURACY ---")
print(f"Max position error:  {np.max(errors):.4f} mm")
print(f"Mean position error: {np.mean(errors):.4f} mm")
 
# 5. Plot the error vs time
plt.figure()
plt.plot(tq[:len(errors)], errors, 'b-')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (mm)')
plt.title('Accuracy of Built-in IK')
plt.grid(True)
plt.show()
 
arm.disconnect()
'''

