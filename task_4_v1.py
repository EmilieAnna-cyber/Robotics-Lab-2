"""
TASK 4 - Comparison of the points in a trajectory using inverse and direct kinematics of a
xArm (Lite 6)

The trajectory is the main swing movement of task 2 (main_swing.py)

For every point in the trajectory:
1-. Calculate the orientation of the arm based on the club position
2-. Calculate the joint variables using the IK of the xArm (built-in, on the robot)
3-. Use the joint variables in the direct kinematics:
      - OUR direct kinematics (task 1, kinematics_2.py)   <- asked by the assignment
      - the direct kinematics of the xArm (built-in)       <- extra check
    and compare with the original position and orientation

MODE = "robot" -> at the lab, connected to the robot (the arm does NOT move)
MODE = "test"  -> without the robot: our own IK replaces the xArm IK, only to check the code runs
MODE = "load"  -> without the robot: reuse the joint angles saved at the lab (task4_joints.npy)

Run from the folder Robotics-Lab-2:   python -m useful.task_4
or from inside useful/:               python task_4.py
"""

import os
import sys
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")  # main_swing calls plt.show() when imported, this stops it from blocking
import matplotlib.pyplot as plt

# so "import useful.xxx" works from inside useful/ too
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import useful.robot_math as rm
import useful.main_swing as ms   # the final trajectory of task 2
from useful.kinematics_2 import T_0_6, theta1, theta2, theta3, theta4, theta5, theta6



# =============================================================================
# Main configuration
# =============================================================================

MODE = "test"                 # "robot", "test" or "load"
XARM_IP = "192.168.1.158"
BALL_XYZ = ms.ball_xyz        # same ball as task 2
HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS_FILE = os.path.join(HERE, "task4_joints.npy")
FIGURE_FILE = os.path.join(HERE, "task4_errors.png")

# our direct kinematics (task 1): full matrix T_0_6 as a numpy function
our_forward_kinematics = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), T_0_6, "numpy")


# =============================================================================
# Orientation
# Convert the rotation matrix R= Rz(α) @ Ry(β) @ Rx(γ) to angles: roll, pitch and yaw (rpy) for the
# xArm (same convention as the Lite 6 manual: rotations about the fixed X, Y, Z axes of the base)

def R_matrix_to_rpy(R):

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=float)

# Convert the angles return by the forward kinematics of the xArm to a rotation matrix to compare with the original rotation matrix
def rpy_to_R_matrix(roll, pitch, yaw):

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp_ = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    Ry = np.array([
        [cp, 0.0, sp_],
        [0.0, 1.0, 0.0],
        [-sp_, 0.0, cp],
    ])
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return Rz @ Ry @ Rx

# Angle (in degrees) between the wanted orientation and the obtained one (0 = same orientation)
def orientation_error_degrees(R_target, R_actual):

    R_delta = R_target.T @ R_actual
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# =============================================================================
# Trajectory of task 2
# Take the 100 points directly from main_swing.py (the final trajectory: Hermite spline + velocity clamp)

def generate_trajectory():

    waypoints = ms.tips_clamped
    t_corners = ms.t_corners
    tq = ms.tq
    path_xyz = ms.path_xyz

    return waypoints, t_corners, tq, path_xyz


# =============================================================================
# Objective pose to the xArm
# Convert a objective point of the club to a pose of the flange of the xArm

def xarm_pose(tip_target):

    T_look_at = rm.get_matrix_look_at(*tip_target)

    # Using the code in task 2 transformation from the club to the flange
    flange_position_m = rm.get_gripper_pos(tip_target)

    rpy_rad = R_matrix_to_rpy(T_look_at[:3, :3])

    return np.array([
        flange_position_m[0] * 1000.0,
        flange_position_m[1] * 1000.0,
        flange_position_m[2] * 1000.0,
        rpy_rad[0],
        rpy_rad[1],
        rpy_rad[2],
    ], dtype=float)


# =============================================================================
# IK and FK of the xArm
# Calculate IK and FK for every point in the trajectory using the SDK

def ik_fk_for_point(arm, path_xyz):

    target_poses = []
    joint_solutions = []
    fk_poses = []
    successful_indices = []
    failed_indices = []

    for point_index, tip_target in enumerate(path_xyz):
        target_pose = xarm_pose(tip_target)

        try:
            code_ik, joints = arm.get_inverse_kinematics(
                target_pose.tolist(),
                input_is_radian=True,
                return_is_radian=True,
            )
        except Exception as exc:
            print(f"[IK EXCEPTION] Point {point_index}: {exc}")
            failed_indices.append(point_index)
            continue

        if code_ik != 0 or joints is None or len(joints) < 6:
            print(f"[IK ERROR] Point {point_index}: SDK code={code_ik}")
            failed_indices.append(point_index)
            continue

        joints = np.asarray(joints[:6], dtype=float)   # the SDK returns 7 values, the Lite 6 has 6 joints

        try:
            code_fk, pose_fk = arm.get_forward_kinematics(
                joints.tolist(),
                input_is_radian=True,
                return_is_radian=True,
            )
        except Exception as exc:
            print(f"[FK EXCEPTION] Point {point_index}: {exc}")
            failed_indices.append(point_index)
            continue

        if code_fk != 0 or pose_fk is None or len(pose_fk) < 6:
            print(f"[FK ERROR] Point {point_index}: SDK code={code_fk}")
            failed_indices.append(point_index)
            continue

        target_poses.append(target_pose)
        joint_solutions.append(joints)
        fk_poses.append(np.asarray(pose_fk[:6], dtype=float))
        successful_indices.append(point_index)

    return (
        np.asarray(target_poses, dtype=float),
        np.asarray(joint_solutions, dtype=float),
        np.asarray(fk_poses, dtype=float),
        np.asarray(successful_indices, dtype=int),
        failed_indices,
    )


# =============================================================================
# Our direct kinematics (task 1)
# Put the joint variables of the xArm IK in OUR forward kinematics and return the pose like the xArm
# does: [x(mm), y(mm), z(mm), roll, pitch, yaw]

def our_fk_poses(joint_solutions):
    '''
    goal: compute the flange pose from the joint angles using our own forward kinematics
    PARAMS:
        joint_solutions: array of joint angles (radians) for each point in the trajectory

    RETURNS:
        poses: array of flange poses (mm, radians) for each point in the trajectory
        pose = [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)]
    '''

    poses = []
    for joints in joint_solutions:
        T = np.array(our_forward_kinematics(*joints), dtype=float)
        rpy = R_matrix_to_rpy(T[:3, :3])
        poses.append([T[0, 3] * 1000.0, T[1, 3] * 1000.0, T[2, 3] * 1000.0, rpy[0], rpy[1], rpy[2]])
    return np.asarray(poses, dtype=float)


# =============================================================================
# Calculation of errors
# Calculate the error between the position and orientation of a point in the original trajectory and
# the position and orientation obtained after using IK and FK

def pose_flange_errors(target_poses, fk_poses):
    '''
    goal: compute the position and orientation errors between the target poses and the obtained poses

    PARAMS:
        target_poses: array of target poses (mm, radians) for each point in the trajectory
        fk_poses: array of obtained poses (mm, radians) for each point in the trajectory

    RETURNS:
        position_errors_mm: array of position errors (mm) for each point in the trajectory
        orientation_errors_deg: array of orientation errors (degrees) for each point in the trajectory
    '''

    position_errors_mm = np.linalg.norm(
        target_poses[:, :3] - fk_poses[:, :3],
        axis=1,
    )

    orientation_errors_deg = []
    for target_pose, actual_pose in zip(target_poses, fk_poses):
        R_target = rpy_to_R_matrix(*target_pose[3:6])
        R_actual = rpy_to_R_matrix(*actual_pose[3:6])
        orientation_errors_deg.append(
            orientation_error_degrees(R_target, R_actual)
        )

    return position_errors_mm, np.asarray(orientation_errors_deg, dtype=float)


def print_errors(name, position_errors_mm, orientation_errors_deg):
    '''
    goal: print the position and orientation errors for a given set of poses

    PARAMS:
        name: name of the set of poses (ex: "xArm FK", "Our FK")
        position_errors_mm: array of position errors (mm) for each point in the trajectory
        orientation_errors_deg: array of orientation errors (degrees) for each point in the trajectory
    '''

    print(f"\n--- {name} ---")
    print(f"Position error    max: {np.max(position_errors_mm):.6f} mm,  "
          f"mean: {np.mean(position_errors_mm):.6f} mm,  RMS: {np.sqrt(np.mean(position_errors_mm ** 2)):.6f} mm")
    print(f"Orientation error max: {np.max(orientation_errors_deg):.6f} deg, "
          f"mean: {np.mean(orientation_errors_deg):.6f} deg, RMS: {np.sqrt(np.mean(orientation_errors_deg ** 2)):.6f} deg")


# =============================================================================
# Graphics
# Plot the error at every point of the trajectory

def plot_results(successful_indices, n_points, pos_err_ours, rot_err_ours, pos_err_xarm, rot_err_xarm):
    '''
    goal: plot the position and orientation errors for the xArm IK + our FK and xArm IK + xArm FK
    
    PARAMS:
        successful_indices: array of indices of the points in the trajectory that were successfully processed
        n_points: total number of points in the trajectory
        pos_err_ours: array of position errors (mm) for each point in the trajectory
        rot_err_ours: array of orientation errors (degrees) for each point in the trajectory
        pos_err_xarm: array of position errors (mm) for each point in the trajectory
        rot_err_xarm: array of orientation errors (degrees) for each point in the trajectory
    ''' 

    fig = plt.figure(figsize=(14, 5))

    # Graphic 1: Position error
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(successful_indices, pos_err_ours, "b.-", label="xArm IK + our FK (task 1)")
    if pos_err_xarm is not None:
        ax1.plot(successful_indices, pos_err_xarm, "g.--", label="xArm IK + xArm FK")
    ax1.set_xlim(0, n_points - 1)
    ax1.set_xlabel("Number of the point in the trajectory")
    ax1.set_ylabel("Position error [mm]")
    ax1.set_title("Position error: original objective vs FK obtained")
    ax1.grid(True)
    ax1.legend()

    # Graphic 2: Orientation error
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(successful_indices, rot_err_ours, "m.-", label="xArm IK + our FK (task 1)")
    if rot_err_xarm is not None:
        ax2.plot(successful_indices, rot_err_xarm, "c.--", label="xArm IK + xArm FK")
    ax2.set_xlim(0, n_points - 1)
    ax2.set_xlabel("Number of the point in the trajectory")
    ax2.set_ylabel("Orientation error [degrees]")
    ax2.set_title("Orientation error: original objective vs FK obtained")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=180)
    print(f"\nGraphics saved in {FIGURE_FILE}")


def main():
    '''
    Main function to run the trajectory generation, IK/FK calculations, error analysis, and plotting.
    Depending on the MODE, it will either connect to the xArm robot, load saved joint angles, or run in test mode with our own IK.
    1. Generate the trajectory points
    2. Depending on the MODE, perform IK and FK calculations
    3. Calculate position and orientation errors
    4. Print error statistics and plot the results  
    '''

    waypoints, t_corners, tq, path_xyz = generate_trajectory()
    n_points = len(path_xyz)
    print(f"\nGenerated cartesian points: {n_points}")
    print("Waypoints [m]:")
    for i, waypoint in enumerate(waypoints):
        print(f"  P{i}: {waypoint.round(5)}")

    fk_poses = None   # the FK of the xArm only exists with the robot connected

    if MODE == "robot":
        from xarm.wrapper import XArmAPI
        arm = None
        try:
            print(f"Connecting with xArm in {XARM_IP}...")
            arm = XArmAPI(XARM_IP, is_radian=True)

            if not arm.connected:
                raise ConnectionError(
                    f"Connection failed with xArm in {XARM_IP}. "
                    "Check the robot is turned on and the network."
                )

            print("Connection established")

            (
                target_poses,
                joint_trajectory,
                fk_poses,
                successful_indices,
                failed_indices,
            ) = ik_fk_for_point(arm, path_xyz)

        finally:
            if arm is not None:
                arm.disconnect()
                print("Connection with xArm closed")

        # save what the robot gave, so the analysis can be redone at home (MODE = "load")
        np.save(JOINTS_FILE, {"target_poses": target_poses, "joints": joint_trajectory, "fk_poses": fk_poses,
                              "successful_indices": successful_indices, "failed_indices": failed_indices})
        print(f"Robot results saved in {JOINTS_FILE}")

    elif MODE == "load":
        saved = np.load(JOINTS_FILE, allow_pickle=True).item()
        target_poses = saved["target_poses"]
        joint_trajectory = saved["joints"]
        fk_poses = saved["fk_poses"]
        successful_indices = saved["successful_indices"]
        failed_indices = saved["failed_indices"]
        print(f"Robot results loaded from {JOINTS_FILE}")

    else:   # MODE == "test"
        print("\nTEST MODE: our own IK replaces the xArm IK (NOT the result of task 4, only checks the code)")
        target_poses = np.array([xarm_pose(p) for p in path_xyz])
        joint_trajectory = np.array(ms.joint_trajectory)
        successful_indices = np.arange(n_points)
        failed_indices = []

    if len(successful_indices) == 0:
        raise RuntimeError("IK and FK failed for every point")

    print(f"\nNumber of points: {n_points}")
    print(f"Points with IK and FK correct: {len(successful_indices)}")
    print(f"Failed points: {len(failed_indices)}")
    if failed_indices:
        print("Failed indices:", failed_indices)

    # xArm IK + OUR direct kinematics (task 1)  <- what the assignment asks
    our_poses = our_fk_poses(joint_trajectory)
    pos_err_ours, rot_err_ours = pose_flange_errors(target_poses, our_poses)
    print_errors("xArm IK + our direct kinematics (task 1)", pos_err_ours, rot_err_ours)

    # xArm IK + xArm direct kinematics (extra check, only with the robot)
    pos_err_xarm, rot_err_xarm = None, None
    if fk_poses is not None:
        pos_err_xarm, rot_err_xarm = pose_flange_errors(target_poses, fk_poses)
        print_errors("xArm IK + xArm direct kinematics", pos_err_xarm, rot_err_xarm)

    plot_results(successful_indices, n_points, pos_err_ours, rot_err_ours, pos_err_xarm, rot_err_xarm)


if __name__ == "__main__":
    main()