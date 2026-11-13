"""
TASK 4 - Comparision of the points in a trajectory using inverse and direct kinematics of a
xArm 

The trajectory is based of the main swing movement in task 2


For every point in the trajectory:
1-. Calculate the orientation of the arm based on the club position
2_. Calculate the joint variables using IK xARm
3-. Using the joint variables and forward kinematics
4-. Compare the position and orientation obtained with the original ones

The robot must be on and connected (same network, XARM_IP). The arm does NOT move:
only the IK and FK of the controller are used.
BALL_XYZ must be the same ball as in main_swing.py.

Run from the folder Robotics-Lab-2:   python -m useful.task_4_
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline  

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import useful.robot_math as rm  



# =============================================================================
# Main configuration
# =============================================================================

XARM_IP = "192.168.1.158"  
BALL_XYZ = np.array([0.25, 0.0, 0.15], dtype=float)  
N_POINTS = 100


# =============================================================================
# Orientation
# Convert the rotation matrix R= Rz(α) @ Ry(β) @ Rx(γ) to angles: roll, pitch and yaw (rpy) for the forward kinematic
# of the xArm

def R_matrix_to_rpy(R):
    '''
    goal: convert a rotation matrix to the roll, pitch, yaw angles used by the xArm
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    PARAMS:
        R: 3x3 rotation matrix
    RETURNS:
        array [roll, pitch, yaw] (radians)
    '''
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
    '''
    goal: convert the roll, pitch, yaw angles of the xArm back to a rotation matrix
    PARAMS:
        roll, pitch, yaw: angles (radians)
    RETURNS:
        R: 3x3 rotation matrix Rz(yaw) @ Ry(pitch) @ Rx(roll)
    '''
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return Rz @ Ry @ Rx

# Angle between the wanted orientation and the obtained one
def orientation_error_degrees(R_target, R_actual):
    '''
    goal: angle of the rotation between the wanted orientation and the obtained one (0 = same orientation)
    PARAMS:
        R_target: 3x3 wanted rotation matrix
        R_actual: 3x3 obtained rotation matrix
    RETURNS:
        angle (degrees) = arccos((trace(R_target^T R_actual) - 1) / 2)
    '''
    
    R_delta = R_target.T @ R_actual
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# =============================================================================
# Generation of the 100 points in trajectory using the task 2
# Reproduces the calculations the waypoints and the interpolation used in task 2

def generate_trajectory():
    '''
   goal: Reproduces the calculations the waypoints and the interpolation used in task 2
    PARAMS:
         None
    RETURNS:
        waypoints: list of waypoints in the trajectory
    '''

    waypoints = rm.find_all_waypoints(BALL_XYZ)

    t_corners = rm.find_time_steps_from_waypoints(waypoints)
    t_corners = rm.clamp_velocity(waypoints, t_corners)   # FIX: same timing as main_swing

    vel = rm.find_waypoint_velocities(waypoints, t_corners)
    spline_x = CubicHermiteSpline(t_corners, waypoints[:, 0], vel[:, 0])
    spline_y = CubicHermiteSpline(t_corners, waypoints[:, 1], vel[:, 1])
    spline_z = CubicHermiteSpline(t_corners, waypoints[:, 2], vel[:, 2])

    tq = np.linspace(0.0, t_corners[-1], N_POINTS)
    path_xyz = np.column_stack((
        spline_x(tq),
        spline_y(tq),
        spline_z(tq),
    ))

    return waypoints, t_corners, tq, path_xyz


# =============================================================================
# Objective pose to the xArm
# Convert a objetive poiint of the club to a pose of the flange of the xArm

def xarm_pose(tip_target):
    '''
    goal: Convert a objetive poiint of the club to a pose of the flange of the xArm

    PARAMS:
        tip_target: point in the trajectory of the club tip 

    RETURNS:
        flange_pose: pose of the flange of the xArm corresponding to the tip_target
    '''
    
    T_look_at = rm.get_matrix_look_at(*tip_target)

    # Using the code in task 2 trasnformation from the club to the flange
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
#Calculate IK and Fk for every point in the trayectory using the SDK

def ik_fk_for_point(arm, path_xyz):
    '''
    goal: Calculate IK and Fk for every point in the trayectory using the SDK

    PARAMS:
        arm: xArmAPI object (ex: arm = XArmAPI(XARM_IP, is_radian=True))
        path_xyz: list of points in the trajectory

    RETURNS:
        target_poses: list of target poses for every point in the trajectory
        joint_solutions: list of joint variables for every point in the trajectory
        fk_poses: list of poses obtained after using IK and FK of the xArm for every point in the trajectory
        successful_indices: list of indices of the points that were successfully calculated
        failed_indices: list of indices of the points that were not successfully calculated
    
    '''
    
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
            print(f"[IK ERROR] Point {point_index}: código SDK={code_ik}")
            failed_indices.append(point_index)
            continue

        joints = np.asarray(joints[:6], dtype=float)

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
            print(f"[FK ERROR] Point {point_index}: código SDK={code_fk}")
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
# Calculation of errors
# Calculate the error between the position and orientation of a point in the original trajectory to the
# obtain positiion and orientation after using IK and FK of the xArm

def pose_flange_errors(target_poses, fk_poses):
    '''
    goal: Calculate the error between the position and orientation of a point in the original trajectory to the
    obtain positiion and orientation after using IK and FK of the xArm
    
    PARAMS:
        target_poses: list of target poses for every point in the trajectory
        fk_poses: list of poses obtained after using IK and FK of the xArm for every point in the trajectory
    
    RETURNS:
        position_errors_mm: list of position errors for every point in the trajectory
        orientation_errors_deg: list of orientation errors for every point in the trajectory

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


# =============================================================================
# Graphics
# Plot the number of points with the error calculated

def plot_results(waypoints, path_xyz, joint_trajectory, successful_indices,
                 position_errors_mm, orientation_errors_deg):
    '''
    goal: Plot the number of points with the error calculated
    
    PARAMS:
        waypoints: list of waypoints in the trajectory
        path_xyz: list of points in the trajectory
        joint_trajectory: list of joint variables for every point in the trajectory
        successful_indices: list of indices of the points that were successfully calculated
        position_errors_mm: list of position errors for every point in the trajectory
        orientation_errors_deg: list of orientation errors for every point in the trajectory    

    RETURNS:
        None
    '''

    fig = plt.figure(figsize=(14, 5))

    # Graphic 1: Position error
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(successful_indices, position_errors_mm, "b.-")
    ax1.set_xlim(0, N_POINTS - 1)
    ax1.set_xlabel("Number of points")
    ax1.set_ylabel("Position error [mm]")
    ax1.set_title("Position error original objective vs FK obtained")
    ax1.grid(True)

    # Graphic 2 Orientation error
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(successful_indices, orientation_errors_deg, "m.-")
    ax2.set_xlim(0, N_POINTS - 1)
    ax2.set_xlabel("Number of points")
    ax2.set_ylabel("Orientation error [degrees]")
    ax2.set_title("Orientation error original objective vs FK obtained")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


def main():
    '''
    goal: connect to the xArm, compute IK and FK of the robot for the 100 points of the swing,
    print the position and orientation errors and plot them
    '''
    from xarm.wrapper import XArmAPI   
    arm = None
    successful_indices = []            

    try:
        print(f"Conecting with xArm in {XARM_IP}...")
        arm = XArmAPI(XARM_IP, is_radian=True)

        if not arm.connected:
            raise ConnectionError(
                f"Connection failed with xArm en {XARM_IP}. "
                "Check the robot turned on and the network."
            )

        print("Connection stablished")

        waypoints, t_corners, tq, path_xyz = generate_trajectory()
        print(f"\nGenerated cartesian points: {len(path_xyz)}")
        print("Waypoints [m]:")
        for i, waypoint in enumerate(waypoints):
            print(f"  P{i}: {waypoint.round(5)}")

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

    if len(successful_indices) == 0:
        raise RuntimeError("Incomplete IK and FK for any point")

    position_errors_mm, orientation_errors_deg = pose_flange_errors(
        target_poses, fk_poses
    )

    print("Final graphics")
    print(f"Number of points: {N_POINTS}")
    print(f"Points of IK and FK correct: {len(successful_indices)}")
    print(f"Failed points: {len(failed_indices)}")
    if failed_indices:
        print("Failed indices:", failed_indices)

    print("\n--- Psition error of the flange ---")
    print(f"Max: {np.max(position_errors_mm):.6f} mm")
    print(f"Med:  {np.mean(position_errors_mm):.6f} mm")
    print(f"RMS:    {np.sqrt(np.mean(position_errors_mm ** 2)):.6f} mm")

    print("\n--- orientation error of the flange ---")
    print(f"Max: {np.max(orientation_errors_deg):.6f} degrees")
    print(f"Med:  {np.mean(orientation_errors_deg):.6f} degrees")
    print(f"RMS: {np.sqrt(np.mean(orientation_errors_deg ** 2)):.6f} degrees"
    )

    plot_results(
        waypoints,
        path_xyz,
        joint_trajectory,
        successful_indices,
        position_errors_mm,
        orientation_errors_deg,
    )


if __name__ == "__main__":
    main()