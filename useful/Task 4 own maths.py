"""
TASK 4 - Offline verification of the Main Swing trajectory.

Workflow:
1. Generate the same four waypoints used by main_swing.
2. Interpolate 100 Cartesian club-tip points with CubicHermiteSpline.
3. For every point, calculate the target flange pose (position + orientation).
4. Solve inverse kinematics with robot_math.solve_ik_with_bounds.
5. Apply forward kinematics to the joint angles to recover flange position
   and orientation.
6. Compare target and recovered poses and plot both errors against point index.

This version is fully offline: it does not connect to an xArm or require its SDK.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import robot_math as rm


# =============================================================================
# 1. DEFINE THE BALL AND BUILD THE SAME WAYPOINTS AS MAIN_SWING
# =============================================================================

ball_xyz = np.array([0.25, 0.0, 0.15])

tips_clamped = rm.find_all_waypoints(ball_xyz)

print("Clamped waypoints:")
for i, point in enumerate(tips_clamped):
    print(f"  P{i}: {point.round(4)}")


# =============================================================================
# 2. BUILD THE SAME MAIN SWING TRAJECTORY
# =============================================================================

t_corners = rm.find_time_steps_from_waypoints(tips_clamped)
t_corners = rm.clamp_velocity(tips_clamped, t_corners)

# Use the original robot_math function if it exists.
# The supplied robot_math.py does not define it, so a standard finite-
# difference fallback is provided to keep the Main Swing spline executable.
if hasattr(rm, "find_waypoint_velocities"):
    vel = rm.find_waypoint_velocities(tips_clamped, t_corners)
else:
    vel = np.zeros_like(tips_clamped, dtype=float)
    for i in range(1, len(tips_clamped) - 1):
        vel[i] = (
            (tips_clamped[i + 1] - tips_clamped[i - 1])
            / (t_corners[i + 1] - t_corners[i - 1])
        )

spline_x = CubicHermiteSpline(t_corners, tips_clamped[:, 0], vel[:, 0])
spline_y = CubicHermiteSpline(t_corners, tips_clamped[:, 1], vel[:, 1])
spline_z = CubicHermiteSpline(t_corners, tips_clamped[:, 2], vel[:, 2])

N_POINTS = 100
tq = np.linspace(0.0, t_corners[-1], N_POINTS)

# Original club-tip positions along the trajectory.
path_xyz = np.column_stack([
    spline_x(tq),
    spline_y(tq),
    spline_z(tq)
])


# =============================================================================
# 3. PREPARE THE FORWARD-KINEMATICS ORIENTATION
# =============================================================================

# robot_math.get_ee_position already calculates FK position from T_0_6.
# The supplied robot_math.py does not expose an FK orientation function.
# Therefore, use the rotation part of that same symbolic T_0_6 model.
rotation_symbolic = rm.T_0_6[:3, :3].subs(rm.LINK_VALUES)

rotation_functions = [
    [
        __import__("sympy").lambdify(
            (rm.theta1, rm.theta2, rm.theta3,
             rm.theta4, rm.theta5, rm.theta6),
            rotation_symbolic[row, col],
            "numpy"
        )
        for col in range(3)
    ]
    for row in range(3)
]


def get_ee_orientation(joint_angles):
    """Return the 3x3 FK rotation matrix for six joint angles (radians)."""
    return np.array([
        [rotation_functions[row][col](*joint_angles) for col in range(3)]
        for row in range(3)
    ], dtype=float)


def rotation_error_degrees(target_rotation, recovered_rotation):
    """
    Angular distance between two rotation matrices.
    Uses the relative rotation angle: acos((trace(R_target.T R_fk)-1)/2).
    """
    relative_rotation = target_rotation.T @ recovered_rotation
    cosine_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


# =============================================================================
# 4. INVERSE KINEMATICS -> FORWARD KINEMATICS FOR EACH OF THE 100 POINTS
# =============================================================================

point_indices = []
target_flange_positions = []
recovered_flange_positions = []
position_errors_mm = []
orientation_errors_deg = []
joint_trajectory = []
failed_points = []

for i, club_tip_target in enumerate(path_xyz):
    try:
        # Convert the club-tip point into the flange position using the
        # existing Main Swing / robot_math transformation.
        flange_target = rm.get_gripper_pos(club_tip_target)

        # Target orientation: same look_at orientation used internally by
        # robot_math.solve_ik_with_bounds.
        target_rotation = rm.get_matrix_look_at(*flange_target)[:3, :3]

        # Inverse kinematics: Cartesian flange target -> six joint angles.
        q_sol = np.asarray(
            rm.solve_ik_with_bounds(flange_target),
            dtype=float
        )

        # Forward kinematics: six joint angles -> recovered flange pose.
        flange_recovered = np.asarray(rm.get_ee_position(q_sol), dtype=float)
        rotation_recovered = get_ee_orientation(q_sol)

        # Position error in millimetres (robot_math uses metres).
        position_error = (
            np.linalg.norm(flange_target - flange_recovered) * 1000.0
        )

        # Orientation error in degrees.
        orientation_error = rotation_error_degrees(
            target_rotation,
            rotation_recovered
        )

        point_indices.append(i + 1)
        target_flange_positions.append(flange_target)
        recovered_flange_positions.append(flange_recovered)
        position_errors_mm.append(position_error)
        orientation_errors_deg.append(orientation_error)
        joint_trajectory.append(q_sol)

    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        print(f"IK/FK failed at point {i + 1}: {exc}")
        failed_points.append(i + 1)


# Convert results to arrays for analysis and plotting.
point_indices = np.asarray(point_indices, dtype=int)
target_flange_positions = np.asarray(target_flange_positions, dtype=float)
recovered_flange_positions = np.asarray(recovered_flange_positions, dtype=float)
position_errors_mm = np.asarray(position_errors_mm, dtype=float)
orientation_errors_deg = np.asarray(orientation_errors_deg, dtype=float)
joint_trajectory = np.asarray(joint_trajectory, dtype=float)


# =============================================================================
# 5. PRINT NUMERICAL ACCURACY RESULTS
# =============================================================================

if len(point_indices) == 0:
    print("\nNo trajectory points could be verified.")
    print("Check waypoint reachability and joint limits.")
else:
    print("\n--- OFFLINE IK/FK TRAJECTORY VERIFICATION ---")
    print(f"Interpolated trajectory points: {N_POINTS}")
    print(f"Successfully verified points:   {len(point_indices)}")
    print(f"Failed points:                  {len(failed_points)}")

    if failed_points:
        print("Failed point indices:", failed_points)

    print("\nPosition error [mm]")
    print(f"  Maximum: {np.max(position_errors_mm):.6f}")
    print(f"  Mean:    {np.mean(position_errors_mm):.6f}")
    print(f"  RMS:     {np.sqrt(np.mean(position_errors_mm ** 2)):.6f}")

    print("\nOrientation error [degrees]")
    print(f"  Maximum: {np.max(orientation_errors_deg):.6f}")
    print(f"  Mean:    {np.mean(orientation_errors_deg):.6f}")
    print(f"  RMS:     {np.sqrt(np.mean(orientation_errors_deg ** 2)):.6f}")


    # =============================================================================
    # 6. PLOT POSITION AND ORIENTATION ERROR AGAINST POINT NUMBER
    # =============================================================================

    fig, ax_position = plt.subplots(figsize=(11, 5))

    line_position, = ax_position.plot(
        point_indices,
        position_errors_mm,
        "b-",
        linewidth=1.6,
        label="Position error"
    )
    ax_position.set_xlabel("Trajectory point number")
    ax_position.set_ylabel("Position error [mm]", color="b")
    ax_position.tick_params(axis="y", labelcolor="b")
    ax_position.set_xlim(1, N_POINTS)
    ax_position.set_xticks(np.arange(1, N_POINTS + 1, 10))
    ax_position.grid(True, alpha=0.3)

    # A second y-axis is used because position and orientation have
    # different physical units (mm and degrees).
    ax_orientation = ax_position.twinx()
    line_orientation, = ax_orientation.plot(
        point_indices,
        orientation_errors_deg,
        "r-",
        linewidth=1.6,
        label="Orientation error"
    )
    ax_orientation.set_ylabel("Orientation error [degrees]", color="r")
    ax_orientation.tick_params(axis="y", labelcolor="r")

    lines = [line_position, line_orientation]
    ax_position.legend(
        lines,
        [line.get_label() for line in lines],
        loc="best"
    )

    plt.title("Position and orientation errors after offline IK/FK")
    fig.tight_layout()
    plt.show()


# =============================================================================
# 7. OPTIONAL: SHOW THE ORIGINAL AND FK-RECOVERED FLANGE PATHS
# =============================================================================

if len(point_indices) > 0:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        target_flange_positions[:, 0],
        target_flange_positions[:, 1],
        target_flange_positions[:, 2],
        "b-",
        label="Original flange path"
    )
    ax.plot(
        recovered_flange_positions[:, 0],
        recovered_flange_positions[:, 1],
        recovered_flange_positions[:, 2],
        "r--",
        label="Recovered flange path (FK)"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Original vs. FK-recovered flange trajectory")
    ax.legend()
    plt.tight_layout()
    plt.show()
