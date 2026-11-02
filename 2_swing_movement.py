import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import sympy as sp
import kinematics_2 as k
from kinematics_2 import (
    D1, A2, A3, D4, D6,
    T_0_6,
    X_tip, Y_tip, Z_tip, L_CLUB,
    theta1, theta2, theta3, theta4, theta5, theta6,
)




# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# give numerical values to the robot: we need to measure so maybe change later 

JOINT_BOUNDS = [
    (-np.pi,   np.pi),    # theta1 limits
    (-5*np.pi/6, 5*np.pi/6),  # theta2 limits (+/- 150 degrees)
    (-3.5*np.pi/180, 300*np.pi/180),  # theta3 limits
    (-2*np.pi,   2*np.pi),    # theta4 limits
    (-124*np.pi/180, 124*np.pi/180),  # theta5 limits
    (-2*np.pi, 2*np.pi)   # theta6 limits
]

# make velocity in rad/s: 
# 180 degrees/s => 180 * pi /180 rad/s = pi rad/s
MAX_JOINT_VELOCITY = np.pi # rad/s

# Maximum distance from the base to the wrist center (meters)
# Calculation: Length of link 2 + length of link 3
MAX_REACH = 0.2 + np.sqrt(0.2276**2 + 0.087**2) 

# CLUB LENGTH: CHANGE LATER:

L_CLUB_VAL = 0.1 #m

LINK_VALUES = {
    L_CLUB: L_CLUB_VAL,
}

# =============================================================================
# FORWARD KINEMATICS — SYMBOLIC → NUMERIC
# =============================================================================

X_raw = T_0_6[0, 3]
Y_raw = T_0_6[1, 3]
Z_raw = T_0_6[2, 3]

# replace theoretical values with the real ones with .subs

X_raw = X_raw.subs(LINK_VALUES)
Y_raw = Y_raw.subs(LINK_VALUES)
Z_raw = Z_raw.subs(LINK_VALUES)



def joint_coords_from_fixed_frame(Matrix_list, joint_i):
    ''' get the coordinates of a joint in the fixed frame (0) given the transformation matrices and the joint index'''
    T_0_i = k.T_i_j(Matrix_list,0,joint_i)
    # Extract raw position expressions
    X_raw = T_0_i[0, 3]
    Y_raw = T_0_i[1, 3]
    Z_raw = T_0_i[2, 3]
    joint_pos = [X_raw,Y_raw,Z_raw]

    return joint_pos

gripper_coord = joint_coords_from_fixed_frame(k.T_list, 6)

# we are using both Sympy & NumPy: lambdify() allows to convert SymPy expressions into NumPy expression

forward_x = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), X_raw, 'numpy')
forward_y = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), Y_raw, 'numpy')
forward_z = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), Z_raw, 'numpy')

def get_ee_position(q):
    """Returns [X, Y, Z] given a joint configuration array 
    PARAMETERS:
        q: list of length 6 containing the angles of the robot (theta angles)
    RETURNS: 
        position the robot in (x,y,z) coordinates, in a Numpy Vector"""
    return np.array([forward_x(*q), forward_y(*q), forward_z(*q)])  

# =============================================================================
# ORIENTATION — LOOK-AT MATRIX
# =============================================================================


def get_matrix_look_at(x, y, z, target_point=np.array([0, 0, 0])):
    """
    From a position (x,y,z) and a target point, returns a 4x4 transformation matrix
    that represents the orientation of the end-effector looking at the target point.
    target_point is a 3D point in space that the end-effector should "look at".

    PARAMETERS:
    x: float, y: float, z: float: coordinates of the end-effector position
    target_point: np.array of shape (3,), the point to look at

    RETURNS:
        T_target: np.array of shape (4,4), the transformation matrix
    """
    position = np.array([x, y, z])
    # Vecteur directionnel entre l'outil et sa cible
    z_axis = target_point - position
    z_axis = z_axis / np.linalg.norm(z_axis) # Normalisation
    
    # Calcul des axes orthogonaux (Up vector temporaire sur l'axe Z global)
    up = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.99 else np.array([0, 1, 0])
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Construction de la matrice
    T_target = np.identity(4)
    T_target[:3, 0] = x_axis
    T_target[:3, 1] = y_axis
    T_target[:3, 2] = z_axis
    T_target[:3, 3] = position
    return T_target


# =============================================================================
# INVERSE KINEMATICS
# =============================================================================

def solve_ik_with_bounds(target_pos, bounds=JOINT_BOUNDS):
    """
    Computes Inverse Kinematics and validates results against specific
    hardware JOINT_BOUNDS limits.
    """
    # 1. Structural Geometric Constants
    A2, A3, D4, D1, D6 = 0.2, 0.2276, 0.087, 0.2433, 0.0615

    x, y, z = target_pos[0], target_pos[1], target_pos[2]
    
    # Generate target matrix from your custom look-at orientation logic
    matrix = get_matrix_look_at(x, y, z)
    r13, r23, r33 = matrix[0, 2], matrix[1, 2], matrix[2, 2]

    # 2. Compute Wrist Center
    x_w = x - D6 * r13
    y_w = y - D6 * r23
    z_w = z - D6 * r33

    # --- STEP 1: Compute Theta 1 ---
    theta1 = np.arctan2(y_w, x_w)

    # --- STEP 2: Compute Theta 3 ---
    kx = x_w * np.cos(theta1) + y_w * np.sin(theta1)
    kz = z_w - D1

    num = kx**2 + kz**2 - A2**2 - (A3**2 + D4**2)
    den = 2 * A2 * np.sqrt(A3**2 + D4**2)
    cos_theta3 = num / den
    
    # Reachable Workspace Check
    if cos_theta3 < -1.0 or cos_theta3 > 1.0:
        raise ValueError(f"Target position {target_pos} is OUT OF REACH. Arm cannot stretch/collapse that far.")
        return None
    
    sin_theta3 = np.sqrt(1 - cos_theta3**2)
    theta3 = np.arctan2(sin_theta3, cos_theta3) - np.arctan2(D4, A3)

    # --- STEP 3: Compute Theta 2 ---
    num_t2 = A3 * np.sin(theta3) + D4 * np.cos(theta3)
    den_t2 = A2 + A3 * np.cos(theta3) - D4 * np.sin(theta3)
    theta2 = np.arctan2(kz, kx) - np.arctan2(num_t2, den_t2)

    # --- STEP 4: Compute Wrist Orientation ---
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c23, s23 = np.cos(theta2 - theta3), np.sin(theta2 - theta3)
    
    R3_0 = np.array([
        [c1*c23, -c1*s23,  s1],
        [s1*c23, -s1*s23, -c1],
        [s23,     c23,     0 ]
    ])

    R_target = matrix[:3, :3]
    R_wrist = R3_0.T @ R_target
    
    w11, w12, w13 = R_wrist[0,0], R_wrist[0,1], R_wrist[0,2]
    w21, w22, w23 = R_wrist[1,0], R_wrist[1,1], R_wrist[1,2]
    w31, w32, w33 = R_wrist[2,0], R_wrist[2,1], R_wrist[2,2]

    theta5 = np.arctan2(np.sqrt(w31**2 + w32**2), w33)

    # Singularity handling
    if np.abs(np.sin(theta5)) > 1e-5:
        theta4 = np.arctan2(w23, w13)
        theta6 = np.arctan2(w32, -w31)
    else:
        theta4 = 0.0  
        theta6 = np.arctan2(-w12, w11)

    theta_angles = [theta1, theta2, theta3, theta4, theta5, theta6]

    # --- CRITICAL STEP: Hardware Limit Validation ---
    for i in range(6):
        min_lim, max_lim = bounds[i]
        angle = theta_angles[i]
        
        # Smart normalize for overlapping continuous joints (theta4 and theta6)
        # If an angle is slightly past 2*pi, wrap it to see if it fits inside bounds
        if max_lim - min_lim >= 2 * np.pi:
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            theta_angles[i] = angle # Update wrapped angle back into the array

        # Check if joint breaches the physical threshold limits
        if angle < min_lim or angle > max_lim:
            deg_calc = np.degrees(angle)
            deg_min = np.degrees(min_lim)
            deg_max = np.degrees(max_lim)
            raise ValueError(
                f"Hardware Limit Error: Joint theta{i+1} violated physical constraints! "
                f"Calculated: {deg_calc:.2f}°, Safe Range: [{deg_min:.1f}°, {deg_max:.1f}°]"
            )

    return theta_angles

def position_is_possible(target_pos):
    try:
        solve_ik_with_bounds(target_pos)
        return True
    except ValueError:
        return False
    
# =============================================================================
# GRIPPER / CLUB TIP POSITION
# =============================================================================

def get_gripper_pos(club_tip_pos):
    """
    Returns the [X, Y, Z] position of the CLUB TIP given the club tip position.
    q: list or array of 6 joint angles (in radians)
    """
    T_0_tip = get_matrix_look_at(*club_tip_pos)

    # write translation matrix along z-axis of the tool frame by L_CLUB_VAL
    T_tip_6 = np.eye(4)
    T_tip_6[2, 3] = - L_CLUB_VAL

    # compute new transformation matrix after translation
    # T_0_6 = T_0_tip @ T_tip_6 
    T_0_6 = T_0_tip @ T_tip_6

    gripper_pos = T_0_6[:3, 3]  # Extract the position of the gripper from the transformation matrix
    return gripper_pos


# =============================================================================
# WAYPOINT CLAMPING
# =============================================================================

MAX_REACH_SAFETY = 0.95   # stay inside 95% of geometric reach
RADIUS_STEPS     = np.linspace(1.0, 0.5, 26)   # 100% -> 50% of tip radius
ANGLE_STEPS      = np.linspace(0.0, np.pi/2, 25)  # 0° -> 90° azimuth shift


def _tip_is_feasible(P_tip):
    """True if the flange corresponding to this tip passes IK + reach check."""
    try:
        flange = get_gripper_pos(P_tip)
    except Exception:
        return False
    if np.linalg.norm(flange) > MAX_REACH * MAX_REACH_SAFETY:
        return False
    try:
        solve_ik_with_bounds(flange)
        return True
    except ValueError:
        return False


def clamp_waypoint(target_pos):
    """
    Keep the waypoint as-is if reachable. Otherwise:
      1) shrink its distance from the base (scale radius),
      2) if that still fails, rotate its azimuth (change XY angle),
      3) if all fails, raise (something is fundamentally wrong).
    Returns the (possibly modified) tip position.
    """
    P_tip = np.asarray(target_pos, float)

    if _tip_is_feasible(P_tip):
        return P_tip

    # --- 1. Radius shrink (keeps direction, shortens distance from base) ---
    r = np.linalg.norm(P_tip)
    if r > 1e-9: # avoid division by zero
        u = P_tip / r # = unit vector pointing from the base to the tip 

        for scale in RADIUS_STEPS: 
            P_try = u * r * scale # scale down the radius
            if _tip_is_feasible(P_try):
                print(f"[clamp] {P_tip.round(3)} -> {P_try.round(3)}  (radius x{scale:.2f})")
                return P_try


    # Only reached if no radius scaling worked. 
    # So the problem is not "too far" but something like the ball being at an awkward angle where joint limits bite.
    # --- 2. Azimuth rotation (rotate around Z, keep radius and height) ---

    rho = np.hypot(P_tip[0], P_tip[1])      # horizontal distance from Z-axis
    phi = np.arctan2(P_tip[1], P_tip[0])    # current azimuth
    z   = P_tip[2]                          # height unchanged

    # Decompose P_tip into cylindrical coordinates (rho, phi, z)
    for dphi in ANGLE_STEPS:
        for sign in (+1, -1):
            phi_new = phi + sign * dphi # rotate clockwise or counterclockwise
            P_try = np.array([rho * np.cos(phi_new),
                              rho * np.sin(phi_new),
                              z])
            if _tip_is_feasible(P_try):
                print(f"[clamp] {P_tip.round(3)} -> {P_try.round(3)}  (azimuth {np.degrees(sign*dphi):+.0f}°)")
                return P_try

    raise ValueError(f"[clamp] waypoint {P_tip.round(3)} unreachable even after radius+angle mods")


def clamped_waypoints(P_ball):
    """Build the 4 waypoints, clamp each individually."""
    tips = find_all_waypoints(P_ball)
    return np.array([clamp_waypoint(P) for P in tips])


# =============================================================================
# WAYPOINT GENERATION
# =============================================================================
# --- Positions for the swing trajectory (all in metres, arbitrary ball) ---

# NOTE: ADD CORNER CASES + CLAMPING

def swing_direction(alpha_deg):
    """Unit swing direction in the XY plane. alpha=0 -> +X, 90 -> +Y."""
    a = np.deg2rad(alpha_deg)
    return np.array([np.cos(a), np.sin(a), 0.0])

def find_P0(P_ball, L0=0.10, h0=0.03):
    """Ready/start: behind the ball along -d, raised by h0."""
    P0 = np.asarray(P_ball, float) - L0 * np.array([1.0, 0, 0]) + h0 * np.array([0, 0, 1.0])
    P0 = clamp_waypoint(P0)
    return P0


def find_P1(P_ball, Lb=0.10, angle = 90, h1 = 0.03):
    """Backswing: on the -d side of the ball."""
    P1 = find_P0(P_ball) - Lb * np.array([1.0, 0, 0]) + h1 * np.array([0, 0, 1.0])
    P1 = clamp_waypoint(P1)
    return P1

def find_P2(P_ball):
    """Hit: exactly at the ball."""
    return np.asarray(P_ball, float).copy()

def find_P3(P_ball, Lf=0.15, angle = 30, h2 = 0.03 ):
    """Follow-through: on the +d side of the ball."""
    P3 = np.asarray(P_ball, float) + Lf * np.array([1.0, 0, 0]) + h2 * np.array([0, 0, 1.0])
    P3 = clamp_waypoint(P3)
    return P3



def find_P_x(P_ball, L0=0.10, h0=0.03, Lb=0.10, h1=0.03, Lf=0.15, angle3=30, h2=0.03):
    return np.array([find_P0(P_ball, L0, h0)[0],
                     find_P1(P_ball, Lb, h1)[0],
                     find_P2(P_ball)[0],
                     find_P3(P_ball, Lf, angle3, h2)[0]])

def find_P_y(P_ball, L0=0.10, h0=0.03, Lb=0.10, h1=0.03, Lf=0.15, angle3=30, h2=0.03):
    return np.array([find_P0(P_ball, L0, h0)[1],
                     find_P1(P_ball, Lb, h1)[1],
                     find_P2(P_ball)[1],
                     find_P3(P_ball, Lf, angle3, h2)[1]])

def find_P_z(P_ball, L0=0.10, h0=0.03, Lb=0.10, h1=0.03, Lf=0.15, angle3=30, h2=0.03):
    return np.array([find_P0(P_ball, L0, h0)[2],
                     find_P1(P_ball, Lb, h1)[2],
                     find_P2(P_ball)[2],
                     find_P3(P_ball, Lf, angle3, h2)[2]])



def find_all_waypoints(P_ball, L0=0.10, h0=0.03,
                       Lb=0.10, h1=0.03,
                       Lf=0.15, angle3=30, h2=0.03):
    return np.vstack([
        find_P0(P_ball, L0, h0),
        find_P1(P_ball, Lb, h1),
        find_P2(P_ball),
        find_P3(P_ball, Lf, angle3, h2),
    ])



# =============================================================================
# TIMING
# ============================================================================

def find_t1(P_ball, v_slow=np.pi/8): # Slower backswing
    P0 = find_P0(P_ball)
    P1 = find_P1(P_ball)                       
    return np.linalg.norm(P1 - P0) / v_slow

def find_t2(P_ball, v_max=np.pi/4): # Slower hit
    P1 = find_P1(P_ball)                       
    P2 = find_P2(P_ball)
    return find_t1(P_ball) + np.linalg.norm(P2 - P1) / v_max

def find_t3(P_ball, v_max=np.pi/4): # Slower follow-through
    P2 = find_P2(P_ball)
    P3 = find_P3(P_ball)
    return find_t2(P_ball) + np.linalg.norm(P3 - P2) / v_max

def find_time_steps_from_waypoints(tips, v_slow=np.pi/4, v_max=3*np.pi/4):
    """Calculate time steps based on the physical distance between the CLAMPED waypoints.
    PARAMETERS:
        tips: np.array of shape (4,3) containing the 4 clamped waypoints (P0, P1, P2, P3)
        v_slow: float, the speed for the first segment (P0 to P1)
        v_max: float, float, the speed for the remaining segments (P1 to P2 and P2 to P3)
    RETURNS:
        np.array of shape (4,) containing the time steps [t0, t1, t2, t3]
    """
    d01 = np.linalg.norm(tips[1] - tips[0])
    d12 = np.linalg.norm(tips[2] - tips[1])
    d23 = np.linalg.norm(tips[3] - tips[2])
    
    t1 = d01 / v_slow
    t2 = t1 + (d12 / v_max)
    t3 = t2 + (d23 / v_max)
    return np.array([0.0, t1, t2, t3])




# =============================================================================
# CUBIC SPLINE
# =============================================================================


def cs_x(P_ball, t): 
    return CubicSpline(t, find_P_x(P_ball), bc_type='clamped')

def cs_y(P_ball, t): 
    return CubicSpline(t, find_P_y(P_ball), bc_type='clamped')

def cs_z(P_ball, t): 
    return CubicSpline(t, find_P_z(P_ball), bc_type='clamped')

def t_fine(P_ball, step_number):
    t3 = find_t3(P_ball)
    return np.linspace(0.0, t3, step_number) 
    
def matrix_of_path(P_ball, step_number=100):
    t = find_time_steps_from_waypoints(P_ball)
    tq = t_fine(P_ball, step_number)

    spline_x = cs_x(P_ball, t)                   
    spline_y = cs_y(P_ball, t)
    spline_z = cs_z(P_ball, t)

    return np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])

# =============================================================================
# Velocity Clamping
# =============================================================================

def clamp_velocity(tips_clamped, t_corners, max_vel=MAX_JOINT_VELOCITY):
    """
    Checks the max joint velocity of the trajectory. 
    If it exceeds max_vel, it scales the time steps to slow the trajectory down.
    Returns the updated t_corners.
    """
    # 1. Build temporary splines with the current time steps
    spline_x = CubicSpline(t_corners, tips_clamped[:, 0], bc_type='clamped')
    spline_y = CubicSpline(t_corners, tips_clamped[:, 1], bc_type='clamped')
    spline_z = CubicSpline(t_corners, tips_clamped[:, 2], bc_type='clamped')

    # 2. Sample the path and solve IK
    tq = np.linspace(0.0, t_corners[-1], 100)
    path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
    
    joint_traj = []
    for target in path_xyz:
        gripper_pos = get_gripper_pos(target)
        joint_traj.append(solve_ik_with_bounds(gripper_pos))
    joint_traj = np.array(joint_traj)

    # 3. Calculate velocities
    dq = np.gradient(joint_traj, tq, axis=0)
    vmax = np.max(np.abs(dq))

    # 4. Scale time if necessary
    if vmax > max_vel:
        scale_factor = vmax / max_vel
        print(f"[velocity clamp] Max velocity {vmax:.2f} rad/s exceeds limit {max_vel:.2f}. Slowing down by x{scale_factor:.2f}.")
        t_corners = t_corners * scale_factor
    else:
        print(f"[velocity clamp] Max velocity {vmax:.2f} rad/s is within limit. No change.")
        
    return t_corners
    
# =============================================================================
# EXAMPLE TRAJECTORY
# ============================================================================

ball_xyz = np.array([0.25, 0.0, 0.15])



# Build the 4 waypoints, clamped them if necessary


tips_clamped = clamped_waypoints(ball_xyz)
print("Clamped waypoints:")
for i, P in enumerate(tips_clamped):
    print(f"  P{i}: {P.round(3)}")

# Build the spline from the CLAMPED tips
t_corners = find_time_steps_from_waypoints(tips_clamped)
t_corners = clamp_velocity(tips_clamped, t_corners)
spline_x = CubicSpline(t_corners, tips_clamped[:, 0], bc_type='clamped')
spline_y = CubicSpline(t_corners, tips_clamped[:, 1], bc_type='clamped')
spline_z = CubicSpline(t_corners, tips_clamped[:, 2], bc_type='clamped')

tq = np.linspace(0.0, t_corners[-1], 100)
path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])

# solve inverse kinematics iteratively 
joint_trajectory = []


for target in path_xyz:
    gripper_pos = get_gripper_pos(target)
    try:
        q_sol = solve_ik_with_bounds(gripper_pos)
    except ValueError as e:
        print(f"IK failed at {target}: {e}")
        raise
    joint_trajectory.append(q_sol) 

# array of shape (100, 6) containing joint angles over the trajectory 
joint_trajectory = np.array(joint_trajectory)


# Velocity check on the spline
#tq = t_fine(ball_xyz, step_number=100) # time vector for the spline

dq = np.gradient(joint_trajectory, tq, axis=0) # compute joint velocities at each time step of each joint
vmax = np.max(np.abs(dq))
print(f"Max joint velocity: {vmax:.2f} rad/s (limit {MAX_JOINT_VELOCITY})")
print(f"Feasible: {vmax <= MAX_JOINT_VELOCITY}")

# ---------------- combined plot ----------------

# Velocity (one sample fewer than trajectory)
dt = tq[1] - tq[0]
joint_velocities = np.diff(joint_trajectory, axis=0) / dt
t_vel = tq[:-1]

# Calculate the Cartesian velocity of the club tip
tip_displacements = np.diff(path_xyz, axis=0)
tip_velocities = tip_displacements / dt
tip_speed = np.linalg.norm(tip_velocities, axis=1) # Magnitude of velocity vector


# Create a 1x3 grid of plots
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
ax2.axhline(MAX_JOINT_VELOCITY, color='r', linestyle='--',
            label=f'limit = {MAX_JOINT_VELOCITY:.2f} rad/s')

# Add vertical lines for the 4 waypoints
for i, t_c in enumerate(t_corners):
    ax2.axvline(t_c, color='gray', linestyle=':', alpha=0.7)
    ax2.text(t_c, ax2.get_ylim()[1]*0.9, f'P{i}', ha='center', color='black')

ax2.set_xlabel('t [s]')
ax2.set_ylabel('|dq/dt| [rad/s]')
ax2.set_title('Joint velocities vs time')
ax2.legend(fontsize=8)
ax2.grid(True)

# --- Right: Club tip Cartesian velocity vs time ---
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t_vel, tip_speed, color='purple', lw=2, label='Tip speed')

# Add vertical lines for the 4 waypoints
for i, t_c in enumerate(t_corners):
    ax3.axvline(t_c, color='gray', linestyle=':', alpha=0.7)
    ax3.text(t_c, ax3.get_ylim()[1]*0.9, f'P{i}', ha='center', color='black')

ax3.set_xlabel('t [s]')
ax3.set_ylabel('Speed [m/s]')
ax3.set_title('Club tip velocity vs time')
ax3.legend(fontsize=8)
ax3.grid(True)

plt.tight_layout()
plt.show()