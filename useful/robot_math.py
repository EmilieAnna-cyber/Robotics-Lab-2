# TASK 2
# robot_math.py
# toolbox of the swing: look-at matrix (orientation of the club), inverse kinematics, clamping of the waypoints,
# waypoints P0..P3, timing and velocity clamping
# this file is not run on its own: it is imported by main_swing.py and the other scripts
# club length: L_CLUB_VAL (0 = the flange is the club tip)
import numpy as np
import sympy as sp
import useful.kinematics_2 as k
from useful.kinematics_2 import (
    D1, A2, A3, D4, D6, T_0_6,
    X_tip, Y_tip, Z_tip, L_CLUB,
    theta1, theta2, theta3, theta4, theta5, theta6,
)

# =============================================================================
# CONSTANTS
# =============================================================================
JOINT_BOUNDS = [
    (-np.pi, np.pi), (-5*np.pi/6, 5*np.pi/6), (-3.5*np.pi/180, 300*np.pi/180),
    (-2*np.pi, 2*np.pi), (-124*np.pi/180, 124*np.pi/180), (-2*np.pi, 2*np.pi)
]
MAX_JOINT_VELOCITY = np.pi
MAX_REACH = 0.2 + np.sqrt(0.2276**2 + 0.087**2) 
L_CLUB_VAL = 0.0 
LINK_VALUES = { L_CLUB: L_CLUB_VAL }

# =============================================================================
# FK & IK
# =============================================================================
X_raw = T_0_6[0, 3].subs(LINK_VALUES)
Y_raw = T_0_6[1, 3].subs(LINK_VALUES)
Z_raw = T_0_6[2, 3].subs(LINK_VALUES)

forward_x = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), X_raw, 'numpy')
forward_y = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), Y_raw, 'numpy')
forward_z = sp.lambdify((theta1, theta2, theta3, theta4, theta5, theta6), Z_raw, 'numpy')

def get_ee_position(q):
    '''
    goal: compute the flange position with our direct kinematics (task 1)
    PARAMS:
        q: list of the 6 joint angles (radians)
    RETURNS:
        3D position of the flange in the robot's base frame (m)
    '''
    return np.array([forward_x(*q), forward_y(*q), forward_z(*q)])  

# the club points AWAY from a swing pivot (the "golfer's shoulders"),
# instead of pointing at the robot base. Pivot is set in find_all_waypoints().
SWING_PIVOT = np.array([0.0, 0.0, 0.0])

def get_matrix_look_at(x, y, z, target_point=None):
    ''' 
    this function returns a 4x4 transformation matrix that positions the end-effector at (x, y, z) and orients it to "look at" the target_point.
    PARAMETERS:
    x, y, z: coordinates of the end-effector position. => end effector position is the tip of the club
    target_point: a 3D point that the end-effector should "look away from".

    RETURNS:
    a 4x4 transformation matrix that positions the end-effector at (x, y, z) and orients it to "look at" the target_point. 
    If target_point is None, it defaults to SWING_PIVOT.
    SWING_PIVOT is a global variable that can be set in find_all_waypoints().'''

    if target_point is None:
        target_point = SWING_PIVOT
    position = np.array([x, y, z])
    # z-axis points from the end-effector position to the target point (the "look at" direction)
    z_axis = position - target_point
    z_axis = z_axis / np.linalg.norm(z_axis)

    if abs(z_axis[2]) < 0.99: # if z_axis is not almost vertical
        up = np.array([0, 0, 1]) # use the global up vector
    else:
        up = np.array([0, 1, 0]) # because z_axis is almost vertical, use a different up vector to avoid singularity

    # x_axis is perpendicular to both up and z_axis
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # y_axis is perpendicular to both z_axis and x_axis
    y_axis = np.cross(z_axis, x_axis)
    T_target = np.identity(4)
    T_target[:3, 0] = x_axis; T_target[:3, 1] = y_axis; T_target[:3, 2] = z_axis; T_target[:3, 3] = position
    return T_target

def solve_ik_with_bounds(target_pos, bounds=JOINT_BOUNDS, tip_pos = None):
    '''
    Solves the inverse kinematics problem for the robot arm with joint angle bounds.
    
    PARAMS:
    target_pos: 3D position of the flange in the robot's base frame (= the club tip when L_CLUB_VAL = 0).
    bounds: list of tuples specifying the joint limits for each of the 6 joints. Each tuple is (min_angle, max_angle) in radians.
    tip_pos: 3D position of the club tip. The orientation of the club is computed there (look-at matrix).
             If None, target_pos is used (only correct when L_CLUB_VAL = 0).

    RETURNS:
    theta_angles: list of 6 joint angles (in radians) that achieve the desired end-effector position, while respecting the joint limits. 
    Raises ValueError if the target position is unreachable or if any joint angle exceeds its specified bounds.
    '''
    # Define the robot's link lengths and offsets
    A2v, A3v, D4v, D1v, D6v = 0.2, 0.087, 0.2276, 0.2433, 0.0615   

    # Compute the wrist center position
    x, y, z = target_pos[0], target_pos[1], target_pos[2]
    # CLUB: the orientation of the club is computed at the club TIP, not at the flange
    if tip_pos is None:
        tip_pos = target_pos
    matrix = get_matrix_look_at(*tip_pos)
    r13, r23, r33 = matrix[0, 2], matrix[1, 2], matrix[2, 2]

    # wrist center position => only depends on theta1, theta2, theta3 (the first three joints), not on the wrist orientation (theta4, theta5, theta6)
    x_w, y_w, z_w = x - D6v * r13, y - D6v * r23, z - D6v * r33 # wrist center position
    
    theta1 = np.arctan2(y_w, x_w)
    kx = x_w * np.cos(theta1) + y_w * np.sin(theta1)
    kz = z_w - D1v

    #  solve theta3:

    # R_el: the distance from the shoulder to the wrist center in the plane of the first three joints


    R_el = np.sqrt(A3v**2 + D4v**2)          # J3 -> wrist centre
    beta = np.arctan2(D4v, A3v)
    num = kx**2 + kz**2 - A2v**2 - R_el**2
    den = 2 * A2v * R_el
    sin_t3b = num / den

    if sin_t3b < -1.0 or sin_t3b > 1.0:
        raise ValueError("OUT OF REACH")

    theta3 = beta + np.arcsin(sin_t3b)       # elbow-up branch (theta3 = 0 at home)

    # solve theta2:
    g = beta - theta3
    num_t2 = A2v - R_el * np.sin(g)
    den_t2 = R_el * np.cos(g)
    theta2 = np.arctan2(num_t2 * kx - den_t2 * kz, num_t2 * kz + den_t2 * kx)


    # solve theta4, theta5, theta6:

    # easier for reading to define c1, s1, c23, s23
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c23, s23 = np.cos(theta2 - theta3), np.sin(theta2 - theta3)
    R3_0 = np.array([[c1*c23, s1, -c1*s23], [s1*c23, -c1, -s1*s23], [-s23, 0, -c23]])   
    R_wrist = R3_0.T @ matrix[:3, :3]
    
    w11, w12, w13 = R_wrist[0,0], R_wrist[0,1], R_wrist[0,2]
    w21, w22, w23 = R_wrist[1,0], R_wrist[1,1], R_wrist[1,2]
    w31, w32, w33 = R_wrist[2,0], R_wrist[2,1], R_wrist[2,2]
    
    theta5 = np.arctan2(np.sqrt(w13**2 + w23**2), w33)

    if np.abs(np.sin(theta5)) > 1e-5: # if theta5 is not near 0 or pi, we can solve for theta4 and theta6
        theta4 = np.arctan2(-w23, -w13)
        theta6 = np.arctan2(-w32, w31)

    else: # if theta5 is near 0 or pi, we have a singularity and cannot solve for theta4 and theta6 independently. We can set theta4 = 0 and solve for theta6 instead.
        theta4 = 0.0  
        theta6 = np.arctan2(w21, w11)

    theta_angles = [theta1, theta2, theta3, theta4, theta5, theta6]

    # Check joint limits
    for i in range(6):
        min_lim, max_lim = bounds[i]
        angle = theta_angles[i]
        if max_lim - min_lim >= 2 * np.pi: 
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            theta_angles[i] = angle
        if angle < min_lim or angle > max_lim:
            raise ValueError("JOINT LIMIT")
    return theta_angles

# =============================================================================
# CLUB TIP & CLAMPING
# =============================================================================
def get_gripper_pos(club_tip_pos):
    '''
    goal: compute the flange position from the club tip position
    PARAMS:
        club_tip_pos: 3D position of the club tip in the robot's base frame
    RETURNS:
        gripper_pos: 3D position of the flange in the robot's base frame
    '''
    T_0_tip = get_matrix_look_at(*club_tip_pos)
    T_tip_6 = np.eye(4)
    T_tip_6[2, 3] = - L_CLUB_VAL
    T_0_6_mat = T_0_tip @ T_tip_6
    return T_0_6_mat[:3, 3]

MAX_REACH_SAFETY = 0.95
RADIUS_STEPS = np.linspace(1.0, 0.5, 26)
ANGLE_STEPS = np.linspace(0.0, np.pi/2, 25)

def _tip_is_feasible(P_tip):
    '''
    goal: check if the robot can put the club tip at P_tip
    PARAMS:
        P_tip: 3D position of the club tip
    RETURNS:
        True if the flange is inside 95% of the reach (measured from the shoulder)
        and the IK finds a solution inside the joint limits, False otherwise
    '''
    try:
        flange = get_gripper_pos(P_tip)
    except Exception:
        return False
    if np.linalg.norm(flange - np.array([0, 0, D1])) > MAX_REACH * MAX_REACH_SAFETY:   # FIX: reach is from the shoulder
        return False
    try:
        solve_ik_with_bounds(flange, tip_pos = P_tip)
        return True
    except ValueError:
        return False

def clamp_waypoint(target_pos):
    '''
    goal: move a waypoint that the robot cannot reach to a reachable point close to it
        stage 1: shrink it towards the base origin, p_new = s p with s = 1.00, 0.98, ..., 0.50
        stage 2: rotate it around the vertical axis of the base, phi -> phi +- dphi with dphi = 0 ... 90 deg (height kept)
    PARAMS:
        target_pos: 3D position of the waypoint (club tip)
    RETURNS:
        the waypoint itself if it is reachable, otherwise the first reachable point found
        Raises ValueError if no reachable point is found
    '''
    P_tip = np.asarray(target_pos, float)
    if _tip_is_feasible(P_tip): return P_tip
    r = np.linalg.norm(P_tip)
    if r > 1e-9:
        u = P_tip / r
        for scale in RADIUS_STEPS: 
            P_try = u * r * scale
            if _tip_is_feasible(P_try): return P_try
    rho = np.hypot(P_tip[0], P_tip[1])
    phi = np.arctan2(P_tip[1], P_tip[0])
    z = P_tip[2]
    for dphi in ANGLE_STEPS:
        for sign in (+1, -1):
            phi_new = phi + sign * dphi 
            P_try = np.array([rho * np.cos(phi_new), rho * np.sin(phi_new), z])
            if _tip_is_feasible(P_try): return P_try
    raise ValueError(f"waypoint {P_tip} unreachable")

# =============================================================================
# WAYPOINTS & TIMING
# =============================================================================
SWING_DIR = np.array([0, 1.0, 0])     # direction the ball is hit
IN_DIR = np.array([-1.0, 0, 0])       # towards the robot
UP = np.array([0, 0, 1.0])

def find_P0(P_ball, L0=0.05, h0=0.01):              # address, just behind the ball
    '''
    goal: address position, just behind the ball and slightly higher
    PARAMS:
        P_ball: 3D position of the ball
        L0: distance behind the ball, along the swing direction (m)
        h0: height above the ball (m)
    RETURNS:
        P0: 3D position of the club tip (clamped if needed)
    '''
    P0 = np.asarray(P_ball, float) - L0 * SWING_DIR + h0 * UP
    return clamp_waypoint(P0)

def find_P1(P_ball, Lb=0.15, in1=0.12, h1=0.10):    # top of the backswing
    '''
    goal: top of the backswing, behind the ball, closer to the robot and higher
    PARAMS:
        P_ball: 3D position of the ball
        Lb: distance behind the ball, along the swing direction (m)
        in1: distance towards the robot (m)
        h1: height above the ball (m)
    RETURNS:
        P1: 3D position of the club tip (clamped if needed)
    '''
    P1 = np.asarray(P_ball, float) - Lb * SWING_DIR + in1 * IN_DIR + h1 * UP
    return clamp_waypoint(P1)

def find_P2(P_ball):                                 # impact
    '''
    goal: impact, the club tip is on the ball
    PARAMS:
        P_ball: 3D position of the ball
    RETURNS:
        P2: copy of the ball position (never clamped)
    '''
    return np.asarray(P_ball, float).copy()

def find_P3(P_ball, Lf=0.15, in3=0.12, h2=0.10):    # follow-through
    '''
    goal: follow-through, after the ball, closer to the robot and higher
    PARAMS:
        P_ball: 3D position of the ball
        Lf: distance after the ball, along the swing direction (m)
        in3: distance towards the robot (m)
        h2: height above the ball (m)
    RETURNS:
        P3: 3D position of the club tip (clamped if needed)
    '''
    P3 = np.asarray(P_ball, float) + Lf * SWING_DIR + in3 * IN_DIR + h2 * UP
    return clamp_waypoint(P3)

def find_all_waypoints(P_ball, L0=0.05, h0=0.01, Lb=0.15, in1=0.12, h1=0.10, Lf=0.15, in3=0.12, h2=0.10,
                       lean_deg=60, radius=None):
    '''
    goal: set the swing pivot c and build the 4 waypoints of the swing
        c = P_ball + radius * (sin(lean) * IN_DIR + cos(lean) * UP), saved in the global SWING_PIVOT
    PARAMS:
        P_ball: 3D position of the ball
        L0, h0, Lb, in1, h1, Lf, in3, h2: offsets of the waypoints from the ball (m), see find_P0 ... find_P3
        lean_deg: angle of the pivot from the vertical above the ball (degrees)
        radius: distance ball - pivot (m). If None: max(0.15, L_CLUB_VAL + 0.05)
    RETURNS:
        4x3 array with the club tip positions P0, P1, P2, P3
    '''
    global SWING_PIVOT # modify global variable used in get_matrix_look_at() and solve_ik_with_bounds()
    if radius is None:   # CLUB: the pivot must be further from the tip than the club length
        radius = max(0.15, L_CLUB_VAL + 0.05)
    L = np.radians(lean_deg)
    SWING_PIVOT = np.asarray(P_ball, float) + radius * (np.sin(L) * IN_DIR + np.cos(L) * UP)
    return np.vstack([
        find_P0(P_ball, L0, h0),
        find_P1(P_ball, Lb, in1, h1),     
        find_P2(P_ball),
        find_P3(P_ball, Lf, in3, h2),
    ])

def find_time_steps_from_waypoints(tips, v_slow=np.pi/4, v_max=3*np.pi/4):
    '''
    goal: time at each waypoint, from the distances between them (before the velocity clamp)
    PARAMS:
        tips: 4x3 array of the waypoints P0..P3
        v_slow: speed of the backswing P0 -> P1 (m/s)
        v_max: speed of the club tip at impact P2 (m/s)
    RETURNS:
        t_corners: array [0, t1, t2, t3] (s)
    '''
    d01 = np.linalg.norm(tips[1] - tips[0])
    d12 = np.linalg.norm(tips[2] - tips[1])
    d23 = np.linalg.norm(tips[3] - tips[2])
    t1 = d01 / v_slow
    t2 = t1 + 2 * (d12 / v_max)   # accelerate from rest at P1 to v_max at P2 (factor 2 = constant acceleration)
    t3 = t2 + 2 * (d23 / v_max)   # decelerate from v_max at P2 to rest at P3
    return np.array([0.0, t1, t2, t3])

def find_waypoint_velocities(tips, t_corners):
    '''
    goal: velocity of the club tip at each waypoint: 0 at P0, P1, P3 and maximum at P2 (impact)
    PARAMS:
        tips: 4x3 array of the waypoints P0..P3
        t_corners: time at each waypoint (s)
    RETURNS:
        vel: 4x3 array of velocities (m/s). At P2: v_imp = 2 d12 / (t2 - t1), in the direction P1 -> P3
    '''
    # fastest at P2 (impact), moving along the swing direction P1 -> P3.
    vel = np.zeros_like(tips)
    direction = (tips[3] - tips[1]) / np.linalg.norm(tips[3] - tips[1])
    v_impact = 2 * np.linalg.norm(tips[2] - tips[1]) / (t_corners[2] - t_corners[1])
    vel[2] = v_impact * direction
    return vel

def clamp_velocity(tips_clamped, t_corners, max_vel=MAX_JOINT_VELOCITY):
    '''
    goal: slow the whole swing down if one joint goes faster than max_vel (same path, slower clock)
    PARAMS:
        tips_clamped: 4x3 array of the waypoints P0..P3
        t_corners: time at each waypoint (s)
        max_vel: joint speed limit (rad/s)
    RETURNS:
        t_corners: unchanged if the swing is slow enough, otherwise multiplied by alpha = 1.05 * max|dq| / max_vel
    '''
    from scipy.interpolate import CubicHermiteSpline  
    vel = find_waypoint_velocities(tips_clamped, t_corners)
    spline_x = CubicHermiteSpline(t_corners, tips_clamped[:, 0], vel[:, 0])
    spline_y = CubicHermiteSpline(t_corners, tips_clamped[:, 1], vel[:, 1])
    spline_z = CubicHermiteSpline(t_corners, tips_clamped[:, 2], vel[:, 2])
    tq = np.linspace(0.0, t_corners[-1], 100)
    path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
    joint_traj = [solve_ik_with_bounds(get_gripper_pos(target), tip_pos=target) for target in path_xyz]   # CLUB
    joint_traj = np.unwrap(np.array(joint_traj),axis=0) 
    dq = np.gradient(joint_traj, tq, axis=0)
    vmax = np.max(np.abs(dq))
    if vmax > max_vel:
        scale_factor = (vmax / max_vel)*1.05 # 5% safety buffer, so that the floating point doesn't hit exactly pi
        print(f"[velocity clamp] Slowing down by x{scale_factor:.2f}.")
        t_corners = t_corners * scale_factor
    return t_corners #t i just import the first file into it to be able to use the code and variables from the first