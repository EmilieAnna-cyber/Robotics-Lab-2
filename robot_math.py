# TASK 4 & TASK 2
# robot_math.py
import numpy as np
import sympy as sp
import kinematics_2 as k
from kinematics_2 import (
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
L_CLUB_VAL = 0.1 
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
    return np.array([forward_x(*q), forward_y(*q), forward_z(*q)])  

def get_matrix_look_at(x, y, z, target_point=np.array([0, 0, 0])):
    position = np.array([x, y, z])
    z_axis = target_point - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    up = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.99 else np.array([0, 1, 0])
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T_target = np.identity(4)
    T_target[:3, 0] = x_axis; T_target[:3, 1] = y_axis; T_target[:3, 2] = z_axis; T_target[:3, 3] = position
    return T_target

def solve_ik_with_bounds(target_pos, bounds=JOINT_BOUNDS):
    A2v, A3v, D4v, D1v, D6v = 0.2, 0.2276, 0.087, 0.2433, 0.0615
    x, y, z = target_pos[0], target_pos[1], target_pos[2]
    matrix = get_matrix_look_at(x, y, z)
    r13, r23, r33 = matrix[0, 2], matrix[1, 2], matrix[2, 2]
    x_w, y_w, z_w = x - D6v * r13, y - D6v * r23, z - D6v * r33
    
    theta1 = np.arctan2(y_w, x_w)
    kx = x_w * np.cos(theta1) + y_w * np.sin(theta1)
    kz = z_w - D1v
    num = kx**2 + kz**2 - A2v**2 - (A3v**2 + D4v**2)
    den = 2 * A2v * np.sqrt(A3v**2 + D4v**2)
    cos_theta3 = num / den
    
    if cos_theta3 < -1.0 or cos_theta3 > 1.0:
        raise ValueError("OUT OF REACH")
    
    sin_theta3 = np.sqrt(1 - cos_theta3**2)
    theta3 = np.arctan2(sin_theta3, cos_theta3) - np.arctan2(D4v, A3v)
    num_t2 = A3v * np.sin(theta3) + D4v * np.cos(theta3)
    den_t2 = A2v + A3v * np.cos(theta3) - D4v * np.sin(theta3)
    theta2 = np.arctan2(kz, kx) - np.arctan2(num_t2, den_t2)

    c1, s1 = np.cos(theta1), np.sin(theta1)
    c23, s23 = np.cos(theta2 - theta3), np.sin(theta2 - theta3)
    R3_0 = np.array([[c1*c23, -c1*s23, s1], [s1*c23, -s1*s23, -c1], [s23, c23, 0]])
    R_wrist = R3_0.T @ matrix[:3, :3]
    
    w11, w12, w13 = R_wrist[0,0], R_wrist[0,1], R_wrist[0,2]
    w21, w22, w23 = R_wrist[1,0], R_wrist[1,1], R_wrist[1,2]
    w31, w32, w33 = R_wrist[2,0], R_wrist[2,1], R_wrist[2,2]
    
    theta5 = np.arctan2(np.sqrt(w31**2 + w32**2), w33)
    if np.abs(np.sin(theta5)) > 1e-5:
        theta4 = np.arctan2(w23, w13)
        theta6 = np.arctan2(w32, -w31)
    else:
        theta4 = 0.0  
        theta6 = np.arctan2(-w12, w11)

    theta_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
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
    T_0_tip = get_matrix_look_at(*club_tip_pos)
    T_tip_6 = np.eye(4)
    T_tip_6[2, 3] = - L_CLUB_VAL
    T_0_6_mat = T_0_tip @ T_tip_6
    return T_0_6_mat[:3, 3]

MAX_REACH_SAFETY = 0.95
RADIUS_STEPS = np.linspace(1.0, 0.5, 26)
ANGLE_STEPS = np.linspace(0.0, np.pi/2, 25)

def _tip_is_feasible(P_tip):
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
def find_P0(P_ball, L0=0.10, h0=0.03):
    P0 = np.asarray(P_ball, float) - L0 * np.array([1.0, 0, 0]) + h0 * np.array([0, 0, 1.0])
    return clamp_waypoint(P0)

def find_P1(P_ball, Lb=0.10, angle=90, h1=0.03):
    P1 = find_P0(P_ball) - Lb * np.array([1.0, 0, 0]) + h1 * np.array([0, 0, 1.0])
    return clamp_waypoint(P1)

def find_P2(P_ball):
    return np.asarray(P_ball, float).copy()

def find_P3(P_ball, Lf=0.15, angle=30, h2=0.03):
    P3 = np.asarray(P_ball, float) + Lf * np.array([1.0, 0, 0]) + h2 * np.array([0, 0, 1.0])
    return clamp_waypoint(P3)

def find_all_waypoints(P_ball, L0=0.10, h0=0.03, Lb=0.10, h1=0.03, Lf=0.15, angle3=30, h2=0.03):
    return np.vstack([
        find_P0(P_ball, L0, h0),
        find_P1(P_ball, Lb, h1),
        find_P2(P_ball),
        find_P3(P_ball, Lf, angle3, h2),
    ])

def find_time_steps_from_waypoints(tips, v_slow=np.pi/4, v_max=3*np.pi/4):
    d01 = np.linalg.norm(tips[1] - tips[0])
    d12 = np.linalg.norm(tips[2] - tips[1])
    d23 = np.linalg.norm(tips[3] - tips[2])
    t1 = d01 / v_slow
    t2 = t1 + (d12 / v_max)
    t3 = t2 + (d23 / v_max)
    return np.array([0.0, t1, t2, t3])

def clamp_velocity(tips_clamped, t_corners, max_vel=MAX_JOINT_VELOCITY):
    from scipy.interpolate import CubicSpline
    spline_x = CubicSpline(t_corners, tips_clamped[:, 0], bc_type='clamped')
    spline_y = CubicSpline(t_corners, tips_clamped[:, 1], bc_type='clamped')
    spline_z = CubicSpline(t_corners, tips_clamped[:, 2], bc_type='clamped')
    tq = np.linspace(0.0, t_corners[-1], 100)
    path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
    joint_traj = [solve_ik_with_bounds(get_gripper_pos(target)) for target in path_xyz]
    joint_traj = np.unwrap(np.array(joint_traj),axis=0) #added unwrap to remove the invalid velocity
    dq = np.gradient(joint_traj, tq, axis=0)
    vmax = np.max(np.abs(dq))
    if vmax > max_vel:
        scale_factor = (vmax / max_vel)*1.05 #added 5% safety buffer, so that the floating point doesn't hit exactly pi
        print(f"[velocity clamp] Slowing down by x{scale_factor:.2f}.")
        t_corners = t_corners * scale_factor
    return t_corners #t i just import the first file into it to be able to use the code and variables from the first 