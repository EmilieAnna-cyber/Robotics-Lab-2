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

X_raw = T_0_6[0, 3]
Y_raw = T_0_6[1, 3]
Z_raw = T_0_6[2, 3]

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

# CLUB LENGTH: CHANGE LATER:

L_CLUB_VAL = 0.1 #m

LINK_VALUES = {
    L_CLUB: L_CLUB_VAL,
}



'''_________________________________________get a numerical function of the positions_________________________________________________________________'''

# replace theoretical values with the real ones with .subs

X_raw = X_raw.subs(LINK_VALUES)
Y_raw = Y_raw.subs(LINK_VALUES)
Z_raw = Z_raw.subs(LINK_VALUES)

# we are using both Sympy & NumPy: lambdify() allows to convert SymPy expressions into NumPy expression

def joint_coords_from_fixed_frame(Matrix_list, joint_i):
    T_0_i = k.T_i_j(Matrix_list,0,joint_i)
    # Extract raw position expressions
    X_raw = T_0_i[0, 3]
    Y_raw = T_0_i[1, 3]
    Z_raw = T_0_i[2, 3]
    joint_pos = [X_raw,Y_raw,Z_raw]

    return joint_pos

gripper_coord = joint_coords_from_fixed_frame(k.T_list, 6)


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



def get_matrix_look_at(x, y, z, target_point=np.array([0, 0, 0])):
    """
    Oriente l'outil du robot pour qu'il "regarde" un point cible de l'espace.
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


'''_________________________________________solve inverse kinematics_________________________________________________________________'''

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

def is_position_possible(target_pos):
    try:
        solve_ik_with_bounds(target_pos)
        return True
    except ValueError:
        return False

'''_________________________________________Positions for the swing trajectory (arbitrary ball position)_________________________________________________________________'''
# --- Positions for the swing trajectory (all in metres, arbitrary ball) ---

# NOTE: ADD CORNER CASES + CLAMPING

def swing_direction(alpha_deg):
    """Unit swing direction in the XY plane. alpha=0 -> +X, 90 -> +Y."""
    a = np.deg2rad(alpha_deg)
    return np.array([np.cos(a), np.sin(a), 0.0])

def find_P0(P_ball, L0=0.10, h0=0.03):
    """Ready/start: behind the ball along -d, raised by h0."""
    return np.asarray(P_ball, float) - L0 * np.array([1.0, 0, 0]) + h0 * np.array([0, 0, 1.0])

def find_P1(P_ball, Lb=0.10, angle = 90, h1 = 0.03):
    """Backswing: on the -d side of the ball."""
    return find_P0(P_ball) - Lb * np.array([1.0, 0, 0]) + h1 * np.array([0, 0, 1.0])

def find_P2(P_ball):
    """Hit: exactly at the ball."""
    return np.asarray(P_ball, float).copy()

def find_P3(P_ball, Lf=0.15, angle = 30, h2 = 0.03 ):
    """Follow-through: on the +d side of the ball."""
    return np.asarray(P_ball, float) + Lf * np.array([1.0, 0, 0]) + h2 * np.array([0, 0, 1.0])



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





'''def find_all_waypoints(P_ball, alpha_deg=0.0, L0=0.30, Lb=0.20, Lf=0.20, h0=0.05):
    """Return the (4,3) array of club-tip waypoints and the swing direction."""
    P_ball = np.asarray(P_ball, float)
    d = swing_direction(alpha_deg)
    P = np.vstack([find_P0(P_ball, d, L0, h0),
                   find_P1(P_ball, d, Lb),
                   find_P2(P_ball),
                   find_P3(P_ball, d, Lf)])
    return P, d'''

'''_________________________________________find time points_________________________________________________________________'''

def find_t1(P_ball, v_slow=np.pi/4):
    P0 = find_P0(P_ball)
    P1 = find_P1(P_ball)                       
    return np.linalg.norm(P1 - P0) / v_slow

def find_t2(P_ball, v_max=3*np.pi/4):
    P1 = find_P1(P_ball)                       
    P2 = find_P2(P_ball)
    return find_t1(P_ball) + np.linalg.norm(P2 - P1) / v_max

def find_t3(P_ball, v_max=3*np.pi/4):
    P2 = find_P2(P_ball)
    P3 = find_P3(P_ball)
    return find_t2(P_ball) + np.linalg.norm(P3 - P2) / v_max

def find_time_steps(P_ball):
    return np.array([0.0, find_t1(P_ball), find_t2(P_ball), find_t3(P_ball)])




'''_________________________________________cubic spline_________________________________________________________________'''


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
    t = find_time_steps(P_ball)
    tq = t_fine(P_ball, step_number)

    spline_x = cs_x(P_ball, t)                   
    spline_y = cs_y(P_ball, t)
    spline_z = cs_z(P_ball, t)

    return np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])

'''_________________________________________waypoint clamping (optional)_________________________________________________________________'''

SAFE_BALL = np.array([0.30, 0.0, 0.20])
SWING_DIR = np.array([1.0, 0.0, 0.0])


def clamp_waypoint(P_tip):
    """
    If P_tip is reachable, return it unchanged.
    If not, move it step by step toward SAFE_BALL until IK works.
    Returns the (possibly moved) tip position.
    """
    for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        P_try = P_tip + s * (SAFE_BALL - P_tip)

        # try this tip with the current club, then a shorter club
        for Lc in (L_CLUB_VAL, 0.05, 0.02):
            flange = P_try - Lc * SWING_DIR
            try:
                solve_ik_with_bounds(flange)
                if s > 0:
                    print(f"[clamp] waypoint {P_tip.round(3)} moved {100*s:.0f}% -> {P_try.round(3)}")
                return P_try
            except ValueError:
                continue

    # nothing worked, return SAFE_BALL
    print(f"[clamp] waypoint {P_tip.round(3)} unreachable, using SAFE_BALL")
    return SAFE_BALL.copy()


def clamped_waypoints(P_ball):
    """Build 4 waypoints, then clamp each one individually."""
    tips = find_all_waypoints(P_ball)
    return np.array([clamp_waypoint(P) for P in tips])


'''_________________________________________create many time samples_________________________________________________________________'''




def tip_to_flange(P_tip, P_ball):
    """Given the look-at-at-origin orientation, convert tip target to flange target."""
    z_tool = -np.asarray(P_tip, float)
    z_tool /= np.linalg.norm(z_tool)
    return np.asarray(P_tip, float) - L_CLUB_VAL * z_tool


    
'''_________________________________________example trajectory_________________________________________________________________'''

ball_xyz = np.array([0.30, 0.05, 0.20])



# Build the 4 corners, clamping any that are unreachable
tips_clamped = clamped_waypoints(ball_xyz)
print("Clamped waypoints:")
for i, P in enumerate(tips_clamped):
    print(f"  P{i}: {P.round(3)}")

# Build the spline from the CLAMPED tips
t_corners = find_time_steps(ball_xyz)
spline_x = CubicSpline(t_corners, tips_clamped[:, 0], bc_type='clamped')
spline_y = CubicSpline(t_corners, tips_clamped[:, 1], bc_type='clamped')
spline_z = CubicSpline(t_corners, tips_clamped[:, 2], bc_type='clamped')

tq = t_fine(ball_xyz, 100)
path_xyz = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])

# solve inverse kinematics iteratively 
joint_trajectory = []


for target in path_xyz:
    flange_target = target - L_CLUB_VAL * np.array([1.0, 0, 0])
    try:
        q_sol = solve_ik_with_bounds(flange_target)
    except ValueError as e:
        print(f"IK failed at {target}: {e}")
        raise
    joint_trajectory.append(q_sol)


joint_trajectory = np.array(joint_trajectory)


# Velocity check on the spline
tq = t_fine(ball_xyz, step_number=100)
dq = np.gradient(joint_trajectory, tq, axis=0)
vmax = np.max(np.abs(dq))
print(f"Max joint velocity: {vmax:.2f} rad/s (limit {MAX_JOINT_VELOCITY})")
print(f"Feasible: {vmax <= MAX_JOINT_VELOCITY}")

# ---------------- 3D plot ----------------
fig = plt.figure(figsize=(14, 5))

# Left: 3D tip path
ax3d = fig.add_subplot(1, 2, 1, projection='3d')

# Tip path
ax3d.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2],
          'k-', lw=1.5, label='tip path')

# Corners
colors = ['blue', 'orange', 'red', 'green']
labels = ['P0', 'P1', 'P2 (hit)', 'P3']
for i in range(4):
    ax3d.scatter(tips_clamped[i,0], tips_clamped[i,1], tips_clamped[i,2],
                 color=colors[i], s=80, label=labels[i])

# Ball
ax3d.scatter(ball_xyz[0], ball_xyz[1], ball_xyz[2],
             color='gold', s=250, edgecolor='k', label='ball')

# Base frame origin
ax3d.scatter(0, 0, 0, color='k', marker='^', s=100, label='robot base')

# Workspace box (Lite 6 envelope: |x|<=0.44, |y|<=0.44, z in [-0.165, 0.684])
x_lim = [-0.44, 0.44]
y_lim = [-0.44, 0.44]
z_lim = [-0.165, 0.684]

for z in z_lim:
    ax3d.plot([x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]],
              [y_lim[0], y_lim[0], y_lim[1], y_lim[1], y_lim[0]],
              [z, z, z, z, z], color='gray', lw=0.6, alpha=0.6)
for x in x_lim:
    for y in y_lim:
        ax3d.plot([x, x], [y, y], [z_lim[0], z_lim[1]],
                  color='gray', lw=0.6, alpha=0.6)

ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('z [m]')
ax3d.set_title('3D tip path')
ax3d.legend(fontsize=8, loc='upper left')

# Force equal aspect ratio (otherwise the box looks stretched)
P_all = np.vstack([path_xyz, tips_clamped, ball_xyz.reshape(1,3), np.zeros((1,3))])
mid = P_all.mean(axis=0)
span = 0.5 * np.max(P_all.max(axis=0) - P_all.min(axis=0))
ax3d.set_xlim(mid[0] - span, mid[0] + span)
ax3d.set_ylim(mid[1] - span, mid[1] + span)
ax3d.set_zlim(mid[2] - span, mid[2] + span)

# Right: joint angles (unchanged)
ax2d = fig.add_subplot(1, 2, 2)
for j in range(6):
    ax2d.plot(tq, np.degrees(joint_trajectory[:,j]), label=f'θ{j+1}')
ax2d.set_xlabel('t [s]')
ax2d.set_ylabel('deg')
ax2d.legend()
ax2d.set_title('Joint trajectory')

plt.tight_layout()
plt.show()




