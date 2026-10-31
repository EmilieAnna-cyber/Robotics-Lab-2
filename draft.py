# useless code

'''_________________________________________Create a swing trajectory_________________________________________________________________'''

# this is an example

# Target Ball Definition

'''ball_xyz = np.array([0.5, 0.0, 0.1])


# timepoints in seconds
time_points = np.array([0.0, 1.0, 1.5, 2.5]) 
x_points = np.array([0.5,  0.3,  ball_xyz[0],  0.2]) 
y_points = np.array([0.0, -0.4,  ball_xyz[1],  0.5])
z_points = np.array([0.1,  0.5,  ball_xyz[2],  0.6])

# with cubic spline, connect the (x,y,z) coordinates to each time point to create a smooth trajectory
cs_x = CubicSpline(time_points, x_points, bc_type='clamped')
cs_y = CubicSpline(time_points, y_points, bc_type='clamped')
cs_z = CubicSpline(time_points, z_points, bc_type='clamped')

# break down 2.5 second trajectory in a 100 little timesteps
t_fine = np.linspace(0, 2.5, 100)

# 100x3 matrix containing x,y,z positions over these 100 timesteps
path_xyz = np.vstack((cs_x(t_fine), cs_y(t_fine), cs_z(t_fine))).T


# solve inverse kinematics iteratively 
joint_trajectory = []
q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Starting position guess

for target in path_xyz:
    q_sol = solve_ik(target, q_current)
    if q_sol is None:
        raise ValueError("Trajectory point out of reachable space! Constraint violated.")
    
    joint_trajectory.append(q_sol)
    q_current = q_sol # Seed the next step with the current solution to preserve continuity

joint_trajectory = np.array(joint_trajectory)




'''_________________________________________functions for swing trajectory________________________________________________________________'''

def club_head_to_gripper_pos(club_head_pos):
    return club_head_pos + np.array([0.0, 0.0, L_CLUB_VAL])


def P1(ball_pos, back_swing_angle):
    ''' calculate the backswing condition '''
    P0 = club_head_to_gripper_pos(ball_pos)

    return 

# create a function for any arbitraty position
def make_swing_trajectory(ball_xyz, current_gripper_xyz, L_club,
                          offset=0.05, backswing_angle=np.radians(30),
                          after_impact_angle=np.radians(35),
                          V01=0.04, V12=0.06, V_MAX=0.35,
                          n_samples=200):
    """
    Build a swing trajectory that places the CLUB HEAD (not the gripper)
    on a circle around `ball_xyz`.

    Assumption: the club hangs vertically beneath the gripper,
    so club_head = gripper + [0, 0, -L_club].
    Therefore the IK target (gripper position) is club_head + [0, 0, +L_club].

    Parameters
    ----------
    ball_xyz           : (3,) ball position in world frame (m)
    current_gripper_xyz: (3,) current gripper position (m)
    L_club             : scalar club length (m)
    offset             : safety/setup distance (m)
                            => how far the club head starts in front of the ball before swinging through it
    backswing_angle    : rotation of swing direction for backswing (rad)
    after_impact_angle: rotation past the ball (rad)
    V01, V12, V_MAX    : segment speeds (m/s)
    n_samples          : # of trajectory samples

    Returns
    -------
    t_fine    : (N,) time samples
    path_xyz  : (N,3) gripper positions (IK targets)
    ball_xyz  : (3,)   ball position (for plotting)
    """
    # convert whatever position type (list,tuple,...) to a NumPy float array
    ball_xyz = np.asarray(ball_xyz, dtype=float)
    current_gripper_xyz = np.asarray(current_gripper_xyz, dtype=float)


    # since the end of the club is length L_club below the gripper, this is the gripper target
    gripper_target = ball_xyz + [0, 0, +L_club]

    # --- 0. Corner case!!!! Reachability: the GRIPPER target is ball + L above the ball ---
    # (check if the gripper target must be within workspace radius)
    # the origin is (0,0,0) 0.9 m is far beyond what the robot can reach # REDO NUMBERS LATER
    if np.linalg.norm(gripper_target) > 0.9:
        raise ValueError("Ball is impossible to reach")

    # --- 1. Swing direction (unit) ---

    # displacement from ball to gripper
    d = current_gripper_xyz - ball_xyz

    # normalised vector of the direction the gripper goes
    swing_dir = d / np.linalg.norm(d)

    # --- 2. Key CLUB-HEAD positions on a circle around the ball ---
    # club_head = ball + r * u_hat  (r = offset, except at contact r=0)
    # gripper   = club_head + [0,0,+L_club]

    # world vertical, used to build the rotation axis.
    up = np.array([0.0, 0.0, 1.0])

    # position of where the gripper should be
    def club_head_to_gripper(club_head):
        return club_head + np.array([0.0, 0.0, L_club])

    # P1: approach — club head offset in front of ball
    P1_club = ball_xyz + offset * swing_dir
    P1      = club_head_to_gripper(P1_club)

    # P2: backswing — rotate swing_dir about horizontal axis# P2: backswing — rotate swing_dir about horizontal axis

    # cross product (swing direction)xr(rotation axis) = axis around which the club will rotate for the bsckswing
    axis = np.cross(swing_dir, up)
    axis = axis / np.linalg.norm(axis) # normalise axis

    # direction for the backswing = swing direction rotated around the axis by a certain angle (backswing angle)

    back_dir = _rotate(swing_dir, axis, backswing_angle)
    P2_club = ball_xyz + offset * back_dir
    P2      = club_head_to_gripper(P2_club)

    # P3: contact — club head exactly on ball
    P3_club = ball_xyz.copy()
    P3      = club_head_to_gripper(P3_club)

    # P4: the club won't stop at the ball: it will keep swinging
    forward_dir = _rotate(swing_dir, axis, -after_impact_angle)
    P4_club = ball_xyz + offset * forward_dir
    P4      = club_head_to_gripper(P4_club)

    # --- 3. get the timesteps ---
    pts    = np.array([current_gripper_xyz, P1, P2, P3, P4])
    velocities = np.array([V01, V12, V_MAX, V_MAX])

    time_pts = [0.0]
    for i in range(1, len(pts)):
        dist = np.linalg.norm(pts[i] - pts[i-1])
        dt = dist / velocities[i-1]
        time_pts.append(time_pts[-1] + dt)
    time_pts = np.array(time_pts)

    # --- 4. Cubic splines (clamped) ---
    x_points = np.array([current_gripper_xyz[0],P1[0], P2[0], P3[0], P4[0]]) 
    y_points = np.array([current_gripper_xyz[1],P1[1], P2[1], P3[1], P4[1]])
    z_points = np.array([current_gripper_xyz[2],P1[2], P2[2], P3[2], P4[2]])

# with cubic spline, connect the (x,y,z) coordinates to each time point to create a smooth trajectory
    cs_x = CubicSpline(time_points, x_points, bc_type='clamped')
    cs_y = CubicSpline(time_points, y_points, bc_type='clamped')
    cs_z = CubicSpline(time_points, z_points, bc_type='clamped')

    # break down trajectory into n_sample timesteps
    t_fine   = np.linspace(time_pts[0], time_pts[-1], n_samples)

    # n_samples x3 matrix containing x,y,z positions over these 100 timesteps

    path_xyz = np.vstack((cs_x(t_fine), cs_y(t_fine), cs_z(t_fine))).T


    return t_fine, path_xyz, ball_xyz, time_pts, pts


def _rotate(v, axis, theta):
    """Rodrigues rotation: rotate v about axis by theta."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(theta)
            + np.cross(axis, v) * np.sin(theta)
            + axis * np.dot(axis, v) * (1 - np.cos(theta)))


'''_________________________________________Calculate velocities_________________________________________________________________'''

swing = make_swing_trajectory(ball_xyz,get_ee_position(np.array[np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]), 1)
t_fine = swing[0]
path_xyz = swing[1]
ball_xyz = swing[2]
time_pts = swing[3]
pts = swing[4]

# --- FIX 1: Calculate the time delta (dt) ---
dt = t_fine[1] - t_fine[0]

# Compute joint velocities via numerical differentiation (dq/dt)
joint_velocities = np.diff(joint_trajectory, axis=0) / dt

# --- Plot 1: 3D Workspace Performance ---
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], label='Club Tip Path', color='b', lw=2)
ax1.scatter(*ball_xyz, color='r', s=100, label='Fixed Ball Target')
ax1.set_title("3D Swing Execution Vector")
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.legend()

# --- Plot 2: Velocity Bounds Verification ---
ax2 = fig.add_subplot(122)
for joint_idx in range(6):
    ax2.plot(t_fine[:-1], np.abs(joint_velocities[:, joint_idx]), label=f'Joint {joint_idx+1}')

ax2.axhline(y=MAX_JOINT_VELOCITY, color='r', linestyle='--', label='Specs Ceiling Limit')
ax2.set_title("Velocity Limits Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Speed (rad/s)")
ax2.legend()

plt.tight_layout()
plt.show()'''
