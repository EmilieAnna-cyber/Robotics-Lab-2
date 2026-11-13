# TASK 2 - FIGURES FOR THE REPORT

# report_figures.py
# makes all the figures of Task 2 in the report (same file names as in the latex)
# run from inside useful/ :  python report_figures.py     or from Robotics-Lab-2 :  python -m useful.report_figures
# the pictures are saved in a folder "figures" next to useful/
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # main_swing calls plt.show(), this stops it from blocking
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
from scipy.interpolate import CubicHermiteSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "import useful.xxx" works
import useful.robot_math as rm
import useful.main_swing as ms  # runs the whole swing (waypoints, timing, IK, velocity clamp)

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(folder, exist_ok=True)

# values we use in almost every figure
ball = ms.ball_xyz
pivot = rm.SWING_PIVOT
waypoints = ms.tips_clamped
colors = ['C0', 'C1', 'C2', 'C3']  # P0, P1, P2, P3 (same colours as the plots in main_swing)

# index of the sample closest to each waypoint time (P0, P1, P2, P3)
wp_index = []
for t in ms.t_corners:
    wp_index.append(np.argmin(np.abs(ms.tq - t)))
impact = wp_index[2]


# positions of the joints for drawing the arm (same DH table as kinematics_2.py)
def dh_matrix(a, alpha, theta, d):
    '''
    goal: numeric transformation matrix of one link (same formula as make_transformation_matrix in kinematics_2.py)
    PARAMS:
        a, alpha, theta, d: DH parameters of the link (m, rad)
    RETURNS:
        4x4 numpy array
    '''
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]])

def arm_points(q):
    '''
    goal: positions of the joints of the arm, to draw it
    PARAMS:
        q: list of the 6 joint angles (radians)
    RETURNS:
        points: 6x3 array [base, shoulder, elbow, end of a3, wrist centre, flange] (m)
        R: 3x3 orientation of the flange
    '''
    T01 = dh_matrix(0,     -np.pi/2, q[0],            0.2433)
    T12 = dh_matrix(0.2,    np.pi,   q[1] - np.pi/2,  0)
    T23 = dh_matrix(0.087,  np.pi/2, q[2] - np.pi/2,  0)
    T34 = dh_matrix(0,      np.pi/2, q[3],            0.2276)
    T45 = dh_matrix(0,     -np.pi/2, q[4],            0)
    T56 = dh_matrix(0,      0,       q[5],            0.0615)
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T06 = T04 @ T45 @ T56
    base = np.array([0, 0, 0])
    shoulder = T01[:3, 3]
    elbow = T02[:3, 3]
    elbow_offset = T03[:3, 3]
    wrist = T04[:3, 3]
    flange = T06[:3, 3]
    points = np.array([base, shoulder, elbow, elbow_offset, wrist, flange])
    return points, T06[:3, :3]   # joint positions + orientation of the flange


# =============================================================================
# 1. PIPELINE - overview of the method
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 7.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def box(x, y, width, title, subtitle, face, edge):
    '''
    goal: draw one rounded box of the pipeline figure
    PARAMS:
        x, y: centre of the box
        width: width of the box
        title, subtitle: text in the box (subtitle can be "")
        face, edge: fill colour and border colour
    '''
    ax.add_patch(FancyBboxPatch((x - width/2, y - 0.38), width, 0.76,
                                boxstyle="round,pad=0.02,rounding_size=0.25", fc=face, ec=edge, lw=1.5))
    if subtitle == "":
        ax.text(x, y, title, ha='center', va='center', fontsize=11.5, fontweight='bold')
    else:
        ax.text(x, y + 0.13, title, ha='center', va='center', fontsize=11.5, fontweight='bold')
        ax.text(x, y - 0.17, subtitle, ha='center', va='center', fontsize=9.5)

def arrow(x0, y0, x1, y1, color='#444444', style='-'):
    '''
    goal: draw an arrow from (x0, y0) to (x1, y1) in the pipeline figure
    PARAMS:
        x0, y0: start of the arrow
        x1, y1: end of the arrow (head)
        color: colour of the arrow
        style: line style ('-' full, '--' dashed)
    '''
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4, ls=style, mutation_scale=15))

x = 3.6      # centre of the main column
w = 5.6      # width of the boxes
y = [9.3, 8.1, 6.9, 5.7, 4.5, 3.3, 2.1, 0.9]   # height of each box
box(x, y[0], w, 'Ball  B', 'chosen position', '#FFF3C4', '#B8860B')
box(x, y[1], w, r'2.3  Waypoints  $P_0 \dots P_3$', 'offsets from B, then clamping', '#E1F5EE', '#0F6E56')
box(x, y[2], w, r'2.4  Times $t_k$ and velocities $v_k$', r'rest at $P_0, P_1, P_3$ ; fastest at $P_2$', '#FAEEDA', '#854F0B')
box(x, y[3], w, '2.4  Hermite spline', 'club-tip position  p(t)', '#FAEEDA', '#854F0B')
box(x, y[4], w, '2.1  Look-at matrix', 'club orientation  R(t)', '#E6F1FB', '#185FA5')
box(x, y[5], w, '2.2  Inverse kinematics', r'joint angles  $q(t)=(\theta_1,\dots,\theta_6)$', '#EEEDFE', '#534AB7')
box(x, y[6], w, r'2.5  Is a joint faster than $\pi$ rad/s ?', '', '#FCEBEB', '#A32D2D')
box(x, y[7], w, 'Final trajectory  q(t)', 'results (2.6), simulation, robot', '#EAF3DE', '#3B6D11')
box(8.3, y[1], 3.0, '2.1  Swing pivot  c', r'from B, r and $\lambda$', '#E6F1FB', '#185FA5')

for i in range(6):
    arrow(x, y[i] - 0.4, x, y[i+1] + 0.4)
arrow(x, y[6] - 0.4, x, y[7] + 0.4, color='#3B6D11')
ax.text(x - 0.25, 1.5, 'no', color='#3B6D11', fontsize=10, ha='right', va='center')

# ball -> pivot -> look-at matrix
ax.plot([x + 1.2, 8.3], [y[0] - 0.4, y[0] - 0.4], color='#444444', lw=1.4)
arrow(8.3, y[0] - 0.4, 8.3, y[1] + 0.4)
ax.plot([8.3, 8.3], [y[1] - 0.4, y[4]], color='#444444', lw=1.4, ls='--')
arrow(8.3, y[4], x + w/2 + 0.05, y[4], style='--')

# "yes" -> slow down and go back to the timing
ax.plot([x + w/2, 7.0], [y[6], y[6]], color='#A32D2D', lw=1.4, ls='--')
ax.plot([7.0, 7.0], [y[6], y[2]], color='#A32D2D', lw=1.4, ls='--')
arrow(7.0, y[2], x + w/2 + 0.05, y[2], color='#A32D2D', style='--')
ax.text(7.15, y[6] - 0.05, 'yes: slow everything\ndown,  ' + r'$t \leftarrow \alpha\, t$', color='#A32D2D', fontsize=10, va='center')

fig.tight_layout()
fig.savefig(os.path.join(folder, "Pipeline.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved Pipeline.png")


# =============================================================================
# 2. CLAMPING - the two clamping stages on two test points
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6))
reach_allowed = rm.MAX_REACH * rm.MAX_REACH_SAFETY   # 0.4215 m

# --- left: stage 1 (radial shrink), side view ---
p = np.array([0.45, 0.10, 0.35])     # a point that is too far away
rho = np.sqrt(p[0]**2 + p[1]**2)     # horizontal distance from the base
ax1.set_aspect('equal'); ax1.grid(alpha=.3); ax1.axhline(0, color='k', lw=.8)

# circles of the allowed reach and the max reach, around the shoulder
angle = np.linspace(-np.pi/2, np.pi/2, 200)
ax1.plot(reach_allowed*np.cos(angle), 0.2433 + reach_allowed*np.sin(angle), '--', color='C0', lw=1.6,
         label=r'allowed reach: $0.95\,(a_2+R)=0.4215$ m from shoulder')
ax1.plot(rm.MAX_REACH*np.cos(angle), 0.2433 + rm.MAX_REACH*np.sin(angle), ':', color='0.5', lw=1.2,
         label=r'max reach $a_2+R=0.4437$ m')
ax1.plot(0, 0.2433, 'o', ms=10, mfc='white', mec='k', mew=2)
ax1.text(-0.012, 0.2583, 'shoulder\n$(0,\\,d_1)$', ha='right', fontsize=10)
ax1.plot(0, 0, 'k^', ms=10)
ax1.text(0.01, -0.03, 'base origin', fontsize=10)
ax1.plot([0, rho], [0, p[2]], '-', color='0.6', lw=1)

# try s = 1.00, 0.98, ... like clamp_waypoint does
for s in rm.RADIUS_STEPS:
    p_new = p * s
    h = np.sqrt(p_new[0]**2 + p_new[1]**2)
    if rm._tip_is_feasible(p_new):
        ax1.plot(h, p_new[2], 'o', color='C2', ms=12, mec='k', zorder=5)   # first one that works
        ax1.annotate('first feasible: $s=%.2f$\n' % s + r'$\mathbf{p}_{new}=%.2f\,\mathbf{p}=(%.3f,\ %.3f,\ %.3f)$' % (s, p_new[0], p_new[1], p_new[2]),
                     (h, p_new[2]), (h - 0.33, p_new[2] + 0.13), fontsize=10.5, color='C2',
                     arrowprops=dict(arrowstyle='->', color='C2'))
        break
    ax1.plot(h, p_new[2], 'x', color='C3', ms=9, mew=2)   # rejected
ax1.text(rho + 0.012, p[2] + 0.008, r'$\mathbf{p}=(0.45,\ 0.10,\ 0.35)$' + '\n' + r'$s=1.00$: too far (0.473 m)', color='C3', fontsize=10.5)
ax1.text(0.475, 0.265, r'$s=0.98,\,0.96,\dots$' + '\nrejected', color='C3', fontsize=10)
ax1.set_xlabel(r'horizontal distance from base $\rho=\sqrt{x^2+y^2}$ [m]'); ax1.set_ylabel('z [m]')
ax1.set_xlim(-0.12, 0.62); ax1.set_ylim(-0.05, 0.72)
ax1.legend(loc='upper left', fontsize=9.5)
ax1.set_title('(a) Stage 1, radial shrink: $\\mathbf{p}_{new}=s\\,\\mathbf{p}$, $s=1.00,\\,0.98,\\dots,0.50$\n(moves along the line to the base origin)', fontsize=11.5)

# --- right: stage 2 (azimuth rotation), top view ---
p = np.array([-0.092, -0.022, 0.283])   # a point too close to the robot
rho = np.sqrt(p[0]**2 + p[1]**2)
phi = np.arctan2(p[1], p[0])
ax2.set_aspect('equal'); ax2.grid(alpha=.3)
circle = np.linspace(0, 2*np.pi, 200)
ax2.plot(rho*np.cos(circle), rho*np.sin(circle), '--', color='0.6', lw=1.2, label=r'circle of radius $\rho$ (height $z$ kept)')
ax2.plot(0, 0, 'k^', ms=10); ax2.text(0.004, -0.012, 'base', fontsize=10)
ax2.plot(pivot[0], pivot[1], 'kx', ms=9, mew=2); ax2.text(pivot[0] + 0.004, pivot[1] + 0.005, r'pivot $\mathbf{c}$', fontsize=10)
ax2.plot(ball[0], ball[1], 'o', color='gold', ms=13, mec='k'); ax2.text(ball[0] + 0.005, ball[1] + 0.008, 'ball', fontsize=10)
ax2.plot([0, p[0]], [0, p[1]], '-', color='0.6', lw=1)
ax2.plot(p[0], p[1], 'x', color='C3', ms=11, mew=2.5)
ax2.annotate(r'$\mathbf{p}=(-0.092,\ -0.022,\ 0.283)$' + '\nstage 1 fails: too close to the shoulder\n(wrist centre inside $|a_2-R|$)',
             (p[0], p[1]), (0.105, 0.06), color='C3', fontsize=10, arrowprops=dict(arrowstyle='->', color='C3', lw=0.8))

# try +dphi then -dphi, with dphi = 3.75, 7.5, ... like clamp_waypoint does
found = False
for dphi in rm.ANGLE_STEPS[1:]:
    for sign in [1, -1]:
        p_new = np.array([rho*np.cos(phi + sign*dphi), rho*np.sin(phi + sign*dphi), p[2]])
        if rm._tip_is_feasible(p_new):
            ax2.plot(p_new[0], p_new[1], 'o', color='C2', ms=12, mec='k', zorder=5)   # first one that works
            ax2.plot([0, p_new[0]], [0, p_new[1]], '-', color='C2', lw=1)
            ax2.annotate(r'first feasible: $\Delta\phi=+%.2f^\circ$' % np.degrees(dphi) + '\n' + r'$\mathbf{p}_{new}=(%.3f,\ %.3f,\ %.3f)$' % (p_new[0], p_new[1], p_new[2]),
                         (p_new[0], p_new[1]), (p_new[0] + 0.05, p_new[1] - 0.03), fontsize=10.5, color='C2',
                         arrowprops=dict(arrowstyle='->', color='C2'))
            found = True
            break
        ax2.plot(p_new[0], p_new[1], '.', color='C3', ms=7)   # rejected
    if found:
        break
ax2.add_patch(Arc((0, 0), 0.06, 0.06, theta1=np.degrees(phi), theta2=np.degrees(phi + dphi), color='C1', lw=1.8))
ax2.text(-0.045, -0.045, r'$+\Delta\phi$', color='C1', fontsize=12)
ax2.text(0.105, -0.045, r'tried in order: $\pm3.75^\circ,\ \pm7.5^\circ,\ \dots$' + '\n(red dots = rejected)', fontsize=10, color='C3')
ax2.set_xlim(-0.14, 0.32); ax2.set_ylim(-0.145, 0.12)
ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.legend(loc='upper right', fontsize=9.5)
ax2.set_title('(b) Stage 2, azimuth rotation (top view): $\\phi\\to\\phi\\pm\\Delta\\phi$, $\\Delta\\phi\\in[0,\\,90^\\circ]$\n(keeps the height $z$ and the radius $\\rho$)', fontsize=11.5)

fig.tight_layout()
fig.savefig(os.path.join(folder, "Clamping.png"), dpi=180, bbox_inches='tight')
plt.close(fig)
print("saved Clamping.png")


# =============================================================================
# 3. WAYPOINT CONSTRUCTION - each waypoint built from the ball, one arrow per term
# =============================================================================
s_dir = np.array([0, 1, 0])    # hitting direction
i_dir = np.array([-1, 0, 0])   # towards the robot
u_dir = np.array([0, 0, 1])    # up

# the terms of each formula: (label, vector, colour)  - same offsets as find_P0 ... find_P3
terms_P0 = [(r'-L_0\,\hat{\mathbf{s}}', -0.05*s_dir, 'C0'), (r'+h_0\,\hat{\mathbf{u}}', 0.01*u_dir, 'C2')]
terms_P1 = [(r'-L_b\,\hat{\mathbf{s}}', -0.15*s_dir, 'C0'), (r'+i_1\,\hat{\mathbf{\imath}}', 0.12*i_dir, 'C1'), (r'+h_1\,\hat{\mathbf{u}}', 0.10*u_dir, 'C2')]
terms_P2 = []
terms_P3 = [(r'+L_f\,\hat{\mathbf{s}}', 0.15*s_dir, 'C0'), (r'+i_3\,\hat{\mathbf{\imath}}', 0.12*i_dir, 'C1'), (r'+h_2\,\hat{\mathbf{u}}', 0.10*u_dir, 'C2')]
all_terms = [terms_P0, terms_P1, terms_P2, terms_P3]
titles = [r'$\mathbf{P}_0$: address   ($L_0=0.05,\ h_0=0.01$ m)' + '\n(zoomed in)',
          r'$\mathbf{P}_1$: top of backswing   ($L_b=0.15,\ i_1=0.12,\ h_1=0.10$ m)',
          r'$\mathbf{P}_2$: impact = ball   (no offset)',
          r'$\mathbf{P}_3$: follow-through   ($L_f=0.15,\ i_3=0.12,\ h_2=0.10$ m)']

fig = plt.figure(figsize=(15, 12))
for k in range(4):
    ax = fig.add_subplot(2, 2, k + 1, projection='3d')
    if k != 0:
        # the other waypoints (faded) and the robot base
        for j in range(4):
            if j != k:
                ax.scatter(*waypoints[j], s=30, color=colors[j], alpha=.25, depthshade=False)
        ax.scatter(0, 0, 0, marker='^', s=70, color='k')
        ax.text(0, 0, -0.03, 'robot base', fontsize=9, ha='center')
    ax.scatter(*ball, s=170, color='gold', edgecolor='k', depthshade=False)
    if k == 0:
        ax.text(ball[0] + 0.006, ball[1] + 0.004, ball[2] - 0.012, r'$\mathbf{B}$', fontsize=13)
    else:
        ax.text(ball[0] + 0.012, ball[1] + 0.01, ball[2] - 0.03, r'$\mathbf{B}$', fontsize=13)

    # draw the formula: start at the ball and add one arrow per term
    point = ball.copy()
    for label, vector, col in all_terms[k]:
        if np.linalg.norm(vector) > 0.03:
            head = 0.18
        else:
            head = 0.4   # small arrow -> bigger arrow head
        ax.quiver(*point, *vector, color=col, lw=2.6, arrow_length_ratio=head)
        middle = point + vector / 2
        if k == 0:
            if col == 'C0': offset = np.array([0.004, 0, -0.012])
            else:           offset = np.array([0.004, 0.012, -0.004])
        else:
            if col == 'C0':   offset = np.array([0.03, 0, -0.035])
            elif col == 'C1': offset = np.array([0, 0, 0.015])
            else:             offset = np.array([0.012, 0, 0])
        ax.text(*(middle + offset), '$' + label + '$', color=col, fontsize=12, fontweight='bold')
        point = point + vector

    # the real waypoint from robot_math (it must be where the last arrow ends)
    P = waypoints[k]
    ax.scatter(*P, s=110, color=colors[k], edgecolor='k', depthshade=False)
    text = r'$\mathbf{P}_%d$ = (%.2f, %.2f, %.2f)' % (k, P[0], P[1], P[2])
    if k == 0:
        ax.text(P[0] - 0.03, P[1] - 0.03, P[2] + 0.012, text, fontsize=11, color=colors[k], fontweight='bold')
        ax.set_xlim(0.2, 0.3); ax.set_ylim(-0.08, 0.03); ax.set_zlim(0.13, 0.19)
        ax.set_box_aspect((0.1, 0.11, 0.06))
    else:
        ax.text(P[0], P[1] + 0.01, P[2] + 0.025, text, fontsize=11, color=colors[k], fontweight='bold')
        ax.set_xlim(0, 0.3); ax.set_ylim(-0.2, 0.2); ax.set_zlim(0, 0.3)
        ax.set_box_aspect((0.3, 0.4, 0.3))
    ax.view_init(22, -58)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title(titles[k], fontsize=12)

# legend at the bottom
fig.legend(handles=[plt.Line2D([], [], color='C0', lw=3, label=r'$\hat{\mathbf{s}}=(0,1,0)$  hitting direction'),
                    plt.Line2D([], [], color='C1', lw=3, label=r'$\hat{\mathbf{\imath}}=(-1,0,0)$  towards robot'),
                    plt.Line2D([], [], color='C2', lw=3, label=r'$\hat{\mathbf{u}}=(0,0,1)$  up'),
                    plt.Line2D([], [], marker='o', ls='', color='gold', mec='k', ms=11, label=r'ball $\mathbf{B}=(0.25,0,0.15)$')],
           loc='lower center', ncol=4, fontsize=11, frameon=True)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(folder, "Waypoint_construction.png"), dpi=180, bbox_inches='tight')
plt.close(fig)
print("saved Waypoint_construction.png")


# =============================================================================
# 4. VISUALISE PIVOT - the whole arm + how the pivot c is built
# =============================================================================
q = ms.joint_trajectory
arm, R_flange = arm_points(q[impact])   # arm at impact

fig = plt.figure(figsize=(16, 7.8))

# --- left: 3D, arm at the four waypoints ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
arm_colors = ['C0', 'C1', '0.15', 'C3']
arm_names = ['address $P_0$', 'top of backswing $P_1$', 'impact $P_2$', 'finish $P_3$']
for k in range(4):
    points, _ = arm_points(q[wp_index[k]])
    if k == 2:   # impact: thick and black
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], '-', color=arm_colors[k], lw=5, solid_capstyle='round', label='arm at ' + arm_names[k])
        ax1.scatter(points[1:5, 0], points[1:5, 1], points[1:5, 2], s=45, color='white', edgecolor=arm_colors[k], depthshade=False)
    else:
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], '-', color=arm_colors[k], lw=2.2, alpha=.6, label='arm at ' + arm_names[k])
        ax1.scatter(points[1:5, 0], points[1:5, 1], points[1:5, 2], s=14, color=arm_colors[k], depthshade=False)
ax1.plot(ms.path_xyz[:, 0], ms.path_xyz[:, 1], ms.path_xyz[:, 2], color='C2', lw=1.6, label='club-tip path')
ax1.scatter(*ball, s=160, color='gold', edgecolor='k', label='ball', depthshade=False)
ax1.scatter(*pivot, marker='x', s=80, color='k', label=r'pivot $\mathbf{c}$', depthshade=False)
ax1.plot([pivot[0], arm[5, 0]], [pivot[1], arm[5, 1]], [pivot[2], arm[5, 2]], ':', color='k', lw=1)
corner = np.array([pivot[0], 0, ball[2]])   # ball -> towards robot -> up = pivot
ax1.plot([ball[0], corner[0]], [ball[1], corner[1]], [ball[2], corner[2]], '-', color='C1', lw=2, label=r'$r\sin\lambda$ towards robot')
ax1.plot([corner[0], pivot[0]], [corner[1], pivot[1]], [corner[2], pivot[2]], '-', color='C2', lw=2, label=r'$r\cos\lambda$ up')
circle = np.linspace(0, 2*np.pi, 60)
ax1.plot(0.06*np.cos(circle), 0.06*np.sin(circle), 0*circle, color='0.6', lw=1)   # base
# base frame x0 y0 z0
ax1.quiver(0, 0, 0, 0.1, 0, 0, color='r', lw=2, arrow_length_ratio=.2); ax1.text(0.115, 0, 0, '$x_0$', color='r', fontsize=11)
ax1.quiver(0, 0, 0, 0, 0.1, 0, color='g', lw=2, arrow_length_ratio=.2); ax1.text(0, 0.115, 0, '$y_0$', color='g', fontsize=11)
ax1.quiver(0, 0, 0, 0, 0, 0.1, color='b', lw=2, arrow_length_ratio=.2); ax1.text(0, 0, 0.115, '$z_0$', color='b', fontsize=11)
# tool frame at the flange (columns of R_flange = x, y, z of the tool)
ax1.quiver(*arm[5], *(0.06*R_flange[:, 0]), color='r', lw=2, arrow_length_ratio=.25)
ax1.quiver(*arm[5], *(0.06*R_flange[:, 1]), color='g', lw=2, arrow_length_ratio=.25)
ax1.quiver(*arm[5], *(0.06*R_flange[:, 2]), color='b', lw=2, arrow_length_ratio=.25)
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_zlabel('z [m]')
ax1.set_xlim(-0.1, 0.35); ax1.set_ylim(-0.22, 0.22); ax1.set_zlim(0, 0.45)
ax1.set_box_aspect((0.45, 0.44, 0.45)); ax1.view_init(18, -62)
ax1.legend(fontsize=8.5, loc='upper left')
ax1.set_title('(a) Whole arm at the four waypoints (impact in black)\nbase frame $x_0y_0z_0$ and tool frame at the flange', fontsize=11.5)

# --- right: side view at impact (theta1 = 0, so the arm is in the plane y = 0 -> use x and z) ---
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_aspect('equal'); ax2.grid(alpha=.3); ax2.axhline(0, color='k', lw=.8)
ax2.add_patch(plt.Rectangle((-0.04, 0), 0.08, 0.025, color='0.85'))
xz = arm[:, [0, 2]]
ax2.plot(xz[:, 0], xz[:, 1], '-', color='0.15', lw=6, solid_capstyle='round', zorder=3)
ax2.plot(xz[1:5, 0], xz[1:5, 1], 'o', ms=11, mfc='white', mec='0.15', mew=2, zorder=4)
# link lengths, written next to the middle of each link
middle = (xz[:-1] + xz[1:]) / 2
ax2.text(middle[0, 0] - 0.055, middle[0, 1], r'$d_1=0.2433$', fontsize=11, ha='center', va='center')
ax2.text(middle[1, 0] - 0.055, middle[1, 1], r'$a_2=0.200$', fontsize=11, ha='center', va='center')
ax2.text(middle[2, 0], middle[2, 1] + 0.022, r'$a_3=0.087$', fontsize=11, ha='center', va='center')
ax2.text(middle[3, 0] + 0.06, middle[3, 1] + 0.02, r'$d_4=0.2276$', fontsize=11, ha='center', va='center')
ax2.annotate(r'$d_6=0.0615$', xy=middle[4], xytext=(0.09, 0.095), fontsize=11, arrowprops=dict(arrowstyle='-', lw=0.8, color='0.4'))
ax2.text(xz[1, 0] - 0.105, xz[1, 1] + 0.012, 'J2 shoulder', fontsize=10)
ax2.text(xz[2, 0] - 0.075, xz[2, 1] + 0.008, 'J3 elbow', fontsize=10)
ax2.text(0.21, 0.222, 'wrist centre\n(J4, J5, J6)', fontsize=10)
ax2.text(0.045, 0.006, 'J1 base', fontsize=10)
# ball, pivot and the line between them
ax2.plot(ball[0], ball[2], 'o', color='gold', ms=15, mec='k', zorder=5)
ax2.plot(pivot[0], pivot[2], 'kx', ms=10, mew=2, zorder=5)
ax2.text(pivot[0] - 0.005, pivot[2] + 0.014, r'pivot $\mathbf{c}=(0.120,0,0.225)$', fontsize=10.5, ha='center')
ax2.plot([pivot[0], ball[0]], [pivot[2], ball[2]], ':', color='k', lw=1)
# z and y of the tool at the flange
flange = xz[5]
z_tool = 0.07 * R_flange[[0, 2], 2]
y_tool = 0.07 * R_flange[[0, 2], 1]
ax2.annotate('', xy=flange + z_tool, xytext=flange, arrowprops=dict(arrowstyle='-|>', lw=2.4, color='b', mutation_scale=16))
ax2.text(flange[0] + z_tool[0] + 0.006, flange[1] + z_tool[1] - 0.012, r'$\mathbf{z}_{tool}$', color='b', fontsize=12)
ax2.annotate('', xy=flange + y_tool, xytext=flange, arrowprops=dict(arrowstyle='-|>', lw=2.4, color='g', mutation_scale=16))
ax2.text(flange[0] + y_tool[0] + 0.006, flange[1] + y_tool[1], r'$\mathbf{y}_{tool}$', color='g', fontsize=12)
ax2.text(0.255, 0.04, 'ball = flange = club tip\n($L_{club}=0$)', fontsize=10)
# how c is built: from the ball, r*sin(lambda) towards the robot, then r*cos(lambda) up
corner = np.array([pivot[0], ball[2]])
ax2.annotate('', xy=corner, xytext=(ball[0], ball[2]), arrowprops=dict(arrowstyle='-|>', lw=2.2, color='C1', mutation_scale=14))
ax2.annotate('', xy=(pivot[0], pivot[2]), xytext=corner, arrowprops=dict(arrowstyle='-|>', lw=2.2, color='C2', mutation_scale=14))
ax2.add_patch(plt.Rectangle(corner, 0.007, 0.007, fill=False, lw=.8))
ax2.text(0.168, 0.157, r'$r\sin\lambda=0.130$', color='C1', fontsize=10.5, ha='center')
ax2.text(pivot[0] - 0.006, 0.188, r'$r\cos\lambda$' + '\n' + r'$=0.075$', color='C2', fontsize=10.5, ha='right', va='center')
ax2.plot([ball[0], ball[0]], [ball[2], ball[2] + 0.06], ':', color='C3', lw=1.3)   # vertical through the ball
angle_to_pivot = np.degrees(np.arctan2(pivot[2] - ball[2], pivot[0] - ball[0]))
ax2.add_patch(Arc((ball[0], ball[2]), 0.07, 0.07, theta1=90, theta2=angle_to_pivot, color='C3', lw=1.6))
ax2.text(ball[0] - 0.016, ball[2] + 0.047, r'$\lambda$', color='C3', fontsize=12)
formula = (r'$\mathbf{c}=\mathbf{B}+r\,(\sin\lambda\ \hat{\mathbf{\imath}}+\cos\lambda\ \hat{\mathbf{u}})$' + '\n'
           r'$\ \ =(0.25,0,0.15)+(-0.130,0,0)+(0,0,0.075)$' + '\n'
           r'$\ \ =(0.120,\ 0,\ 0.225)$' + '\n'
           r'$r=0.15$ m,  $\lambda=60^\circ$,  $\hat{\mathbf{\imath}}=(-1,0,0)$,  $\hat{\mathbf{u}}=(0,0,1)$')
ax2.text(-0.13, 0.465, formula, fontsize=10.5, bbox=dict(fc='white', ec='0.6'), va='bottom')
ax2.set_xlim(-0.14, 0.45); ax2.set_ylim(-0.01, 0.57)
ax2.set_xlabel('x [m]'); ax2.set_ylabel('z [m]')
ax2.set_title(r'(b) Side view at impact ($\theta_1\approx0$: arm lies in the plane $y=0$)', fontsize=11.5)

fig.tight_layout()
fig.savefig(os.path.join(folder, "Visualise_pivot.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved Visualise_pivot.png")


# =============================================================================
# 5. IK STEP 3 - elbow and shoulder triangle in the arm plane, at impact
# =============================================================================
theta2 = q[impact][1]
theta3 = q[impact][2]
a2, a3, d4 = 0.2, 0.087, 0.2276

# points in the arm plane, measured from the shoulder (same formulas as the FK)
shoulder = np.array([0, 0])
elbow = np.array([a2*np.sin(theta2), a2*np.cos(theta2)])
phi = theta2 - theta3
offset = elbow + np.array([a3*np.cos(phi), -a3*np.sin(phi)])        # end of the a3 piece
wrist = offset + np.array([-d4*np.sin(phi), -d4*np.cos(phi)])      # wrist centre
kx = wrist[0]
kz = wrist[1]

fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_aspect('equal'); ax.axis('off')
ax.annotate('', xy=(0.3, 0), xytext=(-0.05, 0), arrowprops=dict(arrowstyle='-|>', color='0.5'))
ax.annotate('', xy=(0, 0.25), xytext=(0, -0.12), arrowprops=dict(arrowstyle='-|>', color='0.5'))
ax.text(0.302, -0.012, 'horizontal\n(in arm plane)', fontsize=9, color='0.4')
ax.text(0.005, 0.252, 'up', fontsize=9, color='0.4')
# real links
ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], '-', color='0.15', lw=6, solid_capstyle='round')
ax.plot([elbow[0], offset[0], wrist[0]], [elbow[1], offset[1], wrist[1]], '-', color='0.15', lw=6, solid_capstyle='round')
# straight forearm R and distance D
ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], '--', color='C3', lw=2.2)
ax.plot([shoulder[0], wrist[0]], [shoulder[1], wrist[1]], '--', color='C0', lw=2.2)
# kx and kz
ax.plot([0, kx], [kz, kz], ':', color='C2', lw=1.8)
ax.plot([kx, kx], [0, kz], ':', color='C2', lw=1.8)
ax.text(kx/2, kz - 0.022, r'$k_x$', color='C2', fontsize=13, ha='center')
ax.text(kx + 0.008, kz/2, r'$k_z$ (negative here)', color='C2', fontsize=11, va='center')
# joints
for point in [shoulder, elbow, wrist]:
    ax.plot(point[0], point[1], 'o', ms=11, mfc='white', mec='0.15', mew=2, zorder=5)
ax.text(shoulder[0] - 0.13, shoulder[1] + 0.01, 'shoulder (J2)', fontsize=11)
ax.text(elbow[0] - 0.075, elbow[1] + 0.012, 'elbow (J3)', fontsize=11)
ax.text(wrist[0] + 0.012, wrist[1] - 0.012, 'wrist centre', fontsize=11)
# labels of the links
ax.text((shoulder[0] + elbow[0])/2 + 0.012, (shoulder[1] + elbow[1])/2 - 0.012, r'$a_2$', fontsize=14)
ax.text((elbow[0] + offset[0])/2, (elbow[1] + offset[1])/2 + 0.012, r'$a_3$', fontsize=14, ha='center')
ax.text((offset[0] + wrist[0])/2 + 0.01, (offset[1] + wrist[1])/2, r'$d_4$', fontsize=14)
ax.text((elbow[0] + wrist[0])/2 - 0.045, (elbow[1] + wrist[1])/2 - 0.01, r'$R$', color='C3', fontsize=15)
ax.text(0.11, -0.008, r'$D=\sqrt{k_x^2+k_z^2}$', color='C0', fontsize=12, ha='center', va='center')
# angle beta at the elbow (between a3 and R) and theta2 at the shoulder (from the vertical)
angle_a3 = np.degrees(np.arctan2(offset[1] - elbow[1], offset[0] - elbow[0]))
angle_R = np.degrees(np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]))
ax.add_patch(Arc(elbow, 0.07, 0.07, theta1=angle_R, theta2=angle_a3, color='C3', lw=1.6))
ax.text(elbow[0] + 0.036, elbow[1] - 0.06, r'$\beta=69.1^\circ$' + '\n(fixed)', color='C3', fontsize=11)
angle_a2 = np.degrees(np.arctan2(elbow[1], elbow[0]))
ax.add_patch(Arc(shoulder, 0.09, 0.09, theta1=angle_a2, theta2=90, color='C1', lw=1.8))
ax.text(0.012, 0.06, r'$\theta_2$', color='C1', fontsize=14)
ax.set_xlim(-0.17, 0.38); ax.set_ylim(-0.2, 0.27)
ax.set_title(r'Step 3 in the arm plane, at impact ($\theta_2=%.1f^\circ$, $\theta_3=%.1f^\circ$)' % (np.degrees(theta2), np.degrees(theta3)), fontsize=12)
steps = ('1. Replace the L-shaped forearm ($a_3$ then $d_4$) by one straight segment $R$.\n'
         '2. Triangle shoulder–elbow–wrist has sides $a_2$, $R$, $D$.\n'
         r'    $a_2$ and $R$ are fixed, so $D$ depends only on the elbow $\to$ gives $\theta_3$.' + '\n'
         r'3. Rotate the triangle about the shoulder until the wrist lands on $(k_x, k_z)$ $\to$ gives $\theta_2$.')
ax.text(-0.17, -0.198, steps, fontsize=10.5, va='bottom', bbox=dict(fc='white', ec='0.6'))

fig.tight_layout()
fig.savefig(os.path.join(folder, "IK_step3.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved IK_step3.png")


# =============================================================================
# 6. SPLINE TIMING - why the factor 2 + velocity clamping before/after
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

# --- left: downswing segment with d = 1 and T = 1, for 3 different end speeds v ---
tau = np.linspace(0, 1, 200)
for v, col, label in [(1, 'C3', r'$v=d/T$: slows down before impact'),
                      (2, 'C2', r'$v=2d/T$: uniform acceleration (used)'),
                      (4, 'C1', r'$v=4d/T$: first moves backwards')]:
    speed = 6*tau*(1 - tau) + v*tau*(3*tau - 2)   # derivative of the Hermite polynomial
    if v == 2:
        ax1.plot(tau, speed, color=col, lw=2.6, label=label)
    else:
        ax1.plot(tau, speed, color=col, lw=1.8, label=label)
ax1.axhline(0, color='k', lw=.8); ax1.axvline(1, color='0.6', ls=':')
ax1.text(1.01, 0.2, 'impact\n$\\mathbf{P}_2$', fontsize=10)
ax1.text(-0.01, -0.3, 'top of\nbackswing $\\mathbf{P}_1$', fontsize=10)
ax1.set_xlabel(r'normalised time $\tau$ on the downswing segment'); ax1.set_ylabel(r'speed along the swing [$d/T$]')
ax1.set_title('(a) Downswing segment: start at rest, cover $d$ in time $T$, end at speed $v$', fontsize=11.5)
ax1.legend(fontsize=10, loc='upper left'); ax1.grid(alpha=.3); ax1.set_xlim(-0.02, 1.15)

# --- right: fastest joint with the nominal times and after the velocity clamp ---
def fastest_joint(t_corners):
    '''
    goal: speed of the fastest joint along the swing, for given waypoint times (same steps as main_swing: spline -> IK -> joint speeds)
    PARAMS:
        t_corners: time at each waypoint (s)
    RETURNS:
        tq: 100 time samples (s)
        max|dq/dt| over the 6 joints at each time sample (rad/s)
    '''
    vel = rm.find_waypoint_velocities(waypoints, t_corners)
    spline_x = CubicHermiteSpline(t_corners, waypoints[:, 0], vel[:, 0])
    spline_y = CubicHermiteSpline(t_corners, waypoints[:, 1], vel[:, 1])
    spline_z = CubicHermiteSpline(t_corners, waypoints[:, 2], vel[:, 2])
    tq = np.linspace(0, t_corners[-1], 100)
    path = np.column_stack([spline_x(tq), spline_y(tq), spline_z(tq)])
    joints = []
    for target in path:
        joints.append(rm.solve_ik_with_bounds(rm.get_gripper_pos(target), tip_pos=target))
    joints = np.unwrap(np.array(joints), axis=0)
    dq = np.gradient(joints, tq, axis=0)
    return tq, np.max(np.abs(dq), axis=1)   # fastest joint at each time

t_nominal = rm.find_time_steps_from_waypoints(waypoints)   # times before the velocity clamp
tq_before, fastest_before = fastest_joint(t_nominal)
tq_after, fastest_after = fastest_joint(ms.t_corners)       # times after the velocity clamp
alpha = ms.t_corners[-1] / t_nominal[-1]
ax2.plot(tq_before, fastest_before, color='C3', lw=2,
         label='nominal timing (%.2f s): peak %.1f rad/s' % (t_nominal[-1], fastest_before.max()))
ax2.plot(tq_after, fastest_after, color='C2', lw=2,
         label=r'scaled by $\alpha=%.2f$ (%.2f s): peak %.2f rad/s' % (alpha, ms.t_corners[-1], fastest_after.max()))
ax2.axhline(np.pi, color='k', ls='--', lw=1.2, label=r'limit $\dot q_{max}=\pi$ rad/s')
ax2.set_xlabel('t [s]'); ax2.set_ylabel(r'fastest joint $\max_j|\dot q_j|$ [rad/s]')
ax2.set_title(r'(b) Velocity clamping: same path, slower clock ($t_{new}=\alpha\,t$)', fontsize=11.5)
ax2.legend(fontsize=10); ax2.grid(alpha=.3)

fig.tight_layout()
fig.savefig(os.path.join(folder, "Spline_timing.png"), dpi=180, bbox_inches='tight')
plt.close(fig)
print("saved Spline_timing.png")


# =============================================================================
# 7. RESULT PLOTS - 3D path, joint velocities, tip speed
# =============================================================================
# --- 3D path of the club tip ---
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
ax.plot(ms.path_xyz[:, 0], ms.path_xyz[:, 1], ms.path_xyz[:, 2], 'k-', lw=1.5, label='tip path')
for i in range(4):
    ax.scatter(*waypoints[i], s=60, color=colors[i], label=f'$P_{i}$')
ax.scatter(*ball, color='gold', s=200, edgecolor='k', label='ball', alpha=0.6)
ax.scatter(0, 0, 0, color='k', marker='^', s=80, label='base')
ax.scatter(*pivot, color='gray', marker='x', s=60, label=r'swing pivot $\mathbf{c}$')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.legend(fontsize=8, loc='upper left'); ax.view_init(22, -60)
fig.tight_layout()
fig.savefig(os.path.join(folder, "3d_path.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved 3d_path.png")

# --- joint velocities vs time ---
fig, ax = plt.subplots(figsize=(7, 4))
for j in range(6):
    ax.plot(ms.t_vel, np.abs(ms.joint_velocities[:, j]), label=f'$\\theta_{j+1}$')
ax.axhline(rm.MAX_JOINT_VELOCITY, color='r', ls='--', label=r'limit $\pi$ rad/s')
for i, t in enumerate(ms.t_corners):
    ax.axvline(t, color='gray', ls=':')
    ax.text(t, 3.25, f'$P_{i}$', ha='center')
ax.set_ylim(0, 3.45); ax.set_xlabel('t [s]'); ax.set_ylabel(r'$|\dot\theta_i|$ [rad/s]')
ax.grid(alpha=.4); ax.legend(fontsize=8, ncol=2, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(folder, "joint_vel.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved joint_vel.png")

# --- club tip speed vs time ---
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ms.t_vel, ms.tip_speed, color='purple', lw=2, label='tip speed')
top = ms.tip_speed.max() * 1.12
for i, t in enumerate(ms.t_corners):
    ax.axvline(t, color='gray', ls=':')
    ax.text(t, top, f'$P_{i}$', ha='center')
ax.set_ylim(0, top * 1.1); ax.set_xlabel('t [s]'); ax.set_ylabel(r'$\|\dot{\mathbf{p}}\|$ [m/s]')
ax.grid(alpha=.4); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(folder, "tip_vel.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("saved tip_vel.png")

print("\nall figures are in", folder)