import os
import sys
import sympy as sp

print("Starting kinematics calculation script...")

# --- 1. System Variables ---
theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols('theta1 theta2 theta3 theta4 theta5 theta6')
a3, d2, d4, d5, d6, L = sp.symbols('a3 d2 d4 d5 d6 L')

# measurements (m)
A2 = 0.2
A3 = 0.087
D1 = 0.2433
D4 = 0.2276
D6 = 0.0615
# DH Table Matrix: [a, alpha, theta, d]
dh_table = sp.Matrix([
    [0,      -sp.pi/2,        theta1,                D1],
    [A2,     sp.pi,           -sp.pi/2 + theta2,      0],
    [A3,   sp.pi/2,           -sp.pi/2 + theta3,      0],
    [0,       sp.pi/2,        theta4,                D4], 
    [0,       -sp.pi/2,       theta5,                 0],
    [0,       0,              theta6,                D6],
])

# --- 2. Transformation Matrix Function ---
def make_transformation_matrix(dh_row):
    a = dh_row[0, 0]
    alpha = dh_row[0, 1]
    theta = dh_row[0, 2]
    d = dh_row[0, 3]
    
    cos_theta = sp.cos(theta)
    sin_theta = sp.sin(theta)
    cos_alpha = sp.cos(alpha)
    sin_alpha = sp.sin(alpha)
    
    row1 = [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta]
    row2 = [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta]
    row3 = [0,          sin_alpha,              cos_alpha,             d]
    row4 = [0,          0,                      0,                     1]
    
    return sp.Matrix([row1, row2, row3, row4])

def transformation_matrix_list(dh_table_):
    T_matrixes = []
    for i in range(dh_table_.rows):
        T = make_transformation_matrix(dh_table_[i, :])
        T_matrixes.append(T)
    return T_matrixes

print("Computing individual transformation matrices...")
T_list = transformation_matrix_list(dh_table)

# --- 3. FAST Matrix Chain Multiplication ---
def T_i_j(Matrix_list, i, j):
    T_total = Matrix_list[i]
    for k in range(i + 1, j):
        T_total = T_total * Matrix_list[k]
    T_total = sp.simplify(T_total)
    return T_total

print("Calculating full T_0_6 matrix...")
T_0_6 = T_i_j(T_list, 0, 6)

# --- 4. Clean Layout Substitutions ---
c1, s1, c2, s2, c3, s3, c4, s4, c5, s5, c6, s6 = sp.symbols('c1 s1 c2 s2 c3 s3 c4 s4 c5 s5 c6 s6')

substitutions = {
    sp.cos(theta1): c1, sp.sin(theta1): s1,
    sp.cos(theta2): c2, sp.sin(theta2): s2,
    sp.cos(theta3): c3, sp.sin(theta3): s3,
    sp.cos(theta4): c4, sp.sin(theta4): s4,
    sp.cos(theta5): c5, sp.sin(theta5): s5,
    sp.cos(theta6): c6, sp.sin(theta6): s6,
}

print("Applying substitutions...")
T_0_6_clean = T_0_6.subs(substitutions)

# --- 5. Display Outputs ---
print("\n=== FINAL FORWARD KINEMATICS MATRIX (T_0_6) ===")
sp.pprint(T_0_6_clean, use_unicode=True)

print("\n=== GRIPPER POSITION EQUATIONS ===")
print(f"X = {T_0_6_clean[0, 3]}")
print(f"Y = {T_0_6_clean[1, 3]}")
print(f"Z = {T_0_6_clean[2, 3]}")


# additional VALUES

L_CLUB = sp.symbols('L_CLUB')   # club length (m)

# Club tip expressed in the flange frame: pure translation along tool Z
T_6_tip = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, L_CLUB],
    [0, 0, 0, 1],
])

T_0_tip = sp.simplify(T_0_6 * T_6_tip)

X_tip = T_0_tip[0, 3]
Y_tip = T_0_tip[1, 3]
Z_tip = T_0_tip[2, 3]