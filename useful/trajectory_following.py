# TASK 3 (execution on the robot)

# trajectory_following.py
# sends the swing of main_swing.py to the real Lite 6 with joint velocity control
# 1. moves slowly to the first pose of the swing (position mode)
# 2. streams the joint velocities every dt (velocity mode 4)
# 3. always stops safely at the end (velocity 0, back to mode 0), even after an error
# THE ROBOT MOVES: test first with trajectory_following_mock.py, keep someone at the emergency stop
# run from Robotics-Lab-2 :  python -m useful.trajectory_following   (close the plot window of main_swing to start)
import os
import sys
import time
import numpy as np
from xarm.wrapper import XArmAPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "import useful.xxx" works
from useful.main_swing import joint_trajectory, joint_velocities, tq   # FIX: main_swing_0 does not exist

robot_ip = '192.168.1.158' #to be changed if different
speed_scale = 0.5 #used to scale the speed at which robot operates(1 = velocity as caluclated)

def follow_trajectory():
    '''
    goal: move the robot to the start of the swing, then play the swing with joint velocity commands
    uses joint_trajectory, joint_velocities and tq from main_swing.py
    speed_scale < 1 plays the swing slower (0.5 = half speed, twice as long)
    '''
    arm = XArmAPI(robot_ip, is_radian=False)
    time.sleep(0.5)

    arm.motion_enable(enable=True) #supplies the power to robot's motors
    arm.clean_error() #this          two clear warnings and fault codes from previous use
    arm.clean_warn()  #     and this

    #Initial position
    arm.set_mode(0) #standard position control mode
    arm.set_state(0) #resets controller's state
    time.sleep(0.5)

    q0_degrees = np.degrees(joint_trajectory[0]).tolist() #extracts the first set of joint angles and converts them to degrees
    arm.set_servo_angle(angle=q0_degrees, speed=20, wait=True) #commands the arm to move the joints to those specific angles. 20 stands for deg/s ~ 0.35 rad/s
    time.sleep(1)

    #Switching to joint velocity control mode
    arm.set_mode(4)
    arm.set_state(0)
    time.sleep(0.5)

    dt = (tq[1] - tq[0]) / speed_scale
    vals_degrees = np.degrees(joint_velocities)*speed_scale #converting rad/s to deg/s and then applying the scale factor

    try:
        for qd in vals_degrees:
            #we take command execution time into consideration
            time_start = time.perf_counter() #high-res clock used to measure how long each command takes

            arm.vc_set_joint_velocity(qd.tolist()) #streaming velocity commands

            #ensuring the robot doesn't lag behind the schedule
            time_elapsed = time.perf_counter() - time_start
            time_sleep = dt - time_elapsed
            if time_sleep > 0:
                time.sleep(time_sleep)
    finally:
        #Ensuring safe stop
        arm.vc_set_joint_velocity([0.0]*6)
        time.sleep(0.5)
        arm.set_mode(0)
        arm.set_state(0)
        arm.disconnect()

if __name__ == '__main__':
    follow_trajectory()