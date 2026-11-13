#!/usr/bin/env python3
# TASK 3 (test without the robot)

# trajectory_following_mock.py
# same as trajectory_following.py, but with a fake robot (MockXArmAPI) that only prints the commands
# used to check the velocity stream without the hardware (no xArm SDK needed)
# run from Robotics-Lab-2 :  python -m useful.trajectory_following_mock   (close the plot window of main_swing to start)
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so "import useful.xxx" works
from useful.main_swing import joint_trajectory, joint_velocities, tq   # FIX: main_swing_0 does not exist

robot_ip = '192.168.1.158' #to be changed if different
speed_scale = 0.5 #used to scale the speed at which robot operates(1 = velocity as caluclated)

# Mock XArmAPI class for local testing without physical hardware
class MockXArmAPI:
    '''
    goal: fake xArm with the same functions as XArmAPI that are used in follow_trajectory()
    every function only prints what the real robot would do
    '''
    def __init__(self, ip, is_radian=False):
        print(f"[Mock] Connecting to robot at {ip} (is_radian={is_radian})")
        
    def motion_enable(self, enable=True):
        print(f"[Mock] Motion enable: {enable}")
        
    def clean_error(self):
        print("[Mock] Error cleared")
        
    def clean_warn(self):
        print("[Mock] Warning cleared")
        
    def set_mode(self, mode):
        print(f"[Mock] Setting control mode to: {mode}")
        
    def set_state(self, state):
        print(f"[Mock] Setting state to: {state}")
        
    def set_servo_angle(self, angle, speed, wait):
        print(f"[Mock] Moving to start angle P0 (speed={speed}, wait={wait})")
        time.sleep(0.5)
        
    def vc_set_joint_velocity(self, speeds):
        print(f"[Mock Velocity Stream] Joint 1 Speed: {speeds[0]:.2f} deg/s", end="\r")

    def disconnect(self):
        print("\n[Mock] Disconnected safely.")

    def vc_set_joint_velocity(self, speeds, step=None):
        formatted_speeds = [f"{s:6.2f}" for s in speeds]
        if step is not None:
            print(f"[Mock Step {step:3d}] Speeds (deg/s): {formatted_speeds}", end="\r")
        else:
            print(f"[Mock Stream] Speeds (deg/s): {formatted_speeds}", end="\r")

def follow_trajectory():
    '''
    goal: same steps as follow_trajectory() in trajectory_following.py, sent to the fake robot
    uses joint_trajectory, joint_velocities and tq from main_swing.py
    speed_scale < 1 plays the swing slower (0.5 = half speed, twice as long)
    '''
    arm = MockXArmAPI(robot_ip, is_radian=False)
    time.sleep(0.5)

    arm.motion_enable(enable=True) #supplies the power to robot's motors
    arm.clean_error() #this two clear warnings and fault codes from previous use
    arm.clean_warn()  # and this

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
        for i,qd in enumerate(vals_degrees):
            #we take command execution time into consideration
            time_start = time.perf_counter() #high-res clock used to measure how long each command takes

            arm.vc_set_joint_velocity(qd.tolist(), step=i) #streaming velocity commands

            #ensuring the robot doesn't lag behind the schedule (fixed with parentheses)
            time_elapsed = time.perf_counter() - time_start
            time_sleep = dt - time_elapsed
            if time_sleep > 0:
                time.sleep(time_sleep)
    finally:
        #Ensuring safe stop
        arm.vc_set_joint_velocity([0.0]*6)
        print()
        time.sleep(0.5)
        arm.set_mode(0)
        arm.set_state(0)
        arm.disconnect()

if __name__ == '__main__':
    follow_trajectory()