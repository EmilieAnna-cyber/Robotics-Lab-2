import time
import numpy as np
from xarm.wrapper import XArmAPI
from main_swing import joint_trajectory, joint_velocities, tq

robot_ip = '192.168.1.158' #to be changed if different
speed_scale = 0.5 #used to scale the speed at which robot operates(1 = velocity as caluclated)

def follow_trajectory():
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