#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2015-02-22

# (C) Copyright 2015-2020 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

# Start a single arm using
# > rosrun dvrk_robot dvrk_console_json -j <console-file>

# To communicate with the arm using ROS topics, see the python based example dvrk_arm_test.py:
# > rosrun dvrk_python dvrk_arm_test.py <arm-name>

import dvrk
import math
import sys
import rospy
import numpy as np
import PyKDL
import argparse
import time
from std_msgs.msg import Float64MultiArray

sys.path.append("/home/fizzer/catkin_ws/src/dvrk-ros/dvrk_python/scripts/trajecPlanning/")
import Trajectory_Toolbox

p = dvrk.psm('PSM1') 
p.enable()
p.home()


'''
Jacobian Set
''' 

jacobian_val = None 


def jacobianCallback(msg): 
	global jacobian_val
	jac_data = np.array(msg.data)
	jac_array = jac_data.reshape((6,6))
	jacobian_val =  jac_array 


# def listen_jacobian(): 
# 		rospy.Subscriber('PSM1/body/jacobian', Float64MultiArray, jacobianCallback)
# 		time.sleep(.01)

rospy.Subscriber('PSM1/body/jacobian', Float64MultiArray, jacobianCallback)


if jacobian_val is not None: 
	print(jacobian_val)
else: 
	print("-------")
print("-------")

'''
Further Testing
'''

newPath = Trajectory_Toolbox.forwardTrajectory((0,.2, .0), (0, .04, 0), target_z_height = .02, total_time = 2, freqeuncy = 40)
points = newPath.returnJustPoints() 
vel = newPath.returnJustVel()
vel[:, 2] = vel[:, 2] * -1

home_start_location = PyKDL.Vector(0,.2,-.1135)
goal = p.measured_cp()
goal.p = home_start_location
p.move_cp(goal).wait()

time.sleep(.5)
q_0 = p.measured_jp()
current_q = q_0

# No height correction 

print(q_0)
print("-----------------")
print("Moved To home")
time.sleep(1)

for i in range(len(points)): 
	
	v = np.append(vel[i], [0, 0, 0], axis=0) # this probably wrong 
	
	q_dot = np.dot(np.linalg.inv(jacobian_val), v)
	#q_dot = np.linalg.inv(jacobian_val) * v 

	current_q = current_q + q_dot * 2/40
	#print(current_q)
	print(jacobian_val)
	print("-----")
	# reformat slightly 
	p.move_jp(current_q).wait()
	#time.sleep(.00001)
	print("Moved: ", i)

print("------------------------")
print(p.measured_cp())
print(points[-1])


"""
test_loc = np.array([np.pi/3,-1*np.pi/2,0.05,0,0,0])
p.move_jp(test_loc)
time.sleep(.5)
print("Measured Point: ",p.measured_cp())
print("Measured Joints",p.measured_jp())
print("-------")
print(jacobian_val)
print("-----")
print(np.matmul(jacobian_val,test_loc ))

# import Trajectory_Toolbox
import syss

# sys.path.insert(0, "/home/fizzer/catkin_ws/src/dvrk-ros/dvrk_python/scripts/trajecPlanning")

target_point_1 = (1 ,1, 0)
target_point_2 = (-3, 4, 0)
target_z_height = 1

plan1 = Trajectory_Toolbox.forwardTrajectory(target_point_1, target_point_2, target_z_height, 4, 100) 
print(plan1.returnJustPoints())
plan1.createPlot()

start_point = PyKDL.Vector(.1,.1,-.1135)
target_point = PyKDL.Vector(-.2,.2,-.1135)
home_point = (target_point + start_point) / 2


goal = p.measured_cp()

goal.p = start_point
p.move_cp(goal).wait()
time.sleep(1)

goal.p = home_point
goal.p[2] = .1
p.move_cp(goal).wait()
time.sleep(1)

goal.p = target_point
p.move_cp(goal).wait()
"""


"""
p.measured_cp()
curPos = p.measured_jp()
p.measured_jv() 

print(p)

target_point = PyKDL.Vector(.5,.6,-.1135) 

goal = p.measured_cp() 
goal.p = start_point
p.move_cp(goal).wait() 

time.sleep(4)
"""
