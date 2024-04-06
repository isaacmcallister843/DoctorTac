#!/usr/bin/env python

import dvrk
import math
import sys
import rospy
import numpy as np
import PyKDL
import argparse
import time
from std_msgs.msg import Float64MultiArray
import Trajectory_Toolbox
import rospy

class TrajctoryNode(object):

    def __init__(self, homeLocation):
        rospy.init_node('TrajctoryNode')  # Set a descriptive node name
        self.jacobian_val = None  
        self.subscriber = rospy.Subscriber( '/PSM1/body/jacobian', Float64MultiArray, self.jacobian_callback, queue_size=1)
        
        p = dvrk.psm('PSM1') 
        p.enable()
        p.home()

        self.p = p 
		
        self.defualtTotalTime = 1.1
        self.defualtFreq = 50
        self.defualtZLayer = -.1135
        self.defualtExtensionHeight = .03

        self.homeLocation = homeLocation 
        self.currentLocation = p.measured_cp().p

    def jacobian_callback(self, msg):
        """
        Callback function that receives the Jacobian data and stores it.
        """
        jac_data = np.array(msg.data)
        self.jacobian_val = jac_data.reshape((6, 6))
        # You can add additional processing of the Jacobian data here if needed.

    def returnHome(self): 
    	self.zAdjust(self.defualtZLayer)
        currentPos = self.currentLocation
        print(currentPos)
    	newPath = Trajectory_Toolbox.forwardTrajectory((currentPos[0],currentPos[1], 0), self.homeLocation, 
    		target_z_height = .02, total_time = self.defualtTotalTime, freqeuncy = self.defualtFreq)
    	self.exicutePath(currentPath=newPath)


    def exicutePath(self, currentPath):
    	points = currentPath.returnJustPoints() 
        vel = currentPath.returnJustVel()
        vel[:, 2] = vel[:, 2] * -1
    	
        q_0 = self.p.measured_jp()
        current_q = q_0

        for i in range(len(vel)): 
			
            v = np.append( [vel[i][1],vel[i][0],vel[i][2]], [0, 0, 0], axis=0) # this probably wrong 
			
            q_dot = np.dot(np.linalg.inv(self.jacobian_val), v)
            current_q = current_q + q_dot * currentPath.total_time/currentPath.freqeuncy
            self.p.move_jp(current_q).wait()
            current_q = self.p.measured_jp()

            current_point = self.p.measured_cp()
            current_point = current_point.p
            err  = math.sqrt((current_point[0]- points[i][0])**2 + (current_point[1]- points[i][1])**2)
            '''
			print("------------------")
			print("Target Point: ",  points[i])
			print("Velocity vec: ", v)
			print("Current Point: ", current_point)
			print(round(err,3))
            '''
        v = np.append( [vel[-1][1],vel[-1][0],vel[-1][2]], [0, 0, 0], axis=0)
        q_dot = np.dot(np.linalg.inv(self.jacobian_val), v)
        current_q = current_q + q_dot * currentPath.total_time/currentPath.freqeuncy
        self.p.move_jp(current_q).wait()

        finalLocation = PyKDL.Vector(points[-1][0],points[-1][1], self.defualtZLayer)
        goal = self.p.measured_cp()
        goal.p = finalLocation
        self.p.move_cp(goal).wait()

        self.currentLocation = self.p.measured_cp().p

    def zAdjust(self, targetZLevel): 
        downLocation = PyKDL.Vector(self.currentLocation[0],self.currentLocation[1],targetZLevel)
        goal = self.p.measured_cp()
        goal.p = downLocation
        self.p.move_cp(goal).wait()
        self.currentLocation = self.p.measured_cp().p

    def pickAndPlace(self, pickLocation, placeLocation, zHeight  =  None, extenstionHeight = None, totalTime = None, freqeuncy = None): 
    	if zHeight is None: 
    		zHeight  = self.defualtZLayer
    	if extenstionHeight is None: 
    		extenstionHeight = self.defualtExtensionHeight
    	if totalTime is None: 
    		totalTime = self.defualtTotalTime
    	if freqeuncy is None: 
    		freqeuncy = self.defualtFreq

    	"""
		Go to pick Location 
    	"""
    	firstPath = Trajectory_Toolbox.forwardTrajectory((self.currentLocation[0],self.currentLocation[1], 0), (pickLocation[0], pickLocation[1], 0), 
    		target_z_height = .02, total_time = totalTime, freqeuncy = freqeuncy)
    	self.exicutePath(firstPath)
    	self.p.jaw.open()
    	self.zAdjust(self.currentLocation[2] - extenstionHeight)


		# Extend Hands 
        time.sleep(1)

    def spin(self):
        """
        Keeps the node running and processing messages.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        TrajctoryNode = TrajctoryNode(homeLocation = (.14,.1, 0))
        TrajctoryNode.returnHome()
        time.sleep(.5)
        TrajctoryNode.pickAndPlace(pickLocation = (0,.1), placeLocation = (0,0))
        
        # TrajctoryNode.spin()
    except rospy.ROSInterruptException:
        pass
