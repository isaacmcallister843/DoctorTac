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
import trajecPlanning.Trajectory_Toolbox as Trajectory_Toolbox
import rospy

class TrajctoryNode(object):

    def __init__(self, homeLocation):
        self.jacobian_val = None  
        self.subscriber = rospy.Subscriber( '/PSM1/body/jacobian', Float64MultiArray, self.jacobian_callback, queue_size=1)
        
        p = dvrk.psm('PSM1') 
        p.enable()
        p.home()

        self.p = p 
                
        self.defualtTotalTime = 1.1
        self.defualtFreq = 50
        self.defualtZLayer = -.1135
        self.defualtExtensionHeight = .01

        self.homeLocation = homeLocation 
        self.currentLocation = p.measured_cp().p
        print(self.currentLocation)

    def jacobian_callback(self, msg):
        """
        Callback function that receives the Jacobian data and stores it.
        """
        jac_data = np.array(msg.data)
        self.jacobian_val = jac_data.reshape((6, 6))

    def returnHome(self): 
        """
        Go back to neutral locaiton  
        """
        # self.zAdjust(self.defualtZLayer)
        homeLocation_vec = PyKDL.Vector(self.homeLocation[0], self.homeLocation[1], self.defualtZLayer)
        goal = self.p.measured_cp()
        goal.p = homeLocation_vec
        self.p.move_cp(goal).wait()
        currentq = self.p.measured_jp()
        currentq[4] = 0 
        currentq[5] = math.radians(20)
        self.p.move_jp(currentq).wait()
	#print(currentPos)
        #newPath = Trajectory_Toolbox.forwardTrajectory((currentPos[0],currentPos[1], 0), self.homeLocation, 
        #        target_z_height = .02, total_time = self.defualtTotalTime, freqeuncy = self.defualtFreq)
        #self.exicutePath(currentPath=newPath)

    def straightlineMovement(self, targetLocation):
        self.currentLocation = self.p.measured_cp().p 
        newLocation = PyKDL.Vector(targetLocation[0], targetLocation[1], self.defualtZLayer)
        goal = self.p.measured_cp()
        goal.p = newLocation
        self.p.move_cp(goal).wait()
        self.currentLocation = self.p.measured_cp().p

         
    def exicutePath(self, currentPath):
        points = currentPath.returnJustPoints() 
        vel = currentPath.returnJustVel()
        vel[:, 2] = vel[:, 2] * -1


        q_0 = self.p.measured_jp()
        current_q = q_0

        for i in range(len(vel)): 
            v = np.append( [vel[i][1],vel[i][0],vel[i][2]], [0, 0, 0], axis=0)  
            q_dot = np.dot(np.linalg.inv(self.jacobian_val), v)
            current_q = current_q + q_dot * currentPath.total_time/currentPath.freqeuncy
            self.p.move_jp(current_q).wait()
            time.sleep(.01)

        finalLocation = PyKDL.Vector(points[-1][0],points[-1][1], self.defualtZLayer)
        goal = self.p.measured_cp()
        goal.p = finalLocation
        self.p.move_cp(goal).wait()
        self.currentLocation = self.p.measured_cp().p

    def zAdjust(self, targetZLevel): 
        """
        Just changes Z level  
        """
        self.currentLocation = self.p.measured_cp().p
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
        print("Going to pickup location ")
        time.sleep(5)
        firstPath = Trajectory_Toolbox.forwardTrajectory((self.currentLocation[0],self.currentLocation[1], 0), (pickLocation[0], pickLocation[1], 0), 
                target_z_height = .01, total_time = 2, freqeuncy = 50)
        # firstPath.createPlot()
        self.exicutePath(firstPath)
        self.currentLocation = self.p.measured_cp().p
        print("At pickup location")
        time.sleep(2)
    
        """
                Pickup Target 
        """
        print("Grabbing Item")        
        self.p.jaw.open()
        time.sleep(3)
        self.currentLocation = self.p.measured_cp().p
        self.zAdjust(zHeight + extenstionHeight)
        
        time.sleep(3)
        self.p.jaw.close()
        time.sleep(2)
        self.zAdjust(zHeight)
        print("Has Item") 
        time.sleep(2)

        """
                Go to place location  
        """
        print("Going to place location") 
        self.currentLocation = self.p.measured_cp().p
        
        secondPath = Trajectory_Toolbox.forwardTrajectory((self.currentLocation[0],self.currentLocation[1], 0), (placeLocation[0], placeLocation[1], 0), 
                target_z_height = .02, total_time = totalTime, freqeuncy = freqeuncy)
        self.exicutePath(secondPath)
        time.sleep(2)
        
        

        """
                Place Target 
        """
        print("Placing piece") 
        self.zAdjust(zHeight + extenstionHeight)
        time.sleep(2)
        self.p.jaw.open()
        time.sleep(2)
        self.p.jaw.close()
        time.sleep(2)
        self.zAdjust(zHeight)
        time.sleep(1)
        
        print("Going back home ") 

        """
                Rehome 
        """
        self.returnHome()
    
    def pickAndPlace_SL(self, pickLocation, placeLocation, zHeight  =  None, extenstionHeight = None, totalTime = None, freqeuncy = None): 
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
        print("Going to pickup location ")
        time.sleep(2)
        self.straightlineMovement(pickLocation)
        print("At pickup location")
        time.sleep(2)
    
        """
                Pickup Target 
        """
        print("Grabbing Item")        
        self.p.jaw.open()
        time.sleep(3)
        self.currentLocation = self.p.measured_cp().p
        self.zAdjust(zHeight + extenstionHeight)
        
        time.sleep(3)
        self.p.jaw.close()
        time.sleep(2)
        self.zAdjust(zHeight)
        print("Has Item") 
        time.sleep(2)

        """
                Go to place location  
        """
        print("Going to place location") 
        
        self.straightlineMovement(placeLocation)
        time.sleep(2)
        
        """
                Place Target 
        """
        print("Placing piece") 
        self.zAdjust(zHeight + extenstionHeight)
        time.sleep(2)
        self.p.jaw.open()
        time.sleep(2)
        self.p.jaw.close()
        time.sleep(2)
        self.zAdjust(zHeight)
        time.sleep(1)
        
        print("Going back home ") 

        """
                Rehome 
        """
        self.returnHome()

    def spin(self):
        """
        Keeps the node running and processing messages.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        TrajctoryNode = TrajctoryNode(homeLocation = (.14,.1, 0))
        TrajctoryNode.defualtZLayer = .6
        TrajctoryNode.returnHome()
        #time.sleep(.5)
        #TrajctoryNode.pickAndPlace(pickLocation = (0,.1), placeLocation = (0,-.1))

# TrajctoryNode.spin()
    except rospy.ROSInterruptException:
        pass
