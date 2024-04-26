#!/usr/bin/env python
import sys
import os
import dvrk
import math
import sys
import rospy
import numpy as np
import PyKDL
import argparse
import time


# Get the absolute paths of the folders
image_processing_path = os.path.abspath("imageProcessing")
trajectory_planning_path = os.path.abspath("trajecPlanning")

# Append the paths to sys.path (modify these paths if your folders are located differently)
sys.path.append(image_processing_path)
sys.path.append(trajectory_planning_path)

import imageProccessing.camera as camera # DVRK camera code
import trajecPlanning.Trajectory_PSM as trajecTools
import imageProccessing.imageProcessingTools as imTools 

if __name__ == '__main__':
	
	# ----- ROS Setup ---------
	rospy.init_node('mainNode')
	r = rospy.Rate(100)

	def TrajecPlanningTest(): 
		TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.04,.09))
		TrajctoryMain.defualtZLayer = -.04
		time.sleep(1)
		TrajctoryMain.returnHomeFree()
		time.sleep(1)
		TrajctoryMain.pickAndPlace2(pickLocation=(-.08,.026), placeLocation=(-.001,.007))
	
	TrajecPlanningTest()



	