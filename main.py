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
	TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.1,.05, 0))
	TrajctoryMain.defualtZLayer = -.17
	print("Going home")
	TrajctoryMain.returnHome()
	print("Homed")
	time.sleep(5)

	TrajctoryMain.pickAndPlace(pickLocation = (-.13,.1), placeLocation = (-.1,-.05))
	
