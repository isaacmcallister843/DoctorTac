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
import cv2 

# Get the absolute paths of the folders
image_processing_path = os.path.abspath("imageProcessing")
trajectory_planning_path = os.path.abspath("trajecPlanning")

# Append the paths to sys.path (modify these paths if your folders are located differently)
sys.path.append(image_processing_path)
sys.path.append(trajectory_planning_path)

import imageProccessing.camera as camera # DVRK camera code
import trajecPlanning.Trajectory_PSM as trajecTools
import imageProccessing.imageProcessingTools as imTools 
import imageProccessing.Player as Player
import imageProccessing.tictactoe as tictactoe

if __name__ == '__main__':
	
	# ----- ROS Setup ---------
	rospy.init_node('mainNode')
	r = rospy.Rate(1000) #per second

	#Camera initiation
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')
	ecm = dvrk.arm('ECM')
	while isinstance(right_cam.get_image(), list):
		r.sleep()
		
	testimg = right_cam.get_image()
	cv2.imshow('img',testimg)
	cv2.waitKey(0)



 