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

<<<<<<< HEAD
	'''
	p = dvrk.psm('PSM1') 
	p.enable()
	p.home()
	downLocation = PyKDL.Vector(0,0, 0)
	goal = p.measured_cp()
	goal.p = downLocation
	p.move_cp(goal).wait()
	currentLocation = p.measured_cp().p
	'''
	'''
	TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.1,.05, 0))
	TrajctoryMain.defualtZLayer = -.17
	print("Going home")
	TrajctoryMain.returnHome()
	print("Homed")
	time.sleep(5)

	TrajctoryMain.pickAndPlace(pickLocation = (-.13,.1), placeLocation = (-.1,-.05))
	'''
	
=======
    # ----- Initalization ----- 
	TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (.14,.1, 0))

>>>>>>> 1f6b51f1fd260de153b12d7b2d717e10b0a92524
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')
	ecm = dvrk.arm('ECM')

<<<<<<< HEAD
	#TrajctoryMain.returnHome()

	# -------------------------
	while isinstance(right_cam.get_image(), list):
			r.sleep()
			#print('sleeping')
	print('cv file recieved')
	
	#get 2d coordinates from images
	_, _, coords_2dR, coords_pickupR, _ = imTools.procImage(right_cam.get_image())
	_, _, coords_2dL, coords_pickupL, _ = imTools.procImage(left_cam.get_image())
	print(coords_2dR)
	print(coords_pickupR)
=======
	TrajctoryMain.returnHome()

	# -------------------------

	#get 2d coordinates from images
	_, _, coords_2dR, coords_pickupR, _ = imTools.procImage(right_cam.get_image())
	_, _, coords_2dL, coords_pickupL, _ = imTools.procImage(left_cam.get_image())
>>>>>>> 1f6b51f1fd260de153b12d7b2d717e10b0a92524

	#get 3d points of starting point
	coords_3d_pickup = imTools.findDepth(coords_pickupR[0], coords_pickupR[1],
							coords_pickupL[0], yl = coords_pickupL[1])
	
	#get 2d coords of end point
	ind_to_play=1
	xr, yr = coords_2dR[ind_to_play]
	xl, yl = coords_2dL[ind_to_play]

	#get 3d points of end point
<<<<<<< HEAD
	coords_3d_putdown = imTools.findDepth(xr,yr,xl,yl)
	
=======
	coords_3d_putdown = imTools.findDepth(xr,yr,xl,yl)
>>>>>>> 1f6b51f1fd260de153b12d7b2d717e10b0a92524
