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

	
	p = dvrk.psm('PSM1') 
	p.enable()
	p.home()
	downLocation = PyKDL.Vector(0,0, 0)
	goal = p.measured_cp()
	goal.p = downLocation
	p.move_cp(goal).wait()
	currentLocation = p.measured_cp().p
	
	
	TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.1,.05, 0))
	TrajctoryMain.defualtZLayer = -.17
	print("Going home")
	TrajctoryMain.returnHome()
	print("Homed")
	time.sleep(5)

	#TrajctoryMain.pickAndPlace(pickLocation = (-.13,.1), placeLocation = (-.1,-.05))
	TrajctoryMain.pickAndPlace(pickLocation = (-0.13,-0.1), placeLocation = (-.05,-.05))


	'''
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')
	ecm = dvrk.arm('ECM')

	#TrajctoryMain.returnHome()

	# -------------------------
	while isinstance(right_cam.get_image(), list):
			r.sleep()
			#print('sleeping')
	print('cv file recieved')
	

	#get 2d coordinates from images
	_, _, coords_2dR, coords_pickupR, _ = imTools.procImage(right_cam.get_image())
	_, _, coords_2dL, coords_pickupL, _ = imTools.procImage(left_cam.get_image())
	print("right camera coords",coords_2dR)
	if not coords_pickupL:
		coords_pickupL=[0,0]
	coords_pickupR=coords_pickupL
	print("right pickup coords",coords_pickupR)
	print("left pickup coords", coords_pickupL)
	#get 3d points of starting point
	
	coords_3d_pickup = imTools.findDepth(coords_pickupR[0], coords_pickupR[1],
							coords_pickupL[0], coords_pickupL[1])
	print('3D pickup coords',coords_3d_pickup)
	#get 2d coords of end point
	#ind_to_play=1
	#print("output coords",coords_2dR)
	#print("outputcoords[0]",coords_2dR[0])
	#print("output coords[0][0]]",coords_2dL[0][0])
	#print("output coords[0][0][0]]",coords_2dL[0][0][0])
	xr, yr = coords_2dR[0][0]
	xl, yl = coords_2dL[0][0]
	print("right dropoff coords",xr,yr)
	print("left dropoff coords", xl, yl)
	#get 3d points of end point
	coords_3d_putdown = imTools.findDepth(xr,yr,xl,yl)
	print("3D dropoff coords",coords_3d_pickup)
	'''