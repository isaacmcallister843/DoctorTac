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
	r = rospy.Rate(1000) #per second

	#setup PSM1
	p = dvrk.psm('PSM1') 
	p.enable()
	p.home()
	downLocation = PyKDL.Vector(0,0, 0)
	goal = p.measured_cp()
	goal.p = downLocation
	p.move_cp(goal).wait()
	currentLocation = p.measured_cp().p
	
	#Send PSM1 home
	TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.1,.05, 0))
	TrajctoryMain.defualtZLayer = -.17
	print("Going home")
	TrajctoryMain.returnHome()
	print("Homed")
	time.sleep(5)

	#testing----------------------
	#TrajctoryMain.pickAndPlace(pickLocation = (-.13,.1), placeLocation = (-.1,-.05))
	TrajctoryMain.pickAndPlace(pickLocation = (-0.13,-0.1), placeLocation = (-.05,-.05))
	#-----------------------------

	#Camera initiation
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')
	#ecm = dvrk.arm('ECM')

	#TrajctoryMain.returnHome()

	while not rospy.is_shutdown():

		#the first few calls happen before an image is sent by dvrk
		#so the image variable (from *_cam.get_image) will be empty lists
		#sleep until an image is sent and the image variable is no longer a list (should be CV image)
		while isinstance(right_cam.get_image(), list):
			r.sleep()
			#print('sleeping')
		#else:
			#print('cv file recieved')


		#-------------Get 2D coordinates from image
  
		status = 0 #set status to 0, procImage will change it if needed.

		#image processing function takes OpenCV image
		status, board, coords_2dR, coords_pickupR, player = imTools.procImage(right_cam.get_image(), status)

		#Wait until player has moved (ex: 9 circles -> 8 circles)
		while(status is not 0):
			#look again
			status, board, coords_2dR, coords_pickupR, player = imTools.procImage(right_cam.get_image(), status)
		
		#status is 0 (robot's turn), get left image now
		_, _, coords_2dL, coords_pickupL, _ = imTools.procImage(left_cam.get_image(), status)

		#-------------------------------------------

		print("right camera coords",coords_2dR)
		# if not coords_pickupL: #not sure what this is?
		# 	coords_pickupL=[0,0]
		# coords_pickupR=coords_pickupL #these should be different for 3d....
		print("right pickup coords",coords_pickupR)
		print("left pickup coords", coords_pickupL)

		#-----------2D to 3D pickup-----------------------------------

		coords_3d_pickup = imTools.findDepth(coords_pickupR[0], coords_pickupR[1],
								coords_pickupL[0], coords_pickupL[1])
		print('3D pickup coords',coords_3d_pickup)

		#-----------------------------------------------------
		#play tictactoe
		ind_to_play, winner = imTools.tictactoe.play(board, player)

		#someone has won game or draw. end game sequence
		if(winner is not 0):
			imTools.end_game()

		#identify location to play (x,y) in both cameras
		xr, yr = coords_2dR[ind_to_play]
		xl, yl = coords_2dL[ind_to_play]
		coords_3d_putdown = imTools.findDepth(xr,yr,xl,yl)
		print("3D dropoff coords",coords_3d_pickup)


		#--------testing???-------------------------------------------
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
		#-------------------------------------------------------------
  

		#todo: trajectory planning
		
		r.sleep()