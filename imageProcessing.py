#!/usr/bin/env python

# Author: Team3
# Date: 2024-03-08

'''
This file initiates node for image processing. 
Subscribes to both cameras, processes images. 
Based on image processing, code either waits for player to move, moves ECM so board is in frame,
	or doesn't move.
Image processing determines tictactoe matrix of play, coordinates of board, coordinates of 1 of remaining pieces
3D Coordinates for piece-to-pick-up and spot-to-put-down are published to "coordinates_3d" topic

TODO
	- all the image processing stuff (Ben)
	- camera calibration matrix
	- verification (all!)
	
	- instead of while-looping for player to make move, maybe this should be another node?
		- this-node gets images and sends them to some player-done task
		- when player has moved, player-done task publishes something (like a locking mechanism)
		- this-node subscribes to topic and finishes function when it receives topic

'''


#Random Empty File. 
#! /usr/bin/env python




import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
import std_msgs
import cv2
import numpy as np
#import xlsxwriter Not necessary for our code and requires a second installation
import dvrk 
import sys
from scipy.spatial.transform import Rotation as R
import os
import camera
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import tictactoe

class boardsquare:
	def __init__(self,x_coord,y_coord,tile) -> None:
		self.x_coord=x_coord
		self.y_coord=y_coord
		self.tile=tile
	def isFull(self)->bool:
		return not (self.tile is None)


#Function is used to turn 2d coordinates into 3d coordinates
def findDepth(ur,vr,ul,vl):

	#need to do calibration using lab scripts. 
	#Found by using camera calibration scripts that are used to find intrinsic and extrinsic camera matrices
	calib_matrix = np.array([[1,2,3,4],
				 [5,6,7,8],
				 [9, 10, 11, 12,]])
	

	#use qr factorization to get intrinsic and extrinsic matrices
	#(returns Q,R which are 3x3 matrixes. R is upper-triangular meaning its the intrinsic matrix)
	#A (matrix) is decomposed into QR, where Q is orthogonal matrix and R is upper triangle matrix.
	ext_matrix_left, int_matrix_left = np.linalg.qr(calib_matrix[0:3,0:3], mode='reduced')
	ext_matrix_right, int_matrix_right = np.linalg.qr(calib_matrix[0:3,0:3], mode='reduced')
	
	t_matrix = np.linalg.inv(int_matrix_right)*calib_matrix[:,3]

	#if right camera is the main camera
	#then difference btw them is tx of left camera
	b = t_matrix[0]

	#I'm not sure if fx is supposed to be left or right camera or if theyre identical
	fx = int_matrix_right[0][0]
	fy = int_matrix_right[1][1]
	ox = int_matrix_right[0][2]
	oy = int_matrix_right[1][2]

	#image processing give (u,v) of both cameras
	#(ur,vr) for right camera; (ul,vl) for left;
	x = (b*(ul-ox)/(ul-ur))
	y = (b*fx*(vl-oy)/(fy*(ul-ur)))
	z = (b*fx/(ul-ur))

	coords_3d = np.array([x, y, z], dtype=np.float32)

	return(coords_3d)



def procImage(image):
	#todo

	#if image is not fully in frame, send 'status' to tell ECM to move
	# 1= move y+, 2= move y-, 3=move x+, 4=move x-, 0=don't move
	#if status=9, it isn't the robots turn yet (player still moving, player hasn't played)
	status = 0

	if(status is not 0):
		return status

	#array corresponding to played squares
	#Board will be initialized using read board to get the x and y coordinates of board and place it in board array.
	#theoretically could just do this once, but player may jostle board while playing
	for row in range(3):
		for column in range(3):
			#board[row][column]=boardsquare(readboard(),None)
			board=0
	
	#use image detection to see what is on board. & determine if player is X or O (based on first move)
	#if player is None:
		#player=readboardforplayer
	player = 'X' 

	#coordinates of one of the pieces (off to the side) to pick up
	#coord_pickup=readboardforpiece()
	coords_pickup = [1,2] 

	#return status, board, np.array([[(board[row][col].x_coord, board[row][col].y_coord) for col in range(3)] for row in range(3)]), coords_pickup, player
	return status, board, np.array([[[1,2], [1,2], [1,2]], [[1,2], [1,2], [1,2]],[[1,2],[1,2],[1,2]]]), coords_pickup, player

def moveECM(status,ecm,r):

	#start position
	goal = ecm.setpoint_cp()

	if status is 9:
		r.sleep() #player hasn't finished playing, just sleep
	elif status is 1:
		goal.p[1] += 0.05 #move 5cm +y
	elif status is 2:
		goal.p[1] -= 0.05 #move 5cm -y
	elif status is 3:
		goal.p[0] += 0.05 #move 5cm +x
	elif status is 2:
		goal.p[0] -= 0.05 #move 5cm -x

	ecm.move_cp(goal).wait()

	


def end_game (winner):
	i=0
	#Move ecm to home position
	if winner is 1:
		print("Player Won")
	elif winner is 2:
		print("DVRK Won")
	elif winner is 3:
		print("Draw Game")
	else:
		("Unknown Value inputed")

def imageProcessingMain():

	#create node
	rospy.init_node('imageProc')

	#initiate camera objects
	#these are subscribed to raw_images from dvrk cameras
	#Creates nodes that subscribes to camera image publisher node
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')

	#initiate ecm object
	ecm = dvrk.arm('ECM')

	#todo- float64 or float32?
	#Note: I got rate to be like 10000 from TA, might want to consider increasing?
	pub = rospy.Publisher('coordinates_3d', numpy_msg(Floats), queue_size=10)
	r = rospy.Rate(10)

	while not rospy.is_shutdown():

		#image processing function takes OpenCV image
		status, board, coords_2dR, coords_pickupR, player = procImage(right_cam.get_image())

		#if image is not in frame, move ECM
		while(status is not 0):
			moveECM(status,ecm,r)
			#look again
			status, board, coords_2dR, coords_pickupR, player = procImage(right_cam.get_image())

		#status is 0, get left image now
		_, _, coords_2dL, coords_pickupL, _ = procImage(left_cam.get_image())

		#play tictactoe
		ind_to_play, winner = tictactoe.play(board, player)

		#someone has won game or draw. end game sequence
		if(winner is not 0):
			end_game(winner)

		#identify piece to play (x,y) in both cameras
		coords_3d_pickup = findDepth(coords_pickupR[0], coords_pickupR[1],
							   coords_pickupL[0], yl = coords_pickupL[1])

		#identify location to play (x,y) in both cameras
		xr, yr = coords_2dR[ind_to_play]
		xl, yl = coords_2dL[ind_to_play]
		coords_3d_putdown = findDepth(xr,yr,xl,yl)


		#combine into 1x6 array [pickup_coords, putdown_coords]
		coords_3d = np.concatenate((coords_3d_pickup, coords_3d_putdown), axis=None)
		pub.publish(coords_3d)
		r.sleep()






