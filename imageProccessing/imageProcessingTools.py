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

#TODO
	- all the image processing stuff (Ben)
	- camera calibration matrix
	- verification (all!)
	
	- instead of while-looping for player to make move, maybe this should be another node?
		- this-node gets images and sends them to some player-done task
		- when player has moved, player-done task publishes something (like a locking mechanism)
		- this-node subscribes to topic and finishes function when it receives topic

'''

#Neccessary libraries
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
import std_msgs
import cv2 #Vision toolbox
import numpy as np #Matrix toolbox
import imageProccessing.AnalysisOpenCV as AnalysisOpenCV #Bens vision code
import dvrk #DVRK toolbox
import sys
from scipy.spatial.transform import Rotation as R
import os
import imageProccessing.camera as camera # DVRK camera code Comment bottom and uncomment this
#import camera
import imageProccessing.tictactoe as tictactoe

#keeps track of coordinates and value (X,O,blank)
class boardsquare:
	def __init__(self,x_coord,y_coord,tile):
		self.x_coord=x_coord
		self.y_coord=y_coord
		self.tile=tile #X,O,blank

	def isFull(self):
		return not (self.tile is None)

#Function is used to turn 2d coordinates into 3d coordinates
def findDepth(ur,vr,ul,vl):

	#Calibration from lab (do not edit):
	#ur = 144, vr =0, ul=89. vl=200
	#Assume its in pixel coords
	#is camera calibrations matrices in different coords?
	
	calib_matrixR = np.array([[1996.53569, 0, 936.08872, -11.36134],
				 [0, 1996.53569, 459.10171, 0],
				 [0, 0, 1, 0,]])
	
	calib_matrixL = np.array([[1996.53569, 0, 936.08872, 0],
				[0, 1996.53569, 459.10171, 0],
				[0, 0, 1, 0,]])
	
	#^^ the ros code for these outputs = [ fx' 0 cx' Tx
	#									0 fy' cy' Ty
	#									0 0 1 0]
	# (see sensor_msgs/CameraInfro Message)
	# So we don't need to decompose anything (red comments below) as we usually would need to
	#instead:
	fx = calib_matrixL[0][0]
	fy = calib_matrixL[1][1]
	ox = calib_matrixL[0][2]
	oy = calib_matrixL[1][2]
	b = -(calib_matrixR[0][3]/fx) #Tx = -fx' * B according to documentation
	
	'''
	#use qr factorization to get intrinsic and extrinsic matrices
	#(returns Q,R which are 3x3 matrixes. R is upper-triangular meaning its the intrinsic matrix)
	#A (matrix) is decomposed into QR, where Q is orthogonal matrix and R is upper triangle matrix.

	ext_matrix_left, int_matrix_left = np.linalg.qr(calib_matrixL[0:3,0:3], mode='reduced')
	ext_matrix_right, int_matrix_right = np.linalg.qr(calib_matrixR[0:3,0:3], mode='reduced')
	
	fx = int_matrix_right[0][0]
	fy = int_matrix_right[1][1]
	ox = int_matrix_right[0][2]
	oy = int_matrix_right[1][2]

	t_matrix = np.linalg.inv(int_matrix_right)*calib_matrixR[:,3]
amera.
	#if right camera is the main camera
	#then difference btw them is tx of left camera
	b = t_matrix[0]
	'''

	#image processing give (u,v) of both cameras
	#(ur,vr) for right camera; (ul,vl) for left;
	x = (b*(ul-ox)/(ul-ur))
	y = (b*fx*(vl-oy)/(fy*(ul-ur)))
	z = (b*fx/(ul-ur))

	#shift x so its between the two cameras
	x -= (b/2)

	coords_3d = np.array([x, y, z], dtype=np.float32)

	return(coords_3d)

def cameraToWorldChange(pixelCoords,Scale):
	calib_matrixR = np.array([[1996.53569, 0, 936.08872, -11.36134],
						   [0, 1996.53569, 459.10171, 0],
						   [0, 0, 1, 0,]])
	
	oxPixel = calib_matrixR[0][2]
	oyPixel = calib_matrixR[1][2]

	oxReal = oxPixel +pixelCoords[0]*Scale
	oyReal = oyPixel +pixelCoords[1]*Scale

	return [oxReal,oyReal]

def findBoardCoords(image):
	cells = AnalysisOpenCV.get_board_template(image)
	board = [None] * 9 
	for i in range(9):
		xcoord=cells[i][0]
		ycoord=cells[i][1]
		shape=None
		board[i]=boardsquare(xcoord,ycoord,shape)
	return board

def getNewBoardState(board,status,image, tolerance=20):
	current_circles=AnalysisOpenCV.findcurrentboardcoords(image)
	current_positions=[(c[0], c[1]) for c in current_circles]  # List of current (x, y) positions

	# Initialize a match found flag for each board square
	matched=[False]*len(board)
	
	# Loop through each board square and determine the closest current circle
	for i, square in enumerate(board):
		closest_dist = float('inf')
		for (cx, cy) in current_positions:
			dist = np.sqrt((square.x_coord - cx) ** 2 + (square.y_coord - cy) ** 2)
			if dist < closest_dist:
				closest_dist = dist
			#If the closest circle is within the tolerance and the square is not yet covered, it's considered unchanged
		if closest_dist <= tolerance:
			matched[i] = True

	# Update board squares where no close circle was found
	for i, square in enumerate(board):
		if not matched[i] and square.tile is None:
			if status%2==1:
				square.tile = 'X'  # Mark the square as covered with an 'X' or 'O'
			elif status%2==0:
				square.tile = 'O'
			status-=1
	
	for i in range(9):
		print(board[i].tile)
	return board,status

def findPickUpCoords(frame):
	largest_contour=AnalysisOpenCV.findComputerPickupBlocks(frame)
	M = cv2.moments(largest_contour)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	return [cX,cY]

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