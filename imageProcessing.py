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
import cv2 #Vision toolbox
import numpy as np #Matrix toolbox
import Player #Bens vision code
#import xlsxwriter Not necessary for our code and requires a second installation
import dvrk #DVRK toolbox
import sys
from scipy.spatial.transform import Rotation as R
import os
import camera #Bobsys DVRK camera code
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import tictactoe

#Constants for MoveECM function & status variable
leftBoundary	= 150
rightBoundary 	= -150
upperBoundary 	= 150
lowerBoundary 	= -150


#Code is getting messy. We have two tictactoe functions, and loaded files on multiple different files that won't run when imported.
#Ben, Mack, Bobsy, lets meet/discuss camera.py, player.py, imageProccessing.py initializations + Framework
#Player.py initializations arent added into this file

class boardsquare:
	def __init__(self,x_coord,y_coord,tile) -> None:
		self.x_coord=x_coord
		self.y_coord=y_coord
		self.tile=tile
	def isFull(self)->bool:
		return not (self.tile is None)

class imageProcessingMain():

	def __init__(self, r):
		#create node
		rospy.init_node('imageProc')
		#initiate camera objects - these are subscribed to raw_images from dvrk cameras
		self.left_cam = camera.camera('left')
		self.right_cam = camera.camera('right')
		#initiate ecm object
		self.ecm = dvrk.arm('ECM')
		#todo- float64 or float32?
		self.pub = rospy.Publisher('coordinates_3d', numpy_msg(Floats), queue_size=10)
		self.r = r

		#Subscribe to arm node so we know when its ready for next movement
		#rospy.Subscriber('ready_state', todo message type, self.image_callback, queue_size = 1, buff_size = 1000000)

		#Should we initiate self.status? self.board?

	def image_callback(self, data):
		#image processing function takes OpenCV image
		status, board, coords_2dR, coords_pickupR, player = procImage(self.right_cam.get_image())

		#if image is not in frame, move ECM
		while(status is not 0):
			moveECM(status,self.ecm,self.r)
			#look again
			status, board, coords_2dR, coords_pickupR, player = procImage(self.right_cam.get_image())

		#status is 0, get left image now
		_, _, coords_2dL, coords_pickupL, _ = procImage(self.left_cam.get_image())

		#play tictactoe
		ind_to_play, winner = tictactoe.play(board, player)

		#someone has won game or draw. end game sequence
		if(winner is not 0):
			end_game()

		#identify piece to play (x,y) in both cameras
		coords_3d_pickup = findDepth(coords_pickupR[0], coords_pickupR[1],
								coords_pickupL[0], yl = coords_pickupL[1])

		#identify location to play (x,y) in both cameras
		xr, yr = coords_2dR[ind_to_play]
		xl, yl = coords_2dL[ind_to_play]
		coords_3d_putdown = findDepth(xr,yr,xl,yl)


		#combine into 1x6 array [pickup_coords, putdown_coords]
		coords_3d = np.concatenate((coords_3d_pickup, coords_3d_putdown), axis=None)
		self.pub.publish(coords_3d)

#Function is used to turn 2d coordinates into 3d coordinates
def findDepth(ur,vr,ul,vl):

	#Calibration from lab (do not edit):
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


def procImage(image):
	#if image is not fully in frame, send 'status' to tell ECM to move
	# 1= move y+, 2= move y-, 3=move x+, 4=move x-, 0=don't move
	#if status=9, it isn't the robots turn yet (player still moving, player hasn't played)
	#we get topdown image and corners array. if array is not 4, we currently move to the right.
	#Need to fix this
	topdownimage, corners = Player.find_board(image)
	if corners.shape[0] == 4:
		status=0
	else:
		status=3

	#For now, lets assume player is always 'x'
	#if player is None: #player=readboardforplayer
	player = 'X' 
	
	#array corresponding to played squares
	#Board will be initialized using read board to get the x and y coordinates of board and place it in board array.
	#cells goes top left, middle left, right left, and holds another list (x,y,width,height)
	cells = Player.get_board_template(topdownimage)
	board = [None] * 9 
	for i in range(9):
			shape = Player.find_shape(cells[i])
			xcoord= cells[i][0]
			ycoord=cells[i][1]
			board[i]=boardsquare(xcoord,ycoord,shape)

	#coordinates of one of the pieces (off to the side) to pick up
	#coord_pickup=readboardforpiece()
	coords_pickup = Player.find_circles(topdownimage)

	#Chatgpt function to turn 1d array into 2d numpy array for later usage.
	coords_2d= np.array([[(board[row * 3 + col].x_coord, board[row * 3 + col].y_coord) for col in range(3)] for row in range(3)]),
	
	return status, board, coords_2d, coords_pickup, player

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

if __name__ == "__main__":

	r = rospy.Rate(10)

	ob = imageProcessingMain(r)
	data=0
	ob.image_callback(data) #I don't think it needs data

	while not rospy.is_shutdown():
		
		r.sleep()



