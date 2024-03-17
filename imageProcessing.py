#!/usr/bin/env python

# Author: Team3
# Date: 2024-03-08

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
import xlsxwriter
import dvrk 
import sys
from scipy.spatial.transform import Rotation as R
import os
import camera
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats




#Function is used to turn 2d coordinates into 3d coordinates
def findDepth(ur,vr,ul,vl):

	#need to do calibration using lab scripts.
	calib_matrix = np.array([[1,2,3,4],
				 [5,6,7,8],
				 [9, 10, 11, 12,]])
	

	#use qr factorization to get intrinsic and extrinsic matrices
	#(returns Q,R which are 3x3 matrixes. R is upper-triangular meaning its the intrinsic matrix)
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


def get2DCoords():
	#todo
	x=1; y=2

	return(x,y)


def imageProcessingMain():

	rospy.init_node('imageProc')

	#iniate camera objects
	#these are subscribed to raw_images from dvrk cameras
	#I think new images are continuously saved by objects (video esque)
	#todo - send images to image processing task
	left_cam = camera.camera('left')
	right_cam = camera.camera('right')

	ecm = dvrk.arm('ECM')

	#todo- float64 or float32?
	pub = rospy.Publisher('coordinates_3d', numpy_msg(Floats), queue_size=10)
	r = rospy.Rate(10)

	while not rospy.is_shutdown():
		#send images to camera processing
		#image processing
		#identify piece to play (x,y) in both cameras

		xr, yr = get2DCoords() #todo
		xl, yl = get2DCoords()

		coords_3d = findDepth(xr,yr,xl,yl)

		pub.publish(coords_3d)
		r.sleep()






