#!/usr/bin/env python

# Author: Sayem/Bobsy Narayan
#Camera Function Class + camera test
# Date: 2024-03-08


import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
import cv2
import numpy as np
#import xlsxwriter
import dvrk 
import sys
from scipy.spatial.transform import Rotation as R
import os
import time

class camera:

	#def __init__(self, camera_name, ros_namespace = '/dVRK/'):
	def __init__(self, camera_name, ros_namespace = '/stereo/'):

		self.__camera_name = camera_name
		self.__ros_namespace = ros_namespace
		self.bridge = CvBridge()
		self.cv_image = []
		self.image_count = 1
		self.image_path = os.path.abspath(os.getcwd()) + '/Images/'



		#full_ros_namespace = self.__ros_namespace + self.__camera_name + '/decklink/camera'
		full_ros_namespace = self.__ros_namespace + self.__camera_name + '/image_raw'

		#subscriber
		rospy.Subscriber(full_ros_namespace, Image, self.image_callback, queue_size = 1, buff_size = 1000000)

	def image_callback(self, data):

		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			print('image called')
		except CvBridgeError as e:
			print(e)
			print('failed callback')
		self.save_image()


	def get_image(self):
		print('image receieved')
		return self.cv_image
		
	#saves the image in a folder. /home/fizzer/catkin_ws/src/dvrk-ros/dvrk_python/Images
	def save_image(self):

		if self.cv_image.size != 0:
			#cv2.imwrite(self.image_path + self.__camera_name+"/"+self.__camera_name+"_Camera" +"_" + str(self.image_count)+".png", self.cv_image)
			cv2.imwrite('/home/fizzer/catkin_ws/src/dvrk-ros/dvrk_python/Images'+str(self.image_count)+'.png',self.cv_image)
			self.image_count = self.image_count + 1
			print('image saved')
			return True
		else:	
			print('image not saved')
			return False
			

if __name__ == '__main__':
	rospy.init_node("Camera_call")	
	rospy.Rate(10000)
	left_cam=camera('left')
	rospy.spin()