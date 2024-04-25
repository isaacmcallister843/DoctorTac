#!/usr/bin/env python

# Author: Bobsy Narayan
# Date: 2024-03-08
# Edited: 2024-04-25
# Camera Function Class for DVRK Robot to access camera in ROS.

#Necessary Libraries
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
import cv2
import numpy as np
import dvrk 
import sys
from scipy.spatial.transform import Rotation as R
import os
import time

class camera:
	'''
	Camera class to access camera, save images, get images, etc/.
	'''
	#When accessing camera class in dvrk, ros node is /ubc_dVRK_ECM/. In simulator, ros_namespace=/stereo/
	def __init__(self, camera_name, ros_namespace = '/ubc_dVRK_ECM/'):
		self.__camera_name = camera_name #Camera_name will be left or right camera of ECM.
		self.__ros_namespace = ros_namespace
		self.bridge = CvBridge() #Used to convert ros Image Messages and Open CV images.
		self.cv_image = [] #array of pixels of CV_image.
		self.image_count = 1
		self.image_path = os.path.abspath(os.getcwd()) + '/Images/' #Path where images will be saved. 

		#Subscribe to ros node that publishes images from the ECM (for real robot) - Changed to compressed image
		#For ECM, use '/decklink/camera/image_raw/compressed'. For DVRK simulator, use '/image_raw/compressed'
		full_ros_namespace = self.__ros_namespace + self.__camera_name + '/decklink/camera/image_raw/compressed'

		#Subscribe to ROS Node where images are published from ECM.
		rospy.Subscriber(full_ros_namespace, CompressedImage, self.image_callback, queue_size = 1, buff_size = 1000000)

	def image_callback(self, data):
		'''
		Function to change ROS Image Messages to CV2 messages using RGB8 Colour Space.
		'''
		try:
			self.cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8") #Image converted from data variable to CV2 Image
		#	print('Image_Callback Successful') #Used to proper image confirmation to Terminal
		except CvBridgeError as e:
			print(e)
			print('Image_Callback Failed')
		#self.save_image()


	def get_image(self):
		return self.cv_image
		
	def save_image(self):
		'''
		Function to save Image to path. Will print to terminal if image path is incorrect or if no image is found.
		'''
		if self.cv_image.size != 0:
			#Save images to following path in Linux program. If path can't be found, prints failure to terminal.
			if not cv2.imwrite('/home/dvrk-pc/Desktop/Images/'+str(self.image_count)+str(self.__camera_name)+'.png',self.cv_image):
				print("Image Did Not Save to Path")
			self.image_count = self.image_count + 1
			return True
		else:	
			print('No Image Found')
			return False
			

if __name__ == '__main__':
	'''
	Testing Space. Will create ROS Node, activate class, and spin until end of program.
	'''
	rospy.init_node("Camera_call")	
	rospy.Rate(10000)
	left_cam=camera('left')
	rospy.spin()
