import os
import sys
import cv2
import argparse
import numpy as np
import imageProccessing.DetectionsOpenCV as DetectionsOpenCV
import imageProccessing.imutils as imutils

#import imageProccessing.imutils as imutils
#import imageProccessing.DetectionsOpenCV as DetectionsOpenCV
#from imageProccessing.alphabeta import Tic, get_enemy, determine
pickupCoordsContourCutoff=300 #Constant to change how much of image is checked for pickup locations
gridCircleCutoff=900 #Constant to change how much of image is checked for grid

def get_board_template(frame):
    '''Finds Board Coordinates using Circle Detection - Only detects left side of Image'''
    #height, width = frame.shape[:2]  # The first two elements are height and width
    #frame=frame[:, :gridCircleCutoff]
    circle_coords=DetectionsOpenCV.find_circles(frame) #Finds circle coordinates using find_Circles
    circle_coords=circle_coords.reshape(-1,2) #Resize array
    sorted_groups = [] #New array to hold sorted values, as circle_Coords is unsorted.
    circle_coords = circle_coords[np.argsort(circle_coords[:, 0])]
    # Since we know we're dealing with groups of 3, iterate through the sorted array in steps of 3
    for i in range(0, len(circle_coords), 3):
        # Extract the current group of 3
        group = circle_coords[i:i+3]
        
        # Sort this group by the y-value
        sorted_group = group[np.argsort(group[:, 1])]
        
        # Append the sorted group to our list of sorted groups
        sorted_groups.append(sorted_group)
    sorted_groups=np.vstack(sorted_groups)
    #print("sorted",sorted_groups)

    bottom_left = sorted_groups[0]
    bottom_center = sorted_groups[1]
    bottom_right = sorted_groups[2]
    middle_left = sorted_groups[3]
    middle_center=sorted_groups[4]
    middle_right = sorted_groups[5]
    top_left = sorted_groups[6]
    top_center = sorted_groups[7]
    top_right = sorted_groups[8]
    return [top_left, top_center, top_right,
            middle_left, middle_center, middle_right,
            bottom_left, bottom_center, bottom_right]
    #print(circle_coords)

def findcurrentboardcoords(frame):
    '''Finds Current Board state. Output compared against previous board state'''
    circle_coords=DetectionsOpenCV.find_circles(frame)#Finds circle coordinates using find_Circles
    circle_coords=circle_coords.reshape(-1,2) #Resize array
    sorted_groups = [] #New array to hold sorted values, as circle_Coords is unsorted.
    circle_coords = circle_coords[np.argsort(circle_coords[:, 0])] #sorts np.argsort
    return circle_coords

def findPickupCoords(frame):
    #Pixels of board, only pickup of board
    '''Finds pickup location of detected coloured area. Will detect outside of pixel coords of board on Right side. Computer Pieces will be blue.'''
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Use cleanup image in DetectionsOpenCV
    image_hsv=DetectionsOpenCV.cleanupImage(image_hsv)

    #Get Boundaries for where to detect for blocks. For this it will check the right 100 Pixels of image
    height, width = image_hsv.shape[:2]  # The first two elements are height and width
    image_hsv=image_hsv[:, -pickupCoordsContourCutoff:]

    #Create upper and Lower Bounds of HSV range
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])
    #Apply blue mask to image according to upper and lower bounds
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    #Find contours for the blue blocks
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Find largest contour to find closest block & add pickupCutoff to fix image coords
    largest_contour = max(blue_contours, key=cv2.contourArea)
    largest_contour[:, :, 0] += width-pickupCoordsContourCutoff
    #Find center pickup pixel coords of block
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    return [cX,cY], largest_contour


def contrast_image(img):
	# converting to LAB color space
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)

	# Applying CLAHE to L-channel
	# feel free to try different values for the limit and grid size:
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl = clahe.apply(l_channel)

	# merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv2.merge((cl,a,b))

	# Converting image from LAB Color model to BGR color spcae
	enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	# Stacking the original image with the enhanced image
	result = np.hstack((img, enhanced_img))
	#cv2.imshow('Result', result)
	return result


#########################################################
    #Read image
if __name__ == '__main__':

    img = cv2.imread('C:/Users/bobsy/Downloads/Test_Both.jpg')
    if img is None:
        print("Failed to load image.")
    else:
        print("Image loaded successfully.")

    pickupCoords, largest_contour = findPickupCoords(img)
    board=get_board_template(img)
    print(board)
    for i in range(9):
        gridCoord=board[i]
        cv2.circle(img, gridCoord, radius = 5, color = (255,0,0), thickness = -1)
    # Draw the largest contour for visualization & Draw center coordinate.
    cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 3)
    cv2.circle(img, pickupCoords, radius = 5, color = (0,255,0), thickness = -1)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
