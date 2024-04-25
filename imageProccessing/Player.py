
import os
import sys
import cv2
import argparse
import numpy as np
import DetectionsOpenCV

import imageProccessing.imutils as imutils
import imageProccessing.DetectionsOpenCV as DetectionsOpenCV
from imageProccessing.alphabeta import Tic, get_enemy, determine

def find_board(frame, add_margin=True):
    """Detect the coords of the sheet of board the game will be played on using Haris corner detection"""
    thresh = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Change image to black&White Colourspace
    stats = DetectionsOpenCV.find_corners(thresh) #Use haris detections function
    # stats[0] is center of coordinates system, so ignored.
    corners = stats[1:, :2] #Get Board's corners
    corners = imutils.order_points(corners) #What's happening Here?
    # Get bird view of game board
    board = imutils.four_point_transform(frame, corners)
    if add_margin:
        board = board[10:-10, 10:-10]
    return board, corners

def find_grid(frame):
    '''Detect the coords of the sheet of board the game will be played on using Shi-Tomasi corner detection Method'''
    gray= DetectionsOpenCV.contrast_image(frame) #Contrasts image for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Changes to greyscale
    
    corners = cv2.goodFeaturesToTrack(gray, BestCorners=20, MinimumQuality0to1=0.01, minEuclideandist=65) #N best corners from image, minimum quality from 0-1, min euc distance between corners
    for corner in corners:
        # corner is array with x,y vals inside another array.
        x, y = corner.ravel()  # removes interior arrays.
        #cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # -1 fills the circle - used to show image
    return corners

def find_color(cell):
    # Convert to HSV
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

    # Define color ranges for detection
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # Threshold the HSV image to get only blue and purple colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Check the presence of colors
    if np.any(blue_mask):
        return 'O'  # Blue for 'O'
    elif np.any(purple_mask):
        return 'X'  # Purple for 'X'
    else:
        return None

def find_centers(image, lower_blue, upper_blue, lower_purple, upper_purple):
    # Convert the image to RGB color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for blue and purple colors
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    purple_mask = cv2.inRange(image_hsv, lower_purple, upper_purple)

    # Find contours for the blue blocks
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find contours for the purple blocks
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of each blue contour
    blue_centers = []
    for cnt in blue_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            blue_centers.append((cX, cY))

    # Calculate the center of each purple contour
    purple_centers = []
    for cnt in purple_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            purple_centers.append((cX, cY))

    return blue_centers, purple_centers

def get_board_template(frame):
    '''Finds Board Coordinates using Circle Detection'''
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Changes image to black and white colour space.
    circle_coords=DetectionsOpenCV.find_circles(frame) #Finds circle coordinates using find_Circles
    circle_coords=circle_coords.reshape(-1,3) #Resize array
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
    sorted_groups=sorted_groups[:,:-1]
    print("sorted",sorted_groups)

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
    

#########################################################
    #Read image
if __name__ == '__main__':
    img = cv2.imread('/Users/ben/Downloads/Test.jpeg')

    #define HSV value ranges
    lower_blue = np.array([50, 200, 200])
    upper_blue = np.array([107, 255, 255])
    lower_purple = np.array([100, 200, 190])
    upper_purple = np.array([150, 255, 200])

    blue_centers, purple_centers = find_centers(img, lower_blue, upper_blue, lower_purple, upper_purple)#works!

    for center in blue_centers:
        cv2.circle(img, center, radius = 5, color = (0,255,0), thickness = -1)#works!

    for center in purple_centers:
        cv2.circle(img, center, radius = 5, color = (0,255,0), thickness = -1)#works!

    corners = find_grid(img)#works!

    for corner in corners:
        x,y = corner.ravel()
        print(x,y) #works!


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, board_thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    state = get_board_template(board_thresh)
    print(state)


    #now test find_color
    # Example coordinates for a cell, replace with actual coordinates of interest
    x, y, w, h = (526, 628, 520, 502)
    cell_roi = img[y:y+h, x:x+w]  # Extract the cell from the image
    result = find_color(cell_roi)
    print(f"The cell contains: {result}")


    '''cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

                #board = draw_shape(board, shape, (x, y, w, h))
                #this function draws an X or O on TEMPLATE, which is the gamestate array. Find_color gets what
                # cell is X or O (purple or blue) eventually, instead of drawing a shape, we want it to place
                # a block. Now it already assigns its next move, so we want
        # Top row
    
'''
def get_board_template(frame):
    """Returns 3 x 3 grid, a.k.a the board using contoured B-Box"""
    # Find grid's center cell, and based on it fetch
    # the other eight cells
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #middle_center = detections.contoured_bbox(thresh) #This is messing up.
    #center_x, center_y, width, height = middle_center
    #print("middle center= ", middle_center)
    center_x,center_y=find_circles(frame)
    width=100
    height=100
    middle_center = center_x,center_y,width,height
    #cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # -1 fills the circle

    # Useful coords
    left = center_x - width
    right = center_x + width
    top = center_y - height
    bottom = center_y + height

    # Middle row
    middle_left = (left, center_y, width, height)
    middle_right = (right, center_y, width, height)
    # Top row
    top_left = (left, top, width, height)
    top_center = (center_x, top, width, height)
    top_right = (right, top, width, height)
    # Bottom row
    bottom_left = (left, bottom, width, height)
    bottom_center = (center_x, bottom, width, height)
    bottom_right = (right, bottom, width, height)
    # Grid's coordinates

    corners = find_grid(frame)#works!

    for corner in corners:
        x,y = corner.ravel()
        #print(x,y) #works!

    return [top_left, top_center, top_right,
            middle_left, middle_center, middle_right,
            bottom_left, bottom_center, bottom_right]
'''