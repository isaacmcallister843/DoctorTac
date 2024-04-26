#!/usr/bin/env python

# Author: Ben Heinrichs, Bobsy Narayan
# Date: 2024-03-08
# Edited: 2024-04-25
# Module with Image Processing & Image Detections functions with OpenCV

#Necessary Libraries
import cv2
import numpy as np

def find_circles(frame, dp=1.2, min_dist=100, param1=100, param2=30, min_radius=1, max_radius=100):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    
    # If no circles were found, return an empty list
    if circles is None:
        return []

    # Convert the circle parameters (x, y, radius) to integers
    circles = np.round(circles[0, :, :2]).astype("int")  # Extract only the x, y coordinates
    
    #Testing code to see what values are found:
    #print("circle numbers", circles) #Print circle Coordinates to Terminal
    #first_circle_coords = circles[0:2] #X,Y Coordinates found
    #cv2.circle(frame, first_circle_coords, 10, (0, 0, 0), 2) #Places images for detected circle onto original fram image.
    #cv2.imshow('original', frame) #Show image
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return circles

def find_corners(img):
    """Function to find corners using Haris Corner Detection Software"""
    corners = cv2.cornerHarris(img, InputPixelSize=5, SobelKernelSize=3, HarisDetectorSensitivity=0.1) #Find corners in image.
    corners = cv2.dilate(corners, None) #Morphological Operation to enlarge regions where corners were detected
    #^Comment to ben. How does this work? Would it make it harder to get proper coordinates after dilation occurs?
    
    #Threshold values are applied to all corners. If Value<0, corner undetected.
    corners = cv2.threshold(corners, Threshold=0.01 * corners.max(), DetectedCornerValue=255, Thresh_binary=0)[1] 
    corners = corners.astype(np.uint8) #Converts array to 8-bit unsigned format for OpenCV compatibality.

    #Used to detect connected neighboring components (?)-Ben?
    numberofLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        corners, connectivity=4)
    # For some reason, stats yielded better results for
    # corner detection than centroids. This might have
    # something to do with sub-pixel accuracy.
    # Check issue #10130 on opencv
    return stats


def contoured_bbox(img):
    """Returns bbox of contoured image"""
    #image, contours, hierarchy = cv2.findContours(img, 1, 2)

    # Largest object is whole image,
    # second largest object is the ROI
    contours, hierarchy = cv2.findContours(img, 1, 2)

    #Following code used to show contoured bbox of image
    #cv2.imshow('image_window',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #Test Code - might not be best setup.
    #Will check if contours are detected or return none.
    if len(contours)>1:
        sorted_cntr = sorted(contours, key=lambda cntr: cv2.contourArea(cntr))
        print("contours?")
        print(sorted_cntr)
        return cv2.boundingRect(sorted_cntr[0])
    else:
        print("not enough countours found")
        return None, None, None, None


def preprocess_input(img):
    """Preprocess image to match model's input shape for shape detection"""
    img = cv2.resize(img, (32, 32))
    # Expand for channel_last and batch size, respectively
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32) / 255

def cleanupImage(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #Get Kernel Size
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) #Apply Gaussian Blue
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel) #Morphological closed Filter
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel) #Morphological Open Filter
    return opened