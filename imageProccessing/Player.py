'''
""Game of tic tac toe using OpenCV to play against computer"""


import os
import sys
import cv2
import argparse
import numpy as np

#from keras.models import load_model

import imageProccessing.imutils as imutils
import imageProccessing.detections as detections
from imageProccessing.alphabeta import Tic, get_enemy, determine


#def parse_arguments(argv):
#    parser = argparse.ArgumentParser()
#
#    parser.add_argument('cam', type=int,
#                       help='USB camera for video streaming')
#    parser.add_argument('--model', '-m', type=str, default='data/model.h5',
#                        help='model file (.h5) to detect Xs and Os')#
#
#    return parser.parse_args()


def find_board(frame, add_margin=True):
    """Detect the coords of the sheet of board the game will be played on"""
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    stats = detections.find_corners(thresh)
    # First point is center of coordinate system, so ignore it
    # We only want board's corners
    corners = stats[1:, :2]
    corners = imutils.order_points(corners)
    # Get bird view of game board
    board = imutils.four_point_transform(frame, corners)
    if add_margin:
        board = board[10:-10, 10:-10]
    return board, corners

def find_circles(frame, dp=1.2, min_dist=100, param1=100, param2=30, min_radius=10, max_radius=0):
    """
    From ChatGPT
    Detect circles in an image using Hough Circle Transform.

    :param image: Input image in which to detect circles.
    :param dp: Inverse ratio of the accumulator resolution to the image resolution.
    :param min_dist: Minimum distance between the centers of the detected circles.
    :param param1: First method-specific parameter. In case of Hough Gradient, it is the higher threshold of the two passed to the Canny edge detector.
    :param param2: Second method-specific parameter. In case of Hough Gradient, it is the accumulator threshold for the circle centers at the detection stage.
    :param min_radius: Minimum circle radius.
    :param max_radius: Maximum circle radius.
    :return: A list of circles found, each represented as (x, y, radius).
    """
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
    circles = np.round(circles[0, :]).astype("int")[0]
    #returns first circles coordinates
    print("circle numbers", circles)
    first_circle_coords = circles[0:2]
    cv2.circle(frame, first_circle_coords, 10, (0, 0, 0), 2)
    #cv2.imshow('original', frame)
    #cv2.waitKey(0)
  

    return first_circle_coords


def find_shape(cell):
    #"""Is shape and X or an O?"""
    #mapper = {0: None, 1: 'X', 2: 'O'}
    #cell = detections.preprocess_input(cell)
    #idx = np.argmax(model.predict(cell))
    return 'O'


def get_board_template(frame):
    """Returns 3 x 3 grid, a.k.a the board"""
    # Find grid's center cell, and based on it fetch
    # the other eight cells
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    middle_center = detections.contoured_bbox(thresh)
    center_x, center_y, width, height = middle_center
    cv2.circle(frame, (center_x,center_y), 10, (0, 255, 0), 2)
    cv2.imshow("image",frame)
    cv2.waitKey(0)

    # Useful coords
    left = center_x - width
    right = center_x + width
    top = center_y - height
    bottom = center_y + height

    # Middle row
    middle_left = (left, center_y, width, height)
    top_left = (left, top, width, height)
    top_center = (center_x, top, width, height)
    top_right = (right, top, width, height)
    # Bottom row
    bottom_left = (left, bottom, width, height)
    bottom_center = (center_x, bottom, width, height)
    bottom_right = (right, bottom, width, height)

    # Grid's coordinates
    return [top_left, top_center, top_right,
            middle_left, middle_center, middle_right,
            bottom_left, bottom_center, bottom_right]


def draw_shape(template, shape, coords):
    """Draw on a cell the shape which resides in it"""
    x, y, w, h = coords
    if shape == 'O':
        centroid = (x + int(w / 2), y + int(h / 2))
        cv2.circle(template, centroid, 10, (0, 0, 0), 2)
    elif shape == 'X':
        # Draws the 'X' shape
        cv2.line(template, (x + 10, y + 7), (x + w - 10, y + h - 7),
                 (0, 0, 0), 2)
        cv2.line(template, (x + 10, y + h - 7), (x + w - 10, y + 7),
                 (0, 0, 0), 2)
    return template


def play(vcap):
    """Play tic tac toe game with computer that uses the alphabeta algorithm"""
    # Initialize opponent (computer)
    board = Tic()
    history = {}
    message = True
    # Start playing
    while True:
        ret, frame = vcap.read()
        key = cv2.waitKey(1) & 0xFF
        if not ret:
            print('[INFO] finished video processing')
            break

        # Stop
        if key == ord('q'):
            print('[INFO] stopped video processing')
            break

        # Preprocess input
        # frame = imutils.resize(frame, 500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
        board, corners = find_board(frame, thresh)
        # Four red dots must appear on each corner of the board,
        # otherwise try moving it until they're well detected
        for c in corners:
            cv2.circle(frame, tuple(c), 2, (0, 0, 255), 2)

        # Now working with 'board' to find grid
        board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        _, board_thresh = cv2.threshold(
            board_gray, 170, 255, cv2.THRESH_BINARY_INV)
        grid = get_board_template(board_thresh)

        # Draw grid and wait until user makes a move
        for i, (x, y, w, h) in enumerate(grid):
            cv2.rectangle(board, (x, y), (x + w, y + h), (0, 0, 0), 2)
            if history.get(i) is not None:
                shape = history[i]['shape']
                board = draw_shape(board, shape, (x, y, w, h))

        # Make move
        if message:
            print('Make move, then press spacebar')
            message = False
        if not key == 32:
            cv2.imshow('original', frame)
            cv2.imshow('bird view', board)
            continue
        player = 'X'

        # User's time to play, detect for each available cell
        # where has he played
        available_moves = np.delete(np.arange(9), list(history.keys()))
        for i, (x, y, w, h) in enumerate(grid):
            if i not in available_moves:
                continue
            # Find what is inside each free cell
            cell = board_thresh[int(y): int(y + h), int(x): int(x + w)]
            shape = find_shape(cell)
            if shape is not None:
                history[i] = {'shape': shape, 'bbox': (x, y, w, h)}
                board.make_move(i, player)
            board = draw_shape(board, shape, (x, y, w, h))

        # Check whether game has finished
        if board.complete():
            break

        # Computer's time to play
        player = get_enemy(player)
        computer_move = determine(board, player)
        board.make_move(computer_move, player)
        history[computer_move] = {'shape': 'O', 'bbox': grid[computer_move]}
        board = draw_shape(board, 'O', grid[computer_move])

        # Check whether game has finished
        if board.complete():
            break

        # Show images
        cv2.imshow('original', frame)
        # cv2.imshow('thresh', board_thresh)
        cv2.imshow('bird view', board)
        message = True

    # Show winner
    winner = board.winner()
    height = board.shape[0]
    text = 'Winner is {}'.format(str(winner))
    cv2.putText(board, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('bird view', board)
    cv2.waitKey(0) & 0xFF

    # Close windows
    vcap.release()
    cv2.destroyAllWindows()
    return board.winner()


def main(args):
    """Check if everything's okay and start game"""
    # Load model
    #global model
    #assert os.path.exists(args.model), '{} does not exist'
    #model = load_model(args.model)

    # Initialize webcam feed - Vcap is frame is videocapture
    vcap = cv2.VideoCapture(args.cam)
    if not vcap.isOpened():
        raise IOError('could not get feed from cam #{}'.format(args.cam))

    # Announce winner!
    winner = play(vcap)
    print('Winner is:', winner)
    sys.exit()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

'''

import os
import sys
import cv2
import argparse
import numpy as np

#from keras.models import load_model

import imageProccessing.imutils as imutils
import imageProccessing.detections as detections
from imageProccessing.alphabeta import Tic, get_enemy, determine

def find_circles(frame, dp=1.2, min_dist=100, param1=100, param2=30, min_radius=10, max_radius=0):
    """
    From ChatGPT
    Detect circles in an image using Hough Circle Transform.

    :param image: Input image in which to detect circles.
    :param dp: Inverse ratio of the accumulator resolution to the image resolution.
    :param min_dist: Minimum distance between the centers of the detected circles.
    :param param1: First method-specific parameter. In case of Hough Gradient, it is the higher threshold of the two passed to the Canny edge detector.
    :param param2: Second method-specific parameter. In case of Hough Gradient, it is the accumulator threshold for the circle centers at the detection stage.
    :param min_radius: Minimum circle radius.
    :param max_radius: Maximum circle radius.
    :return: A list of circles found, each represented as (x, y, radius).
    """
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
    circles = np.round(circles[0, :]).astype("int")[0]
    #returns first circles coordinates
    print("circle numbers", circles)
    first_circle_coords = circles[0:2]
    #cv2.circle(frame, first_circle_coords, 10, (0, 0, 0), 2)
    #cv2.imshow('original', frame)
    #cv2.waitKey(0)
  

    return first_circle_coords

def find_board(frame, add_margin=True):
    """Detect the coords of the sheet of board the game will be played on"""
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    stats = detections.find_corners(thresh)
    # First point is center of coordinate system, so ignore it
    # We only want board's corners
    corners = stats[1:, :2]
    corners = imutils.order_points(corners)
    # Get bird view of game board
    board = imutils.four_point_transform(frame, corners)
    if add_margin:
        board = board[10:-10, 10:-10]
    return board, corners

def find_grid(frame):
    gray=contrast_image(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("image",gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    corners = cv2.goodFeaturesToTrack(gray, 20, 0.01, 65) #N best corners from image, minimum quality from 0-1, min euc distance between corners
    for corner in corners:
        # corner is array with x,y vals inside another array.
        x, y = corner.ravel()  # removes interior arrays.
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # -1 fills the circle
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
    """Returns 3 x 3 grid, a.k.a the board"""
    # Find grid's center cell, and based on it fetch
    # the other eight cells
    thresh=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    middle_center = detections.contoured_bbox(thresh)
    center_x, center_y, width, height = middle_center
    print("middle center= ", middle_center)

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
    print("middle center= ",middle_center)

    corners = find_grid(frame)#works!

    for corner in corners:
        x,y = corner.ravel()
        print(x,y) #works!
    cv2.imshow('original', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    
    return [top_left, top_center, top_right,
            middle_left, middle_center, middle_right,
            bottom_left, bottom_center, bottom_right]

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
    
