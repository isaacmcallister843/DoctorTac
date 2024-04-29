import cv2 as cv
import numpy as np
import glob
from scipy import linalg
import matplotlib.pyplot as plt

# The purpose of this file is to do stereo calibration,
# and eventually use triagnulation (DLT) to project the 2d coord to 3d

#stereoCalibration
def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):

    #go through all the images in folder and put in array of cv images
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #------------------------------------------------------------------------
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    rows = 5 #checkerboard rows
    columns = 8 # checkerboard columns
    world_scaling = 15. #checkerboard square size (mm)
 
    #setup for 3d coordinates
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp    
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #arrays for checkerboard coordinates
    imgpoints_left = [] # 2d points (pixel)
    imgpoints_right = [] # 2d points (pixel)
    objpoints = [] # 3d points (mm)
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
        # cv.imshow('img',frame1)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        #if corners detected are true
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 

    #debug
    # print(objpoints)
    # print(imgpoints_left)
    # print(imgpoints_right)
    # print(width, height)
    # print(criteria)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, _, dist1, _, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T

#trangulation
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    _, _, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

if __name__ == '__main__':
    #main code section just used for calibration (getting projection matrices)
    #-------------------------------------------------------

    #camera_left
    mtx1 =  np.array([[1804.718, 0, 580.277],
                   [0, 1804.682, 555.3117],
                   [0, 0, 1]])
    dist1 = np.array([-0.1948, -0.853745, -0.00275, -0.005319, 7.5466])

    #camera_right
    mtx2 =  np.array([[1855.127, 0, 759.1504],
                    [0, 1846.08, 455.815],
                    [0, 0, 1]])
    dist2 = np.array([-0.276, 0.63499, 0.00393, 0.000713, -1.9277])
    
    #stereo calibration
    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, "C:\\Users\\User\\Desktop\\doctor_tic\\imageProccessing\\Images\\*")
    #"C:\\Users\\bobsy\\Documents\\GitHub\\DoctorTac\\imageProccessing\\Images\\*")

    #-----------Projection Matrices-------------------------------
    #Right camera
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
    
    #left camera
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2

    #need to manually save the projection matrices
    print(P1)
    print(P2)

    # #Triangulate points
    # p3ds = []
    # for uv1, uv2 in zip(uvs1, uvs2):
    #     _p3d = DLT(P1, P2, uv1, uv2)
    #     p3ds.append(_p3d)
    # p3ds = np.array(p3ds)


