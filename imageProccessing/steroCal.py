import cv2 as cv
import numpy as np
import glob
from scipy import linalg

# The purpose of this file is to do stereo calibration,
# and eventually use triagnulation (DLT) to project the 2d
# coord to 3d
#
# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html



def frame_by_frame(frame1,frame2):
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
    c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)

    if c_ret1 == True and c_ret2 == True:
        corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
        cv.imshow('img', frame1)

        cv.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
        cv.imshow('img2', frame2)
        k = cv.waitKey(0)

        '''objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)'''

        return corners1, corners2

def stereo_calib(c1_images, c2_images, mtx1, dist1, mtx2, dist2):

    
    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.
    
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
    
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
    
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
    
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
    
            cv.drawChessboardCorners(frame1, (5,8), corners1, c_ret1)
            cv.imshow('img', frame1)
    
            cv.drawChessboardCorners(frame2, (5,8), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(0)
    
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    return R, T

#trangulation
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A

    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

if __name__ == '__main__':

    '''#get images from folder
    images_names = glob.glob('Images/*')
    images_names = sorted(images_names)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]
    print(c1_images_names)
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)'''
    
    i = cv.imread('left1.png',0)
    cv.imshow('img',i)
    i2 = cv.imread("\Desktop\Images\1left.png")
    frame_by_frame(i,i2)

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
    R, T = stereo_calib(c1_images, c2_images, mtx1, dist1, mtx2, dist2)

    #get points from both cameras to triangulate
    '''
    uvs1 = [[458, 86], [451, 164], [287, 181],
        [196, 383], [297, 444], [564, 194],
        [562, 375], [596, 520], [329, 620],
        [488, 622], [432, 52], [489, 56]]
 
    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]
    
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
    
    
    frame1 = cv.imread('testing/_C1.png')
    frame2 = cv.imread('testing/_C2.png')
    
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show()
    
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()
    '''


    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
    
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2

    # #Triangulate points
    # p3ds = []
    # for uv1, uv2 in zip(uvs1, uvs2):
    #     _p3d = DLT(P1, P2, uv1, uv2)
    #     p3ds.append(_p3d)
    # p3ds = np.array(p3ds)
    
    #p3ds should be 3d coords 


#mtx1, dist1 = calibrate_camera(images_folder = 'D2/*')
#mtx2, dist2 = calibrate_camera(images_folder = 'J2/*')