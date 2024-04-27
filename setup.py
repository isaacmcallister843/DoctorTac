#!/usr/bin/env python
import sys
import os
import dvrk
import math
import sys
import rospy
import numpy as np
import PyKDL
import argparse
import time



# Get the absolute paths of the folders
image_processing_path = os.path.abspath("imageProcessing")
trajectory_planning_path = os.path.abspath("trajecPlanning")

# Append the paths to sys.path (modify these paths if your folders are located differently)
sys.path.append(image_processing_path)
sys.path.append(trajectory_planning_path)

import imageProccessing.camera as camera # DVRK camera code
import trajecPlanning.Trajectory_PSM as trajecTools
import imageProccessing.imageProcessingTools as imTools 
import imageProccessing.Player as Player
import imageProccessing.tictactoe as tictactoe
if __name__ == '__main__':
    p = dvrk.psm('PSM1') 
    p.enable()
    p.home()
    print(p.measured_cp().M)

    x_Vec = PyKDL.Vector(-0.0182456,    0.997603,   0.0667555)
    y_Vec = PyKDL.Vector(0.990335,  -0.0272133,    0.136001)
    z_Vec = PyKDL.Vector(0.137492,  -0.0636289,    0.988457)
    baseOre = PyKDL.Rotation(x_Vec,y_Vec,z_Vec)
        
    def TrajecPlanningTest(): 
        TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.04,.09), orientation=baseOre)
        TrajctoryMain.defualtZLayer = -.23
        TrajctoryMain.extensionLayer = -.2
        time.sleep(1)
        TrajctoryMain.returnHomeFree()
        # time.sleep(1)
        TrajctoryMain.pickAndPlace2(pickLocation=(-.08,.026), placeLocation=(-.001,.007))
    
    TrajecPlanningTest() 
    #TrajctoryMain = trajecTools.TrajctoryNode(homeLocation = (-.04,.09), orientation=baseOre)
    #TrajctoryMain.defualtZLayer = -.23
    #TrajctoryMain.moveCoordinateFree((0,0), targetZlayer= -.2)