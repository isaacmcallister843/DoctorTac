import Trajectory_Toolox

target_point_1 = (1 ,1, 0)
target_point_2 = (-3, 4, 0)
target_z_height = 1

plan1 = Trajectory_Toolox.forwardTrajectory(target_point_1, target_point_2, target_z_height, 4, 100) 
print(plan1.returnJustPoints())
plan1.createPlot()
