import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class forwardTrajectory(object):
    
    def __init__(self, target_point_1, target_point_2, target_z_height : float, total_time : float, freqeuncy : float ) -> None:
        
        self.midPoint = [(target_point_1[0] + target_point_2[0])/2, (target_point_1[1] + target_point_2[1])/2, target_z_height]
        self.target_z_height = target_z_height

        self.target_point_1 = target_point_1 
        self.target_point_2 =  target_point_2

        self.total_time = total_time
        self.freqeuncy  = freqeuncy

        # Polynomial Fits 
        self.coefficients = None
        self.coefficients_y = None
        self.x_t = None
        self.y_t = None
        self.x_prime = None
        self.y_prime = None
        self.x_double_prime = None
        self.y_double_prime = None

        self.time_values = np.linspace(0, self.total_time, self.freqeuncy)
        self.generatePoly()


    def path(self, t: float):
        # Calculate trajectory components in 1D (along the line connecting the two points)
        x_t = self.x_t(t)
        x_prime_t = self.x_prime(t)
        x_double_prime_t = self.x_double_prime(t)
        
        # Calculate the Z-axis components using the y functions (since it's vertical motion)
        current_point_z = self.y_t(t)
        current_velocity_z = self.y_prime(t)
        current_acceleration_z = self.y_double_prime(t)
        
        # Calculate the angle between the two points to project the motion in 2D
        delta_x = self.target_point_2[0] - self.target_point_1[0]
        delta_y = self.target_point_2[1] - self.target_point_1[1]
        angle = np.arctan2(delta_y, delta_x)
        
        # Project the 1D motion onto the 2D plane
        current_point_x = np.cos(angle) * x_t + self.target_point_1[0]
        current_velocity_x = np.cos(angle) * x_prime_t
        current_acceleration_x = np.cos(angle) * x_double_prime_t

        current_point_y = np.sin(angle) * x_t + self.target_point_1[1]
        current_velocity_y = np.sin(angle) * x_prime_t
        current_acceleration_y = np.sin(angle) * x_double_prime_t

        # Return the position, velocity, and acceleration vectors
        return [
            [current_point_x, current_point_y, current_point_z],
            [current_velocity_x, current_velocity_y, current_velocity_z],
            [current_acceleration_x, current_acceleration_y, current_acceleration_z]
        ]

    def generatePoly(self): 
        distance = math.sqrt(((self.target_point_2[0] - self.target_point_1[0])**2 + (self.target_point_2[1] - self.target_point_1[1])**2 ))
            
        initial_conditions = np.array([[0], [distance/2], [distance], [0], [0], [0], [0]])
        initial_conditions_y = np.array([[0], [self.target_z_height], [0], [0], [0], [0], [0]])

        A = np.array([
                        [0, 0, 0, 0, 0, 0, 1],
                        [(self.total_time/2)**6, (self.total_time/2)**5, (self.total_time/2)**4, (self.total_time/2)**3, (self.total_time/2)**2, (self.total_time/2)**1, 1],
                        [(self.total_time)**6, (self.total_time)**5, (self.total_time)**4, (self.total_time)**3, (self.total_time)**2, (self.total_time)**1, 1],
                        [0, 0, 0, 0, 0, 1, 0],
                        [6*(self.total_time)**5, 5*(self.total_time)**4, 4*(self.total_time)**3, 3*(self.total_time)**2, 2*(self.total_time), 1, 0],
                        [0, 0, 0, 0, 2, 0, 0],
                        [30*(self.total_time)**4, 20*(self.total_time)**3, 12*(self.total_time)**2, 6*(self.total_time), 2, 0, 0],
                        ])
        A_inv = np.linalg.inv(A)

        coefficents = np.dot(A_inv,initial_conditions)
        coefficents_y = np.dot(A_inv,initial_conditions_y)
        self.coefficents = coefficents
        self.coefficents_y = coefficents_y

        # Are better ways todo this - think this is fastest tho 

        self.x_t = lambda t : coefficents[0][0] * t**6 + coefficents[1][0] * t**5 + coefficents[2][0] * t**4 +  coefficents[3][0] * t**3 + coefficents[4][0] * t**2 + coefficents[5][0] * t + coefficents[6][0]
        self.y_t = lambda t : coefficents_y[0][0] * t**6 + coefficents_y[1][0] * t**5 + coefficents_y[2][0] * t**4 +  coefficents_y[3][0] * t**3 + coefficents_y[4][0] * t**2 + coefficents_y[5][0] * t + coefficents_y[6][0]
        self.x_prime = lambda t: 6*coefficents[0][0]*t**5 + 5*coefficents[1][0]*t**4 + 4*coefficents[2][0]*t**3 + 3*coefficents[3][0]*t**2 + 2*coefficents[4][0]*t + coefficents[5][0]
        self.y_prime = lambda t: 6*coefficents_y[0][0]*t**5 + 5*coefficents_y[1][0]*t**4 + 4*coefficents_y[2][0]*t**3 + 3*coefficents_y[3][0]*t**2 + 2*coefficents_y[4][0]*t + coefficents_y[5][0]
        self.x_double_prime = lambda t: 30*coefficents[0][0]*t**4 + 20*coefficents[1][0]*t**3 + 12*coefficents[2][0]*t**2 + 6*coefficents[3][0]*t + 2*coefficents[4][0]
        self.y_double_prime = lambda t: 30*coefficents_y[0][0]*t**4 + 20*coefficents_y[1][0]*t**3 + 12*coefficents_y[2][0]*t**2 + 6*coefficents_y[3][0]*t + 2*coefficents_y[4][0]

    def generatePoints(self):
        return np.array([self.path(t) for t in self.time_values])
    
    def returnJustPoints(self): 
        return np.array([self.path(t)[0] for t in self.time_values])

    def createPlot(self): 
        arr = self.generatePoints()
             
        # Initialize lists to store the x, y, and z coordinates
        x_coords = []
        y_coords = []
        z_coords = []

        x_vel = []
        y_vel = []
        z_vel = []

        x_acel = []
        y_acel= []
        z_acel = []

        # Extract coordinates from arr
        for point in arr:
            x_coords.append(point[0][0])
            y_coords.append(point[0][1])
            z_coords.append(point[0][2])

            x_vel.append(point[1][0])
            y_vel.append(point[1][1])
            z_vel.append(point[1][2])

            x_acel.append(point[2][0])
            y_acel.append(point[2][1])
            z_acel.append(point[2][2])
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # 3 rows, 3 columns for position, velocity, acceleration

        # Time values, assuming the time steps are uniform


        # Plotting position
        axs[0, 0].plot(self.time_values, x_coords, label='X Position')
        axs[1, 0].plot(self.time_values, y_coords, label='Y Position')
        axs[2, 0].plot(self.time_values, z_coords, label='Z Position')

        # Plotting velocity
        axs[0, 1].plot(self.time_values, x_vel, label='X Velocity')
        axs[1, 1].plot(self.time_values, y_vel, label='Y Velocity')
        axs[2, 1].plot(self.time_values, z_vel, label='Z Velocity')

        # Plotting acceleration
        axs[0, 2].plot(self.time_values, x_acel, label='X Acceleration')
        axs[1, 2].plot(self.time_values, y_acel, label='Y Acceleration')
        axs[2, 2].plot(self.time_values, z_acel, label='Z Acceleration')

        # Setting titles for the first row as an example
        axs[0, 0].set_title('X Position over Time')
        axs[0, 1].set_title('X Velocity over Time')
        axs[0, 2].set_title('X Acceleration over Time')

        # Setting y labels for the first column as an example
        axs[0, 0].set_ylabel('X')
        axs[1, 0].set_ylabel('Y')
        axs[2, 0].set_ylabel('Z')

        # Setting x labels for the bottom row
        for ax in axs[2]:
            ax.set_xlabel('Time')

        # Adding legends to each plot
        for row in axs:
            for ax in row:
                ax.legend()

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
        

    