# %%
from racecar.SDRaceCar import SDRaceCar
import numpy as np
import matplotlib.pyplot as plt

track_id = 'Linear'
env = SDRaceCar(render_env = True, track = track_id)
env.reset()

def rotation_matrix(t):
    '''
    Function: Calculate the rotation matrix in 2D
    Input: 
        t: Angle
    '''
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])

def rotation_to_car_frame(t,h,x,y):
    '''
    Function: Convert from world to car frame
    Inputs:
        t: Angle
        h: Reference point
        x: x coord world frame
        y: y coord world frame
    Ouput: Reference position in car frame 
    '''   
    w_R_c = rotation_matrix(t)
    e_x = h[0] - x
    e_y = h[1] - y
    return w_R_c.T @ np.array([e_x,e_y])

def angle_controller(c_p_h, e_p):
    '''
    Function: controller for the angle
    Inputs:
        c_p_h: Reference position in car frame
        e_p: error from the past used for derivative controller
    Output: Angle to turn
    '''
    K_p = 3
    K_d = 0.1
    return K_p * np.arctan2(c_p_h[1],c_p_h[0]) + K_d * (c_p_h[1] - e_p[1])
     

time = 500
e_p = np.zeros((2,time+1))
xy = np.zeros((2,time))
xy_ref = np.zeros((2,time))
for i in range(time):
    env.render()
    x, y, theta, v_x, v_y, omega, h = env.get_observation()
    xy[0,i], xy[1,i] = x, y
    xy_ref[0,i], xy_ref[1,i] = h[0], h[1]
    c_p_h  = rotation_to_car_frame(theta,h,x,y)   # reference point in car frame
    angle = angle_controller(c_p_h, e_p[:,i])
    thrust = 1
    if np.sqrt(v_x**2 + v_y**2) > 5.5:
        thrust = -1
    action = [angle, thrust]  # [steer, thrust]
    env.step(action)
    e_p[:,i+1] = c_p_h

# %%  Plot the trajectory
plt.plot(xy[0,:],xy[1,:], label = 'car trajectory')
plt.plot(xy_ref[0,:], xy_ref[1,:], label = 'reference trajectory')
plt.title('Car vs Reference Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
# %%
