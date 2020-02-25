import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import threading
import time
import logging


fig = plt.figure()
ax = fig.gca(projection='3d')

origin_bs1 = [-1.958483,  0.542299,  3.152727]
origin_bs2 = [1.062398,  -2.563488,  3.112367]
origin_0 = [0,0,0]


ax.scatter(origin_0[0], origin_0[1], origin_0[2], c='r', label='origin')
ax.scatter(origin_bs2[0], origin_bs2[1], origin_bs2[2], marker='^', c='c',  label='basestation 2')



tvec_est = np.zeros([1,3])
tvec_est[0,0] = 0
tvec_est[0,1] = -2
tvec_est[0,2] = 3


rmat =np.float32([[0.79721498, -0.004274, 0.60368103],[.0, 0.99997503, 0.00708], [-0.60369599, -0.005645, 0.79719502]])

rvec_est = np.zeros([1,3])
rvec_est,_=cv.Rodrigues(rmat)
print(rvec_est)


file_name = 'test.txt'
content_array = []
with open(file_name) as f:
    for line in f:
        content_array = np.array(line.split(', '), dtype=np.float32)
        angles = content_array[0:16]
        angles=angles.reshape([8,2])
        position = content_array[16:19]
        print(position)
        ax.scatter(position[0], position[1], position[2], marker='^', c='y',  label='origin crazyflie')
        real_vector=np.subtract(origin_bs1,position)
        ax.scatter(real_vector[0], real_vector[1], real_vector[2], marker='^', c='b', label='basestation 1')

        #print(angles)

        sensor_distance_width=0.015
        sensor_distance_lenght=0.03

        lighthouse_3d = np.float32([[-sensor_distance_lenght/2, sensor_distance_width/2, 0],
                                    [-sensor_distance_lenght/2, -sensor_distance_width/2, 0],
                                    [sensor_distance_lenght/2, -sensor_distance_width/2, 0],
                                    [sensor_distance_lenght/2, -sensor_distance_width/2, 0]])

        lighthouse_image_projection = np.float32([[ math.tan(angles[0,0]) , math.tan(angles[0,1])],
                                                [ math.tan(angles[1,0]) , math.tan(angles[1,1])],
                                                [ math.tan(angles[2,0]) , math.tan(angles[2,1])],
                                                [ math.tan(angles[3,0]) , math.tan(angles[3,1])]])

        lighthouse_image_projection_2 = np.float32([[ math.tan(angles[4,0]) , math.tan(angles[4,1])],
                                                [ math.tan(angles[5,0]) , math.tan(angles[5,1])],
                                                [ math.tan(angles[6,0]) , math.tan(angles[6,1])],
                                                [ math.tan(angles[7,0]) , math.tan(angles[7,1])]])


    
        K = np.float64([[1, 0, 0],
                        [0, 1, 0],
                        [0.0,0.0,1.0]])
        dist_coef = np.zeros(4)
  
        
        _ret, rvec, tvec = cv.solvePnP(lighthouse_3d, lighthouse_image_projection, K, dist_coef, rvec=rvec_est, tvec=tvec_est, useExtrinsicGuess=True, flags=cv.SOLVEPNP_ITERATIVE)
       # rvec, tvec = cv.solvePnPRefineLM(lighthouse_3d, lighthouse_image_projection, K, dist_coef,rvec_est, tvec_est)
       # print(tvec)
       # print(rvec)

        print('rvec',rvec)
        print('tvec',tvec)

        Rmatrix, _ = cv.Rodrigues(rvec)
        print(Rmatrix)
        rotated_vector=   np.matmul(np.linalg.inv(Rmatrix),tvec_est.T)

        print(rotated_vector)
        ax.scatter(rotated_vector[0], -rotated_vector[1], -rotated_vector[2], marker='^', c='m', label='estimated_BS_1_pos')
        plt.pause(0.1)
        