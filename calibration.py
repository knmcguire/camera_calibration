#!/usr/bin/env python

'''
Planar augmented reality
==================
This sample shows an example of augmented reality overlay over a planar object
tracked by PlaneTracker from plane_tracker.py. solvePnP function is used to
estimate the tracked object location in 3d space.
video: http://www.youtube.com/watch?v=pzVbhxx6aog
Usage
-----
plane_ar.py [<video source>]
Keys:
   SPACE  -  pause video
   c      -  clear targets
Select a textured planar object to track by drawing a box with a mouse.
Use 'focal' slider to adjust to camera focal length for proper video augmentation.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math


sensor_distance_width=0.015
sensor_distance_lenght=0.03

lighthouse_sweep_angles =  np.float32(
    [[-0.15976623, -0.52780449],
    [-0.16472188, -0.52778876],
    [-0.15795201, -0.52006865],
    [-0.16278201, -0.51993513]]
    )

lighthouse_3d = np.float32([[-sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                            [sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                            [-sensor_distance_width/2, sensor_distance_lenght/2, 0],
                            [sensor_distance_width/2, sensor_distance_lenght/2, 0]])

lighthouse_image_projection = np.float32([[ math.tan(lighthouse_sweep_angles[0,0]) , math.tan(lighthouse_sweep_angles[0,1])],
                                         [ math.tan(lighthouse_sweep_angles[1,0]) , math.tan(lighthouse_sweep_angles[1,1])],
                                         [ math.tan(lighthouse_sweep_angles[2,0]) , math.tan(lighthouse_sweep_angles[2,1])],
                                         [ math.tan(lighthouse_sweep_angles[3,0]) , math.tan(lighthouse_sweep_angles[3,1])]])


K = np.float64([[1, 0, 0],
                [0, 1, 0],
                [0.0,0.0,1.0]])

dist_coef = np.zeros(4)

_ret, rvec, tvec = cv.solvePnP(lighthouse_3d, lighthouse_image_projection, K, dist_coef, flags=cv.cv2.SOLVEPNP_ITERATIVE)


print(tvec)
print(rvec)

Rmatrix, _ = cv.Rodrigues(rvec)
print(Rmatrix)

vector = np.matmul(np.linalg.inv(Rmatrix),tvec)
print(vector)
#vector = np.matmul(Rmatrix,tvec)
#print(vector)

#Rt = np.append(Rmatrix,tvec,axis=1)
#P= np.matmul(K,Rt)

#vector = np.matmul(Rmatrix,tvec)

#one_sensor=[ math.tan(lighthouse_sweep_angles[0,0]) , math.tan(lighthouse_sweep_angles[0,1]),0]
#print(P)

#vector = np.matmul(np.linalg.inv(P),one_sensor)

