import numpy as np
from vispy.visuals.transforms import MatrixTransform
import math
import threading
import time
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.mem import MemoryElement
import cv2 as cv

logging.basicConfig(level=logging.ERROR)

# Initialize the low-level drivers (don't list the debug drivers)
cflib.crtp.init_drivers(enable_debug_driver=False)

uri = "radio://0/10/2M/E7E7E7E705"

names = [
    [0, 0, 'lighthouse.angle0x'],
    [0, 1, 'lighthouse.angle0y'],
    [1, 0, 'lighthouse.angle0x_1'],
    [1, 1, 'lighthouse.angle0y_1'],
    [2, 0, 'lighthouse.angle0x_2'],
    [2, 1, 'lighthouse.angle0y_2'],
    [3, 0, 'lighthouse.angle0x_3'],
    [3, 1, 'lighthouse.angle0y_3'],
]

lg_block1 = LogConfig(name='RAW', period_in_ms=100)
lg_block1.add_variable('lighthouse.rawAngle0x', 'float')
lg_block1.add_variable('lighthouse.rawAngle0y', 'float')

lg_block2 = LogConfig(name='RAW', period_in_ms=100)
lg_block1.add_variable('lighthouse.rawAngle0x_1', 'float')
lg_block1.add_variable('lighthouse.rawAngle0y_1', 'float')



def read_angles(scf, first, last, angles):
    block = LogConfig(name='RAW', period_in_ms=100)
    for i in range(first, last):
        block.add_variable(names[i][2], 'float')

    with SyncLogger(scf, block) as logger:
        for log_entry in logger:
            data = log_entry[1]

            for i in range(first, last):
                sensor = names[i][0]
                sweep = names[i][1]
                name = names[i][2]

                angles[sensor][sweep] = data[name]

            break


angles = np.zeros([4, 2])

cf = Crazyflie(rw_cache='./cache')
with SyncCrazyflie(uri, cf=cf) as scf:
    read_angles(scf, 0, 4, angles)
    read_angles(scf, 4, 8, angles)

sensor_distance_width=0.015
sensor_distance_lenght=0.03

lighthouse_3d = np.float32([[-sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                            [sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                            [-sensor_distance_width/2, sensor_distance_lenght/2, 0],
                            [sensor_distance_width/2, sensor_distance_lenght/2, 0]])

lighthouse_image_projection = np.float32([[ math.tan(angles[0,0]) , math.tan(angles[0,1])],
                                         [ math.tan(angles[1,0]) , math.tan(angles[1,1])],
                                         [ math.tan(angles[2,0]) , math.tan(angles[2,1])],
                                         [ math.tan(angles[3,0]) , math.tan(angles[3,1])]])
K = np.float64([[1, 0, 0],
                [0, 1, 0],
                [0.0,0.0,1.0]])
dist_coef = np.zeros(4)
_ret, rvec, tvec = cv.solvePnP(lighthouse_3d, lighthouse_image_projection, K, dist_coef,cv.cv2.SOLVEPNP_UPNP)
print(angles)
#print(tvec)
# print(rvec)
Rmatrix, _ = cv.Rodrigues(rvec)
# print(Rmatrix)

rotated_vector= np.matmul(np.linalg.inv(Rmatrix),tvec)
print(rotated_vector)
