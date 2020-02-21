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

import keyboard

logging.basicConfig(level=logging.ERROR)

# Initialize the low-level drivers (don't list the debug drivers)
cflib.crtp.init_drivers(enable_debug_driver=False)

uri = "radio://0/56/2M/E7E7E7E705"

names_BS1 = [
    [0, 0, 'lighthouse.angle0x'],
    [0, 1, 'lighthouse.angle0y'],
    [1, 0, 'lighthouse.angle0x_1'],
    [1, 1, 'lighthouse.angle0y_1'],
    [2, 0, 'lighthouse.angle0x_2'],
    [2, 1, 'lighthouse.angle0y_2'],
    [3, 0, 'lighthouse.angle0x_3'],
    [3, 1, 'lighthouse.angle0y_3'],
]

names_BS2 = [
    [0, 0, 'lighthouse.angle1x'],
    [0, 1, 'lighthouse.angle1y'],
    [1, 0, 'lighthouse.angle1x_1'],
    [1, 1, 'lighthouse.angle1y_1'],
    [2, 0, 'lighthouse.angle1x_2'],
    [2, 1, 'lighthouse.angle1y_2'],
    [3, 0, 'lighthouse.angle1x_3'],
    [3, 1, 'lighthouse.angle1y_3'],
]

angles = np.zeros([8,2])
position = np.zeros([1,3])


def read_angles_BS1(scf, first, last, angles):
    block = LogConfig(name='RAW', period_in_ms=100)
    for i in range(first, last):
        block.add_variable(names_BS1[i][2], 'float')

    with SyncLogger(scf, block) as logger:
        for log_entry in logger:
            data = log_entry[1]

            for i in range(first, last):
                sensor = names_BS1[i][0]
                sweep = names_BS1[i][1]
                name = names_BS1[i][2]

                angles[sensor][sweep] = data[name]

            break
    block.delete


def read_angles_BS2(scf, first, last, angles):
    block = LogConfig(name='RAW', period_in_ms=100)
    for i in range(first, last):
        block.add_variable(names_BS2[i][2], 'float')

    with SyncLogger(scf, block) as logger:
        for log_entry in logger:
            data = log_entry[1]

            for i in range(first, last):
                sensor = names_BS2[i][0]
                sweep = names_BS2[i][1]
                name = names_BS2[i][2]

                angles[sensor+4][sweep] = data[name]

            break
    block.delete




def get_position(scf,position):
    block = LogConfig(name='RAW', period_in_ms=100)
    block.add_variable('stateEstimate.x', 'float')
    block.add_variable('stateEstimate.y', 'float')
    block.add_variable('stateEstimate.z', 'float')

    with SyncLogger(scf, block) as logger:
        for log_entry in logger:
            data = log_entry[1]
            print(data)

            position[0,0] =  data['stateEstimate.x']
            position[0,1] =  data['stateEstimate.y']
            position[0,2] =  data['stateEstimate.z']

            break
    block.delete

def cb_logging_bs1_x(timestamp, data, logconf):
    print(data)
    for i in range(0, 8,2):
        sensor = names_BS1[i][0]
        sweep = names_BS1[i][1]
        name = names_BS1[i][2]
        angles[sensor][sweep] = data[name]


def cb_logging_bs1_y(timestamp, data, logconf):
    print(data)
    for i in range(1, 8,2):
        sensor = names_BS1[i][0]
        sweep = names_BS1[i][1]
        name = names_BS1[i][2]
        angles[sensor][sweep] = data[name]

def configure_logging_bs1_x(scf,block):    
    
    block.add_variable('lighthouse.angle0x','float')
    block.add_variable('lighthouse.angle0x_1','float')
    block.add_variable('lighthouse.angle0x_2','float')
    block.add_variable('lighthouse.angle0x_3','float')

    scf.cf.log.add_config(block)
    block.data_received_cb.add_callback(cb_logging_bs1_x)
    block.start()

def configure_logging_bs1_y(scf,block):    
    
    block.add_variable('lighthouse.angle0y','float')
    block.add_variable('lighthouse.angle0y_1','float')
    block.add_variable('lighthouse.angle0y_2','float')
    block.add_variable('lighthouse.angle0y_3','float')

    scf.cf.log.add_config(block)
    block.data_received_cb.add_callback(cb_logging_bs1_y)
    block.start()



def cb_logging_bs2_x(timestamp, data, logconf):
    print(data)

    for i in range(0, 8,2):
        sensor = names_BS2[i][0]
        sweep = names_BS2[i][1]
        name = names_BS2[i][2]
        angles[sensor+4][sweep] = data[name]


def cb_logging_bs2_y(timestamp, data, logconf):
    print(data)

    for i in range(1, 8,2):
        sensor = names_BS2[i][0]
        sweep = names_BS2[i][1]
        name = names_BS2[i][2]
        angles[sensor+4][sweep] = data[name]


def configure_logging_bs2_x(scf,block):    
    
    block.add_variable('lighthouse.angle1x','float')
    block.add_variable('lighthouse.angle1x_1','float')
    block.add_variable('lighthouse.angle1x_2','float')
    block.add_variable('lighthouse.angle1x_3','float')

    scf.cf.log.add_config(block)
    block.data_received_cb.add_callback(cb_logging_bs2_x)
    block.start()

def configure_logging_bs2_y(scf,block):    
    
    block.add_variable('lighthouse.angle1y','float')
    block.add_variable('lighthouse.angle1y_1','float')
    block.add_variable('lighthouse.angle1y_2','float')
    block.add_variable('lighthouse.angle1y_3','float')

    scf.cf.log.add_config(block)
    block.data_received_cb.add_callback(cb_logging_bs2_y)
    block.start()


def cb_logging_pos(timestamp, data, logconf):

    position[0,0] =  data['stateEstimate.x']
    position[0,1] =  data['stateEstimate.y']
    position[0,2] =  data['stateEstimate.z']


def configure_logging_pos(scf,block):
    block.add_variable('stateEstimate.x', 'float')
    block.add_variable('stateEstimate.y', 'float')
    block.add_variable('stateEstimate.z', 'float')

    scf.cf.log.add_config(block)
    block.data_received_cb.add_callback(cb_logging_pos)
    block.start()


#angles = np.zeros([4+4, 2])
#position = np.zeros([3,1])

#angles = np.reshape(np.arange(16),[8,2])


cf = Crazyflie(rw_cache='./cache')



with SyncCrazyflie(uri, cf=cf) as scf:


    log_bs1_x = LogConfig(name='BS1_x', period_in_ms=100)
    log_bs1_y = LogConfig(name='BS1_y', period_in_ms=100)
    
    log_bs2_x = LogConfig(name='BS2_x', period_in_ms=100)
    log_bs2_y = LogConfig(name='BS2_y', period_in_ms=100)
    #log_bs2 = LogConfig(name='BS2', period_in_ms=100)
    log_pos = LogConfig(name='POS', period_in_ms=100)
 
    configure_logging_bs1_x(scf,log_bs1_x)
    configure_logging_bs1_y(scf,log_bs1_y)

    configure_logging_bs2_x(scf,log_bs2_x)
    configure_logging_bs2_y(scf,log_bs2_y)
    #configure_logging_bs2(scf,log_bs2,1)
    configure_logging_pos(scf,log_pos)
    
    

    with open('test.txt','w') as f:
       
        bool_key_is_pressed = False
        while(bool_key_is_pressed!=True):
            bool_key_is_pressed=keyboard.is_pressed('q')
            print(bool_key_is_pressed)
            

            logging_array= np.concatenate((angles.reshape(1,16), position), axis=1)
            print(logging_array)
            np.savetxt(f,logging_array,fmt='%f',delimiter = ', ')
            time.sleep(0.1)
f.close        
        




