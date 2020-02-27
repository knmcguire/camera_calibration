# -*- coding: utf-8 -*-

# Calculating position from one bs only
# Unified coordinates, base station geometry in CF coordinate system

import math
import numpy as np
import threading
import logging

from vispy import scene
from vispy.scene import XYZAxis, LinePlot, TurntableCamera, Markers
from vispy import app
import matplotlib.pyplot as plt

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import LighthouseBsGeometry

import cv2 as cv

class Visualizer:
    def __init__(self):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600),
                                        show=True)                            
        view = self.canvas.central_widget.add_view()
        view.bgcolor = '#ffffff'
        view.camera = TurntableCamera(fov=10.0, distance=40.0, up='+z',
                                      center=(0.0, 0.0, 1.0))
        XYZAxis(parent=view.scene)
        self.scene = view.scene

    def marker(self, pos, color='black', size=8, prev=None):
        if prev is None:
            return Markers(pos=np.array(pos, ndmin=2), face_color=color,
                           size=size, parent=self.scene)

        prev.set_data(pos=np.array(pos, ndmin=2), face_color=color,
                      size=size)
        prev.parent = self.scene
        return prev

    def lines(self, points, color='black', prev=None):
        if prev is None:
            return LinePlot(points, color=color, marker_size=0.0, parent=self.scene)

        prev.set_data(points, color=color, marker_size=0.0)
        prev.parent=self.scene
        return prev

    def line(self, a, b, color='black', prev=None):
        return self.lines([a, b], color, prev=prev)

    def body_orientation(self, center, rot_mat, length=0.3, prev=None):
        xv = np.dot(rot_mat, np.array([length, 0.0, 0.0]))
        yv = np.dot(rot_mat, np.array([0.0, length, 0.0]))
        zv = np.dot(rot_mat, np.array([0.0, 0.0, length]))

        prev_l = prev
        if prev is None:
            prev_l = [None, None, None]

        prev_l[0] = self.line(center, center + xv, color="red", prev=prev_l[0])
        prev_l[1] = self.line(center, center + yv, color="green", prev=prev_l[1])
        prev_l[2] = self.line(center, center + zv, color="blue", prev=prev_l[2])

        return prev_l

    def run(self):
        self.canvas.app.run()

    def run_timer(self, interval, callback):
        timer = app.Timer(interval=interval, connect=callback, iterations=-1, start=True, app=self.canvas.app)
        timer.start()
        self.run()

    def remove(self, visual):
        visual.parent = None


class Marker:
    def __init__(self, color='black', size=8):
        self.color = color
        self.marker = None
        self.size = size

    def visualize(self, pos, visualizer):
        if pos is None:
            if self.marker is not None:
                self.marker.parent = None
        else:
            if self.marker is None:
                self.marker = visualizer.marker(pos, color=self.color, size=self.size)
            else:
                self.marker = visualizer.marker(pos, color=self.color, size=self.size, prev=self.marker)


class Line:
    def __init__(self, color='black'):
        self.color = color
        self.line = None

    def visualize(self, a, b, visualizer):
        if a is None or b is None:
            if self.line is not None:
                self.line.parent = None
        else:
            if self.line is None:
                self.line = visualizer.line(a, b, color=self.color)
            else:
                self.line = visualizer.line(a, b, color=self.color, prev=self.line)

class Deck:
    def __init__(self):
        self.pos = np.array((0.0, 0.0, 0.0))
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.r = self._rotation_matrix(self.roll, self.pitch, self.yaw)

        self.vis_body_or = None
        self.vis_marker = Marker(color='red')
        self.vis_sensors = [
            Marker(color='red', size=4),
            Marker(color='red', size=4),
            Marker(color='red', size=4),
            Marker(color='red', size=4),
        ]

        w = 0.015 / 2
        l = 0.030 / 2
        self.sensor_pos = np.array([
            [-l, w, 0.0],
            [-l, -w, 0.0],
            [l, w, 0.0],
            [l, -w, 0.0],
        ])

    def update_data(self, coms):
        self.roll = coms.roll
        self.pitch = coms.pitch
        self.yaw = coms.yaw
        self.pos[0] = coms.x
        self.pos[1] = coms.y
        self.pos[2] = coms.z

        self.r = self._rotation_matrix(self.roll, self.pitch, self.yaw)

    def visualize(self, visualizer):
        self.vis_marker.visualize(self.pos, visualizer)
        self.vis_body_or = visualizer.body_orientation(self.pos, self.r, prev=self.vis_body_or)

        self.visualize_sensor(0, visualizer)
        self.visualize_sensor(1, visualizer)
        self.visualize_sensor(2, visualizer)
        self.visualize_sensor(3, visualizer)


    def visualize_sensor(self, s, visualizer):
        pos = self.sensor_point(s)
        self.vis_sensors[s].visualize(pos, visualizer)

    def _rotation_matrix(self, roll, pitch, yaw):
        # http://planning.cs.uiuc.edu/node102.html
        # Pitch reversed compared to page above
        cg = math.cos(roll)
        cb = math.cos(-pitch)
        ca = math.cos(yaw)
        sg = math.sin(roll)
        sb = math.sin(-pitch)
        sa = math.sin(yaw)

        r = [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ]

        return np.array(r)

    def rotate(self, vec):
        return np.dot(self.r, vec)

    def calc_projected_sensor(self, sensor, normal):
        n_unit = normal / np.linalg.norm(normal)

        # The vector to the sensors
        s = self.rotate(self.sensor_pos[sensor])

        # Project the vector on the plane defined by normal
        # https://en.wikipedia.org/wiki/Vector_projection
        dist_along_normal = np.dot(s, n_unit)
        a1 = n_unit * dist_along_normal
        return s - a1

    def intersection(self, ray):
        # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        # n = p01 x p02
        n = self.rotate(np.array([0, 0, 1]))
        la = ray.pos
        lab = ray.vec
        p0 = self.pos

        if la is None or lab is None or p0 is None:
            return None

        # TODO Check that the solution exists

        t = np.dot(n, la - p0) / np.dot(-lab, n)
        ip = la + lab * t
        return ip

    def sensor_point(self, sensor):
        return self.pos + self.rotate(self.sensor_pos[sensor])


class Basestation:
    def __init__(self, index):
        self.index = index
        self.pos = None
        self.r = None
        self.is_geo_set = False

        self.vis_body_or = None
        self.vis_marker = Marker(color='green')

    def update_data(self, coms):
        if coms.got_geo_data and not self.is_geo_set:
            self.pos = coms.geo_data[self.index].origin
            self.r = coms.geo_data[self.index].rotation_matrix

            self.is_geo_set = True

    def visualize(self, visualizer):
        self.vis_marker.visualize(self.pos, visualizer)

        if self.r is not None:
            self.vis_body_or = visualizer.body_orientation(self.pos, self.r, prev=self.vis_body_or)

    def rotate(self, mat):
        if self.r is not None:
            return np.dot(self.r, mat)

        return mat


class Ray:
    def __init__(self, basestation, sensor):
        self.sensor = sensor
        self.basestation = basestation

        self.angle = np.array([0.0, 0.0])

        self.pos = None
        self.vec = None

        self.vis_line = Line(color='gray')

    def update_data(self, coms):
        self.pos = self.basestation.pos

        sensor_data = coms.angle[self.basestation.index][self.sensor]
        for axis in range(0, 2):
            self.angle[axis] = sensor_data[axis]

        if (self.angle[0] != 0) or (self.angle[1] != 0):
            self.vec = self.calc_ray(self.angle[0], self.angle[1])
        else:
            self.vec = None

    def visualize(self, visualizer):
        length = 7
        if self.vec is not None and self.pos is not None:
            end = self.pos + self.vec * length
            self.vis_line.visualize(self.pos, end, visualizer)
        else:
            self.vis_line.visualize(None, None, visualizer)

    def calc_ray(self, ax, ay):
        n1 = np.array([math.sin(ax), -math.cos(ax), 0.0])
        n2 = np.array([-math.sin(ay), 0.0, math.cos(ay)])
        n21 = np.cross(n2, n1)
        normal = n21 / np.linalg.norm(n21)

        return self.basestation.rotate(normal)




class Setup:
    def __init__(self, basestations, rays, deck, estimator):
        self.bss = basestations
        self.rays = rays
        self.deck = deck
        self.estimator = estimator

    def update_data(self, coms):
        self.deck.update_data(coms)
        for bs in self.bss:
            bs.update_data(coms)
        for ray_group in self.rays:
            for ray in ray_group:
                ray.update_data(coms)

    def visualize(self, visualizer):
        try:
            self.estimator.visualize(visualizer)
        except Exception as error:
            print(error)
        for bs in self.bss:
            bs.visualize(visualizer)
        self.deck.visualize(visualizer)
        for ray_group in self.rays:
            for ray in ray_group:
                ray.visualize(visualizer)


class Communicator:
    def __init__(self, uri):
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers(enable_debug_driver=False)

        self.uri = uri

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.x = 0.0
        self.y = -0.3
        self.z = 0.0

        # bs (0-1) ; sensor (0-4), angle (x,y)
        self.angle = np.array([
            [
                [-0.1474195,  -0.36719632],
                [-0.15149319, -0.36709914],
                [-0.145974,   -0.36058375],
                [-0.150142,   -0.36046359],
            ],
            [
                [0.27944592, -0.18743005],
                [0.28004351, -0.19028285],
                [0.27216661, -0.18700157],
                [0.27271748, -0.18992472],
            ]
        ])

        '''self.got_geo_data = True

        bs1 = LighthouseBsGeometry()
        bs1.origin = np.array([-1.9584829807281494, 0.5422989726066589, 3.152726888656616])
        bs1.rotation_matrix = np.array([
            [0.0, 0.0, 1.0, ],
            [0.0, 1.0, 0.0, ],
            [-1.0, 0.0, 0.0, ]
        ])
        bs1.rotation_matrix = np.array([[0.79721498, -0.004274, 0.60368103, ], [0.0, 0.99997503, 0.00708, ], [-0.60369599, -0.005645, 0.79719502, ]])

        bs2 = LighthouseBsGeometry()
        bs2.origin = np.array([1.0623979568481445, -2.563488006591797, 3.1123669147491455])
        bs2.rotation_matrix = np.array([
            [0.0, 0.0, 1.0, ],
            [0.0, 1.0, 0.0, ],
            [-1.0, 0.0, 0.0, ]
        ])
        bs2.rotation_matrix = np.array([[0.018067000433802605, -0.9993360042572021, 0.03164700046181679], [0.7612509727478027, 0.034269001334905624, 0.6475520133972168], [-0.6482059955596924, 0.012392000295221806, 0.7613639831542969]])
          
        self.geo_data = [bs1, bs2]'''
        
        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % self.uri)

        self._cf.open_link(self.uri)
        self.is_connected = True

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)

        lg_pose = LogConfig(name='test', period_in_ms=50)
        lg_pose.add_variable('stabilizer.roll', 'float')
        lg_pose.add_variable('stabilizer.pitch', 'float')
        lg_pose.add_variable('stabilizer.yaw', 'float')
        lg_pose.add_variable('stateEstimate.x', 'float')
        lg_pose.add_variable('stateEstimate.y', 'float')
        lg_pose.add_variable('stateEstimate.z', 'float')

        lg_lh_master = LogConfig(name='master', period_in_ms=50)
        lg_lh_master.add_variable('lighthouse.angle0x', 'float')
        lg_lh_master.add_variable('lighthouse.angle0y', 'float')
        lg_lh_master.add_variable('lighthouse.angle0x_3', 'float')
        lg_lh_master.add_variable('lighthouse.angle0y_3', 'float')
        lg_lh_master2 = LogConfig(name='master2', period_in_ms=50)
        lg_lh_master2.add_variable('lighthouse.angle0x_1', 'float')
        lg_lh_master2.add_variable('lighthouse.angle0y_1', 'float')
        lg_lh_master2.add_variable('lighthouse.angle0x_2', 'float')
        lg_lh_master2.add_variable('lighthouse.angle0y_2', 'float')

        lg_lh_slave = LogConfig(name='slave', period_in_ms=50)
        lg_lh_slave.add_variable('lighthouse.angle1x', 'float')
        lg_lh_slave.add_variable('lighthouse.angle1y', 'float')
        lg_lh_slave.add_variable('lighthouse.angle1x_3', 'float')
        lg_lh_slave.add_variable('lighthouse.angle1y_3', 'float')
        lg_lh_slave2 = LogConfig(name='slave2', period_in_ms=50)
        lg_lh_slave2.add_variable('lighthouse.angle1x_1', 'float')
        lg_lh_slave2.add_variable('lighthouse.angle1y_1', 'float')
        lg_lh_slave2.add_variable('lighthouse.angle1x_2', 'float')
        lg_lh_slave2.add_variable('lighthouse.angle1y_2', 'float')

        self._cf.log.add_config(lg_pose)
        lg_pose.data_received_cb.add_callback(self.data_receivedPose)
        lg_pose.start()

        self._cf.log.add_config(lg_lh_master)
        lg_lh_master.data_received_cb.add_callback(self.data_receivedSensor)
        lg_lh_master.start()

        self._cf.log.add_config(lg_lh_master2)
        lg_lh_master2.data_received_cb.add_callback(self.data_receivedSensor)
        lg_lh_master2.start()

        self._cf.log.add_config(lg_lh_slave)
        lg_lh_slave.data_received_cb.add_callback(self.data_receivedSensor)
        lg_lh_slave.start()

        self._cf.log.add_config(lg_lh_slave2)
        lg_lh_slave2.data_received_cb.add_callback(self.data_receivedSensor)
        lg_lh_slave2.start()

        self.read_geo_data()

    def data_receivedPose(self, timestamp, data, logconf):
        self.roll = math.radians(data['stabilizer.roll'])
        self.pitch = math.radians(data['stabilizer.pitch'])
        self.yaw = math.radians(data['stabilizer.yaw'])
        self.x = data['stateEstimate.x']
        self.y = data['stateEstimate.y']
        self.z = data['stateEstimate.z']

    def data_receivedSensor(self, timestamp, data, logconf):
        self.assign(self.angle, 0, 0, 0, data, 'lighthouse.angle0x')
        self.assign(self.angle, 0, 0, 1, data, 'lighthouse.angle0y')
        self.assign(self.angle, 0, 1, 0, data, 'lighthouse.angle0x_1')
        self.assign(self.angle, 0, 1, 1, data, 'lighthouse.angle0y_1')
        self.assign(self.angle, 0, 2, 0, data, 'lighthouse.angle0x_2')
        self.assign(self.angle, 0, 2, 1, data, 'lighthouse.angle0y_2')
        self.assign(self.angle, 0, 3, 0, data, 'lighthouse.angle0x_3')
        self.assign(self.angle, 0, 3, 1, data, 'lighthouse.angle0y_3')
        self.assign(self.angle, 1, 0, 0, data, 'lighthouse.angle1x')
        self.assign(self.angle, 1, 0, 1, data, 'lighthouse.angle1y')
        self.assign(self.angle, 1, 1, 0, data, 'lighthouse.angle1x_1')
        self.assign(self.angle, 1, 1, 1, data, 'lighthouse.angle1y_1')
        self.assign(self.angle, 1, 2, 0, data, 'lighthouse.angle1x_2')
        self.assign(self.angle, 1, 2, 1, data, 'lighthouse.angle1y_2')
        self.assign(self.angle, 1, 3, 0, data, 'lighthouse.angle1x_3')
        self.assign(self.angle, 1, 3, 1, data, 'lighthouse.angle1y_3')

    def assign(self, angle, bs, sensor, dir, data, log_name):
        if log_name in data:
            angle[bs][sensor][dir] = data[log_name]

    def _connection_failed(self, link_uri, msg):
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

    def read_geo_data(self):
        mems = self._cf.mem.get_mems(MemoryElement.TYPE_LH)
        self.got_geo_data = False

        print('Rquesting geo data')
        mems[0].update(self.mem_updated)

    def mem_updated(self, mem):
        self.geo_data = mem.geometry_data
        self.got_geo_data = True
        mem.dump()

class BaseStationEstimation:
    def __init__(self):
        self.angle = []
        self.vis_marker = Marker(color='orange')
        self.vis_body_or = None
        self.vis_body_or_nr = None

        self.bs1_est_loc  = np.array((0.0, 0.0, 0.0))
        self.first_run = True

    
    def do_estimation(self, coms):
        sensor_distance_width=0.015
        sensor_distance_lenght=0.03

        try:
            lighthouse_3d = np.float32([[-sensor_distance_lenght/2, sensor_distance_width/2, 0],
                    [-sensor_distance_lenght/2, -sensor_distance_width/2, 0],
                    [sensor_distance_lenght/2, sensor_distance_width/2, 0],
                    [sensor_distance_lenght/2, -sensor_distance_width/2, 0]])
            lighthouse_image_projection = np.float32(
                [
                    [ math.tan(coms.angle[0][0][0]) , math.tan(coms.angle[0][0][1])],
                    [ math.tan(coms.angle[0][1][0]) , math.tan(coms.angle[0][1][1])],
                    [ math.tan(coms.angle[0][2][0]) , math.tan(coms.angle[0][2][1])],
                    [ math.tan(coms.angle[0][3][0]) , math.tan(coms.angle[0][3][1])]
                ])
            K = np.float64([[1, 0, 0],
                            [0, 1, 0],
                            [0.0,0.0,1.0]])
            dist_coef = np.zeros(4)

            #tvec_est = np.float32([[ 0.15805522], [-1.28796501], [ 3.38995395]])
            #rvec_est = np.float32([[2.42781741], [2.62646853], [1.43358334]])
            #if self.first_run is True: 
            self.tvec_est = np.float32([[0], [-1], [1]])
            self.rvec_est = np.float32([[0], [0], [0]])
                #self.first_run = False

            _ret, self.rvec_est, self.tvec_est = cv.solvePnP(lighthouse_3d, lighthouse_image_projection, K, dist_coef,flags=cv.SOLVEPNP_ITERATIVE,
                rvec=self.rvec_est,tvec=self.tvec_est, useExtrinsicGuess=True)
            
            

            #_ret, rvec_test, tvec_test,_=cv.solvePnPGeneric(objectPoints=lighthouse_3d, imagePoints= lighthouse_image_projection, cameraMatrix= K, distCoeffs=dist_coef,flags=cv.SOLVEPNP_ITERATIVE)
            #self.rvec_est, self.tvec_est	=	cv.solvePnPRefineLM(	lighthouse_3d, lighthouse_image_projection, K, dist_coef, rvec=self.rvec_est,tvec=self.tvec_est,criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_COUNT, 50,0.0001)	)
            #print(rvec_test.shape)
            
            Rmatrix, _ = cv.Rodrigues(self.rvec_est)
            
            #self.r=np.linalg.inv(Rmatrix)
            rotation = np.linalg.inv(Rmatrix)

            opencv_to_cf = np.array([
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                ])
            cf_to_opencv = np.array([
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                ])

            z_min = np.array([
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
                ])

            y_min= np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0]
                ])
            #self.r = np.dot(opencv_to_cf, rotation)
           
            rotated_vector= np.matmul(rotation,self.tvec_est)

            #euler_angles = self.rotationMatrixToEulerAngles(rotation)
           # rotation_matrix = self._rotation_matrix(euler_angles[0],euler_angles[1],euler_angles[2])
            #print(rotated_vector)
            #if(rotated_vector[2][0]<0 and rotated_vector[0][0]>0):
              #  print(tvec_est)
              #  print(rvec_est)
            #print(tvec_est)
            #print(rvec_est)
            self.bs1_est_loc  = np.array((0.0, 0.0, 0.0))
            self.bs1_est_loc[0] = -rotated_vector[0][0]+coms.x
            self.bs1_est_loc[1] = -rotated_vector[1][0]+coms.y
            self.bs1_est_loc[2] = -rotated_vector[2][0]+coms.z
            #print(np.matmul(np.linalg.inv(z_min), rotation))
            self.r2 = rotation
            #print(rotation.shape)
           # print(rotation)
           #print(np.matmul(z_min,rotation))

            self.r = rotation
            #rot_temp = np.matmul(np.matmul(np.linalg.inv(z_min), rotation),z_min)

            #self.r = np.matmul(rotation,z_min)
            self.r = np.matmul(np.matmul(rotation,z_min),y_min)

                #self.r =  np.matmul(np.matmul(np.linalg.inv(z_min), rotation), z_min)
           # print(rotated_vector)

            #self.r = np.dot(opencv_to_cf, np.dot(rotation, cf_to_opencv))
            #print(rotated_vector)



        except Exception as error:
            print(error)            


    def visualize(self, visualizer):
       # print('estimation vector',self.bs1_est_loc.T)
        self.vis_marker.visualize(self.bs1_est_loc, visualizer)
        #self.vis_body_or_nr = visualizer.body_orientation(self.bs1_est_loc, self.r2, prev=self.vis_body_or_nr)
        self.vis_body_or = visualizer.body_orientation(self.bs1_est_loc, self.r, prev=self.vis_body_or)

    def _rotation_matrix(self, roll, pitch, yaw):
        # http://planning.cs.uiuc.edu/node102.html
        # Pitch reversed compared to page above
        cg = math.cos(roll)
        cb = math.cos(-pitch)
        ca = math.cos(yaw)
        sg = math.sin(roll)
        sb = math.sin(-pitch)
        sa = math.sin(yaw)

        r = [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ]

        return np.array(r)

    def rotationMatrixToEulerAngles(self, R) :
    
        
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])


################################################################

class Main:
    def run(self):
        self.coms = Communicator('radio://0/56/2M/E7E7E7E705')

        self.bss = [Basestation(0), Basestation(1)]

        self.rays = [
            [
                Ray(self.bss[0], 0),
                Ray(self.bss[0], 1),
                Ray(self.bss[0], 2),
                Ray(self.bss[0], 3),
            ],
            [
                Ray(self.bss[1], 0),
                Ray(self.bss[1], 1),
                Ray(self.bss[1], 2),
                Ray(self.bss[1], 3),
            ],
        ]

        self.deck = Deck()
        self.estimator = BaseStationEstimation()

        self.setup = Setup(self.bss, self.rays, self.deck, self.estimator)


        self.visualizer = Visualizer()
        self.visualizer.run_timer(0.1, self.update)

    def update(self, ev):
        self.setup.update_data(self.coms)
        self.estimator.do_estimation(self.coms)
        self.setup.visualize(self.visualizer)



Main().run()
