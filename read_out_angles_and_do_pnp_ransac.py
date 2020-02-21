import numpy as np
import math
import threading
import time
import logging
import cv2 as cv
import matplotlib.pyplot as plt



file_name = 'test.txt'
content_array = []

origin_bs1 = [-1.958483,  0.542299,  3.152727]

it=0

X_pnp= []
Y_pnp= []
Z_pnp= []

X_real= []
Y_real= []
Z_real= []

lighthouse_3d_store = np.zeros(0,dtype=np.float32)
lighthouse_image_projection_store = np.zeros(0,dtype=np.float32)

first_run =True
tvec_est = np.zeros([1,3])
rvec_est = np.zeros([1,3])
it =0
with open(file_name) as f:
    for line in f:
        content_array = np.array(line.split(', '), dtype=np.float32)
        angles = content_array[0:16]
        angles=angles.reshape([8,2])
        position = content_array[16:19]
        #print(angles)

        sensor_distance_width=0.015*1000
        sensor_distance_lenght=0.03*1000

        lighthouse_3d = np.float32([[-sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                                    [sensor_distance_width/2, -sensor_distance_lenght/2, 0],
                                    [-sensor_distance_width/2, sensor_distance_lenght/2, 0],
                                    [sensor_distance_width/2, sensor_distance_lenght/2, 0]])

        lighthouse_image_projection = np.float32([[ math.tan(angles[0,0]) , math.tan(angles[0,1])],
                                                [ math.tan(angles[1,0]) , math.tan(angles[1,1])],
                                                [ math.tan(angles[2,0]) , math.tan(angles[2,1])],
                                                [ math.tan(angles[3,0]) , math.tan(angles[3,1])]])

        lighthouse_image_projection_2 = np.float32([[ math.tan(angles[4,0]) , math.tan(angles[4,1])],
                                                [ math.tan(angles[5,0]) , math.tan(angles[5,1])],
                                                [ math.tan(angles[6,0]) , math.tan(angles[6,1])],
                                                [ math.tan(angles[7,0]) , math.tan(angles[7,1])]])


 
      #  print(lighthouse_image_projection.dtype)

        if first_run:
            lighthouse_3d_store = lighthouse_3d
            lighthouse_image_projection_store = lighthouse_image_projection
        else:
            if it<10:
               # print(np.shape(lighthouse_3d_store))
               # print(np.shape(lighthouse_3d))

                lighthouse_3d_store =np.append(lighthouse_3d_store, lighthouse_3d,axis=0)
                lighthouse_image_projection_store =np.append(lighthouse_image_projection_store,lighthouse_image_projection,axis=0)
            else:
                lighthouse_3d_store=np.delete(lighthouse_3d_store, (0,1,2,3), axis=0)
                lighthouse_3d_store =np.append(lighthouse_3d_store, lighthouse_3d,axis=0)
                lighthouse_image_projection_store=np.delete(lighthouse_image_projection_store, (0,1,2,3), axis=0)
                lighthouse_image_projection_store =np.append(lighthouse_image_projection_store,lighthouse_image_projection,axis=0)

                print(np.shape(lighthouse_image_projection_store))



       # print(lighthouse_3d_store.dtype)

        K = np.float64([[1*1000, 0, 0],
                        [0, 1*1000, 0],
                        [0.0,0.0,1.0]])
        dist_coef = np.zeros(4)





        val ,tvec_est,rvec_est, inliers = cv.solvePnPRansac(lighthouse_3d_store, lighthouse_image_projection_store, K, dist_coef,rvec_est, tvec_est )
    
        print(val, inliers)
       # print(_ret)
        Rmatrix, _ = cv.Rodrigues(rvec_est)

        rotated_vector=tvec_est#np.matmul(np.linalg.inv(Rmatrix),tvec_est)
       # print(rotated_vector)
        #print(position)
        real_vector=np.subtract(position,origin_bs1)
       # print(np.subtract(position,origin_bs1))

        X_pnp.append(rotated_vector[0]/1000)
        Y_pnp.append(rotated_vector[1]/1000)
        Z_pnp.append(rotated_vector[2]/1000)

        X_real.append(real_vector[0])
        Y_real.append(real_vector[1])
        Z_real.append(real_vector[2])
        it=it+1


    #print(lighthouse_3d_store)
    #print(lighthouse_image_projection_store)

    _ret, rvec_est, tvec_est = cv.solvePnP(lighthouse_3d_store, lighthouse_image_projection_store, K, dist_coef,flags=cv.SOLVEPNP_ITERATIVE)
     
    Rmatrix, _ = cv.Rodrigues(rvec_est)

    print(np.matmul(np.linalg.inv(Rmatrix),tvec_est))
    print(X_real[0],Y_real[0],Z_real[0] )
        
    plt.subplot(3,1,1)
    plt.axis([0, 600, -4,4])
    plt.plot(range(0,596),X_pnp, c='r')
    plt.plot(range(0,596),X_real, c='b')

    plt.subplot(3,1,2)
    plt.axis([0, 600, -4,4])

    plt.plot(range(0,596),Y_pnp, c='r')
    plt.plot(range(0,596),Y_real, c='b')

    plt.subplot(3,1,3)
    plt.axis([0, 600, -4,4])

    plt.plot(range(0,596),Z_pnp, c='r')
    plt.plot(range(0,596),Z_real, c='b')

    plt.show()
    
  



