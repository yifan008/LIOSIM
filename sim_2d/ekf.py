#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math as m

from robot_system import *

class Centralized_EKF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration
        
        self.F = np.identity(4)
        self.F[0:2, 2:4] = self.dt * np.identity(2)

        self.G = np.zeros((4, 2))
        self.G[0:2, 0:2] = 0.5 * self.dt**2 * np.identity(2)
        self.G[2:4, 0:2] = self.dt * np.identity(2)

        self.running_time = 0

        self.Wk = np.zeros(shape=(4, 4), dtype=float)
        self.Fk = np.identity(4)

    def prediction(self, t):
        dataset = self.dataset

        start_time = time.time()

        # get nearest index
        idx = int(t / self.dt - 1)

        # extract odometry
        a = -dataset['odometry'][idx]

        # print(a.shape)

        # extract state vector
        r_p = self.robot_system.xyt[0:2]
        r_v = self.robot_system.xyt[2:4]

        # print(r_p.shape, r_v.shape)

        # state prediction
        r_p = r_p + self.dt * r_v + 0.5 * self.dt**2 * a 
        r_v = r_v + self.dt * a

        self.robot_system.xyt[0:2] = r_p
        self.robot_system.xyt[2:4] = r_v

        self.Fk = self.F @ self.Fk

        self.Q = SENSOR_VAR * np.identity(2)

        # covariance prediction
        self.robot_system.cov = self.F @ self.robot_system.cov @ self.F.T + self.G @ self.Q @ self.G.T
    
        end_time = time.time()
        
        self.running_time += (end_time - start_time)
    
    def absolute_observation(self, t):
        dataset = self.dataset
        
        start_time = time.time()

        idx = int(t / self.dt)
        
        bearing = dataset['measurement'][idx]

        r_p = self.robot_system.xyt[0:2]
        r_v = self.robot_system.xyt[2:4]

        # z_hat = m.atan2(r_p[1], r_p[0])          
        z_hat = m.atan2(r_p[0], r_p[1])          
        z = m.atan2(m.sin(bearing), m.cos(bearing))
        
        dz = z - z_hat

        dz = m.atan2(m.sin(dz), m.cos(dz))

        # construct measurement matrix
        H = np.zeros((1, 4))

        # H[0, 0] = - r_p[1] / (r_p[0]**2 + r_p[1]**2)        
        # H[0, 1] = r_p[0] / (r_p[0]**2 + r_p[1]**2)

        H[0, 0] = r_p[1] / (r_p[0]**2 + r_p[1]**2)        
        H[0, 1] = - r_p[0] / (r_p[0]**2 + r_p[1]**2)

        self.Wk = self.Wk + (H @ self.Fk).T @ (H @ self.Fk)

        cov = self.robot_system.cov

        innovation_inv = 1.0 / (H @ cov @ H.T + BEARING_VAR)

        Kalman_gain = cov @ H.T * innovation_inv

        self.robot_system.xyt = self.robot_system.xyt + Kalman_gain[:, 0] * dz
        
        self.robot_system.cov = (np.identity(4) - Kalman_gain @ H) @ cov

        #print(self.robot_system.xyt)
        #print(self.robot_system.cov)
        #print(dataset['gt'][str(1)]['p'][idx])
        #print('----end----')
        end_time = time.time()
        self.running_time += (end_time - start_time)
        
    def save_est(self, t):
        x = self.robot_system.xyt[0]
        y = self.robot_system.xyt[1]
        
        vx = self.robot_system.xyt[2]
        vy = self.robot_system.xyt[3]

        cov = self.robot_system.cov

        self.robot_system.history.append({'x': np.copy(x), 'y': np.copy(y), 'vx': np.copy(vx), 'vy': np.copy(vy), 'cov': np.copy(cov)})

    def run(self):
        # initialize time
        t = self.dt

        while t <= self.duration: 

          # prediction (time propagation) step
          self.prediction(t)
          
          self.absolute_observation(t)

          # save the estimate
          self.save_est(t)

          # update the time
          t = t + self.dt
        
        if PRINT_TIME:
          print('ekf duration: {} / rank: {} / trace: {} \n'.format(self.running_time / self.duration, np.linalg.matrix_rank(self.Wk), np.trace(self.Wk)))
