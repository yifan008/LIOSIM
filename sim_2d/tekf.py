#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math as m

from robot_system import *

from scipy.optimize import fminbound

class T_EKF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration
        
        self.J = np.array([[0, -1], [1, 0]])

        self.running_time = 0

        self.Wk = np.zeros(shape=(5, 5), dtype=float)
        self.Fk = np.identity(5)

        p = self.robot_system.xyt[0:2]
        psi = self.robot_system.xyt[2]
        pt = self.robot_system.xyt[3:5]

        Aki = np.identity(2)
        if Flag3:
            Aki[0, 0] = 1 / (pt[0] - p[0])
            Aki[1, 0] = - (pt[1] - p[1]) / (pt[0] - p[0])

        Ts = np.identity(5)
        Ts[0:2, 2] = - self.J @ p
        Ts[3:5, 0:2] = -Aki
        Ts[3:5, 2] = -Aki @ self.J @ (pt - p)
        Ts[3:5, 3:5] = Aki

        self.cov = Ts @ self.robot_system.cov @ Ts.T

    def prediction(self, t):
        dataset = self.dataset

        start_time = time.time()

        # get nearest index
        idx = int(t / self.dt - 1)

        # extract odometry
        u_ctl = dataset['odometry'][idx]

        v = u_ctl['v']
        w = u_ctl['w']

        # extract state vector
        p = self.robot_system.xyt[0:2]
        psi = self.robot_system.xyt[2]
        pt = self.robot_system.xyt[3:5]

        self.F = np.identity(5)
        self.F[0:2, 2] = self.J @ rot_mtx(psi) @ v * self.dt # TODO

        self.G = np.zeros((5, 3))
        self.G[0:2, 0:2] = self.dt * rot_mtx(psi)
        self.G[2, 2] = self.dt

        self.Q = np.identity(3)
        self.Q[0:2, 0:2] = self.Q[0:2, 0:2] * VT_SIGMA**2
        self.Q[2, 2] = self.Q[2, 2] * WT_SIGMA**2

        Ak = np.identity(2)
        if Flag3:
            Ak[0:2, 0] = pt - p

        Ti = np.identity(5)
        Ti[0:2, 2] = self.J @ p
        Ti[3:5, 0:2] = np.identity(2)
        Ti[3:5, 2] = self.J @ pt
        Ti[3:5, 3:5] = Ak

        # state prediction
        p = p + self.dt * rot_mtx(psi) @ v
        psi = psi + self.dt * w

        Aki = np.identity(2)
        if Flag3:
            Aki[0, 0] = 1 / (pt[0] - p[0])
            Aki[1, 0] = - (pt[1] - p[1]) / (pt[0] - p[0])

        Ts = np.identity(5)
        Ts[0:2, 2] = - self.J @ p
        Ts[3:5, 0:2] = -Aki
        Ts[3:5, 2] = -Aki @ self.J @ (pt - p)
        Ts[3:5, 3:5] = Aki

        # print(Ts)

        self.robot_system.xyt[0:2] = p
        self.robot_system.xyt[2] = psi
        self.robot_system.xyt[3:5] = pt

        self.F = Ts @ self.F @ Ti
        self.G = Ts @ self.G
 
        self.Fk = self.F @ self.Fk

        # print(self.F)

        # covariance prediction
        self.cov = self.F @ self.cov @ self.F.T + self.G @ self.Q @ self.G.T
    
        Ak = np.identity(2)
        if Flag3:
            Ak[0:2, 0] = pt - p

        Ti = np.identity(5)
        Ti[0:2, 2] = self.J @ p
        Ti[3:5, 0:2] = np.identity(2)
        Ti[3:5, 2] = self.J @ pt
        Ti[3:5, 3:5] = Ak

        self.robot_system.cov = Ti @ self.cov @ Ti.T

        end_time = time.time()
        
        self.running_time += (end_time - start_time)

    def absolute_observation(self, t):
        dataset = self.dataset
        
        start_time = time.time()

        idx = int(t / self.dt)
        
        bearing = dataset['measurement'][idx]

        p = self.robot_system.xyt[0:2]
        pt = self.robot_system.xyt[3:5]

        r_p = pt - p

        z_hat = m.atan2(r_p[1], r_p[0])
        z = m.atan2(m.sin(bearing), m.cos(bearing))
        
        dz = z - z_hat
        dz = m.atan2(m.sin(dz), m.cos(dz))

        # construct measurement matrix
        H = np.zeros((1, 5))

        H[0, 0:2] = r_p.T @ self.J / (r_p[0]**2 + r_p[1]**2)        
        H[0, 3:5] = - r_p.T @ self.J / (r_p[0]**2 + r_p[1]**2)
        H[0, 2] = -1

        Ak = np.identity(2)
        if Flag3:
            Ak[0:2, 0] = pt - p

        Ti = np.identity(5)
        Ti[0:2, 2] = self.J @ p
        Ti[3:5, 0:2] = np.identity(2)
        Ti[3:5, 2] = self.J @ pt
        Ti[3:5, 3:5] = Ak

        H = H @ Ti

        # print(H)
        
        self.Wk = self.Wk + (H @ self.Fk).T @ (H @ self.Fk)

        cov = self.cov

        innovation_inv = 1.0 / (H @ cov @ H.T + BEARING_VAR)

        Kalman_gain = cov @ H.T * innovation_inv

        self.cov = (np.identity(5) - Kalman_gain @ H) @ cov

        Ak = np.identity(2)
        if Flag3:
            Ak[0:2, 0] = pt - p

        Ti = np.identity(5)
        Ti[0:2, 2] = self.J @ p
        Ti[3:5, 0:2] = np.identity(2)
        Ti[3:5, 2] = self.J @ pt
        Ti[3:5, 3:5] = Ak

        self.robot_system.cov = Ti @ self.cov @ Ti.T

        self.robot_system.xyt = self.robot_system.xyt + Ti @ Kalman_gain[:, 0] * dz

        end_time = time.time()
        self.running_time += (end_time - start_time)

    def save_est(self, t):
        px = self.robot_system.xyt[0]
        py = self.robot_system.xyt[1]
        
        psi = self.robot_system.xyt[2]

        ptx = self.robot_system.xyt[3]
        pty = self.robot_system.xyt[4]

        cov = self.robot_system.cov

        self.robot_system.history.append({'px': np.copy(px), 'py': np.copy(py), 'psi': np.copy(psi), 'ptx': np.copy(ptx), 'pty': np.copy(pty), 'cov': np.copy(cov)})

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
          print('tekf duration: {} / rank: {} / trace: {} \n'.format(self.running_time / self.duration, np.linalg.matrix_rank(self.Wk), np.trace(self.Wk)))
