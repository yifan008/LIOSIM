#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math as m

from robot_system import *

class MSC_EKF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration

        self.x = np.zeros(shape=(4, ), dtype=float)
        self.cov = np.zeros(shape=(4, 4), dtype=float)

        self.x = self.state_transformation_inv(self.robot_system.xyt)

        T = self.Jacobian_trans(self.robot_system.xyt)

        self.cov = T @ self.robot_system.cov @ T.T
        
        self.running_time = 0

        self.Wk = np.zeros(shape=(4, 4), dtype=float)
        self.Fk = np.identity(4)

    def prediction(self, t):
        dataset = self.dataset

        start_time = time.time()

        # get nearest index
        idx = int(t / self.dt - 1)

        # extract odometry
        a = - dataset['odometry'][idx]

        # Jacobian computation
        F, G = self.Jacobian_FG(self.x, a, self.dt)

        self.Fk = F @ self.Fk

        # state prediction
        self.x = self.state_propagation(self.x, a, self.dt)
        self.robot_system.xyt = self.state_transformation(self.x)

        # covariance prediction
        self.Q = SENSOR_VAR * np.identity(2)
        self.cov = F @ self.cov @ F.T + G @ self.Q @ G.T

        INT = self.Jacobian_trans_inv(self.x)

        self.robot_system.cov = INT @ self.cov @ INT.T

        end_time = time.time()
        
        self.running_time += (end_time - start_time)
    
    def absolute_observation(self, t):
        dataset = self.dataset
        
        start_time = time.time()

        idx = int(t / self.dt)
        
        bearing = dataset['measurement'][idx]

        z_hat = m.atan2(m.sin(self.x[1]), m.cos(self.x[1]))
        # z = m.atan2(m.cos(bearing), m.sin(bearing))
        z = m.atan2(m.sin(bearing), m.cos(bearing))

        dz = z - z_hat
        dz = m.atan2(m.sin(dz), m.cos(dz))

        # construct measurement matrix
        H = np.zeros((1, 4))

        H[0, 1] = 1

        self.Wk = self.Wk + (H @ self.Fk).T @ (H @ self.Fk)

        innovation_inv = 1.0 / (H @ self.cov @ H.T + BEARING_VAR)

        Kalman_gain = self.cov @ H.T * innovation_inv

        self.x = self.x + Kalman_gain[:, 0] * dz

        self.cov = (np.identity(4) - Kalman_gain @ H) @ self.cov

        self.robot_system.xyt = self.state_transformation(self.x)
        
        INT = self.Jacobian_trans_inv(self.x)

        self.robot_system.cov = INT @ self.cov @ INT.T

        end_time = time.time()
        self.running_time += (end_time - start_time)

    def Jacobian_trans(self, state):
        state_z = self.state_transformation_inv(state)

        rho = state_z[0]
        beta = state_z[1]
        rho_d = state_z[2]
        beta_d = state_z[3]

        A1 = np.identity(4)

        A1[0, 0] = m.sin(beta)
        A1[0, 1] = m.cos(beta)
        A1[1, 0] = m.cos(beta)
        A1[1, 1] = -m.sin(beta)

        A1[2:4, 2:4] = A1[0:2, 0:2]

        A2 = np.identity(4)

        A2[0, 0] = - rho
        A2[2, 0] = - rho_d
        A2[2, 1] = beta_d
        A2[3, 0] = - beta_d
        A2[3, 1] = - rho_d
        A2 = rho * A2

        M3 = A2 @ A1

        return M3

    def Jacobian_trans_inv(self, state_z):
        rho = state_z[0]
        beta = state_z[1]
        rho_d = state_z[2]
        beta_d = state_z[3]

        A1 = np.identity(4)

        A1[0, 0] = m.sin(beta)
        A1[0, 1] = m.cos(beta)
        A1[1, 0] = m.cos(beta)
        A1[1, 1] = -m.sin(beta)

        A1[2:4, 2:4] = A1[0:2, 0:2]

        A2 = np.identity(4)

        A2[0, 0] = - 1 / rho
        A2[2, 0] = - 1 / rho * rho_d
        A2[2, 1] = - beta_d
        A2[3, 0] = - 1 / rho * beta_d
        A2[3, 1] = rho_d
        A2 = A2 * 1 / rho

        M1 = A1 @ A2

        return M1

    def state_propagation(self, state, input_a, delta_t):
        # (1/rho beta dot(1/rho) dot(beta)) -----> (x y vx vy)
        state_x = self.state_transformation(state)

        state_x[0:2] = state_x[0:2] + delta_t * state_x[2:4] + 0.5 * delta_t**2 * input_a 
        state_x[2:4] = state_x[2:4] + delta_t * input_a

        # (x y vx vy) -----> (1/rho beta dot(1/rho) dot(beta))
        state_z = self.state_transformation_inv(state_x)

        return state_z

    def Jacobian_FG(self, state, a, delta_t):
        rho = state[0]
        beta = state[1]
        rho_d = state[2]
        beta_d = state[3]

        A1 = np.identity(4)

        A1[0, 0] = m.sin(beta)
        A1[0, 1] = m.cos(beta)
        A1[1, 0] = m.cos(beta)
        A1[1, 1] = -m.sin(beta)

        A1[2:4, 2:4] = A1[0:2, 0:2]

        A2 = np.identity(4)

        A2[0, 0] = - 1 / rho
        A2[2, 0] = - 1 / rho * rho_d
        A2[2, 1] = - beta_d
        A2[3, 0] = - 1 / rho * beta_d
        A2[3, 1] = rho_d
        A2 = A2 * 1 / rho

        M1 = A1 @ A2

        M2 = np.identity(4)
        M2[0:2, 2:4] = delta_t * np.identity(2)

        G2 = np.zeros((4, 2))
        G2[0:2, 0:2] = 0.5 * delta_t**2 * np.identity(2)
        G2[2:4, 0:2] = delta_t * np.identity(2)

        state_z = self.state_propagation(state, a, delta_t)

        rho = state_z[0]
        beta = state_z[1]
        rho_d = state_z[2]
        beta_d = state_z[3]

        A1 = np.identity(4)

        A1[0, 0] = m.sin(beta)
        A1[0, 1] = m.cos(beta)
        A1[1, 0] = m.cos(beta)
        A1[1, 1] = -m.sin(beta)

        A1[2:4, 2:4] = A1[0:2, 0:2]

        A2 = np.identity(4)

        A2[0, 0] = - rho
        A2[2, 0] = - rho_d
        A2[2, 1] = beta_d
        A2[3, 0] = - beta_d
        A2[3, 1] = - rho_d
        A2 = rho * A2

        M3 = A2 @ A1

        F = M3 @ M2 @ M1
        G = M3 @ G2

        return F, G

    # (1/rho beta dot(1/rho) dot(beta)) -----> (x y vx vy)
    def state_transformation(self, state):
        rho = state[0]
        beta = state[1]
        rho_d = state[2]
        beta_d = state [3]

        rx = 1.0 / rho * m.sin(beta)
        ry = 1.0 / rho * m.cos(beta)
        vx = 1.0 / rho * (rho_d * m.sin(beta) + beta_d * m.cos(beta))
        vy = 1.0 / rho * (rho_d * m.cos(beta) - beta_d * m.sin(beta))

        return np.array([rx, ry, vx, vy])
    
    # (x y vx vy) -----> (1/rho beta dot(1/rho) dot(beta))
    def state_transformation_inv(self, state):
        rx = state[0]
        ry = state[1]
        vx = state[2]
        vy = state[3]

        rho = 1.0 / m.sqrt( rx**2 + ry**2 )
        beta = m.atan2(rx, ry)
        rho_d = (rx * vx + ry * vy) / ( rx**2 + ry**2 ) 
        beta_d = (ry * vx - rx * vy) / ( rx**2 + ry**2 ) 

        return np.array([rho, beta, rho_d, beta_d])

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
          print('msc duration: {} / rank: {} / trace: {} \n'.format(self.running_time / self.duration, np.linalg.matrix_rank(self.Wk), np.trace(self.Wk)))
