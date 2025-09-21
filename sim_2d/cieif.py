#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import math as m

from robot_system import *

from scipy.optimize import fminbound

class CI_EIF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration
        
        self.J = np.array([[0, -1], [1, 0]])

        self.running_time = 0

        self.xyt = np.copy(self.robot_system.xyt)
        self.cov = np.copy(self.robot_system.cov)

        dim = self.xyt.shape[0]

        if dim == 3:
            self.landmark_seq = []
            
        self.Q = np.identity(3)
        self.Q[0:2, 0:2] = self.Q[0:2, 0:2] * VT_SIGMA**2
        self.Q[2, 2] = self.Q[2, 2] * WT_SIGMA**2

    def prediction(self, t):
        dataset = self.dataset

        start_time = time.time()

        # get nearest index
        idx = int(t / self.dt - 1)

        # extract odometry
        u_ctl = dataset['odometry'][idx]

        # print('odometry: {} \n'.format(len(dataset['odometry'])))
        # print('measurement: {} \n'.format(len(dataset['measurement'])))
        # print('gt: {} \n'.format(len(dataset['gt'])))

        v = u_ctl['v']
        w = u_ctl['w']

        # extract state vector
        p = self.xyt[0:2]
        psi = self.xyt[2]

        # dim = self.xyt.shape[0]
        # self.F = np.identity(dim)
        # self.F[0:2, 2] = self.J @ rot_mtx(psi) @ v * self.dt # TODO

        self.F = np.identity(3)
        self.F[0:2, 2] = self.J @ rot_mtx(psi) @ v * self.dt # TODO
        
        self.G = np.zeros((3, 3))
        self.G[0:2, 0:2] = self.dt * rot_mtx(psi)
        self.G[2, 2] = self.dt

        # state prediction
        p = p + self.dt * rot_mtx(psi) @ v
        psi = psi + self.dt * w

        # covariance prediction
        # self.cov = self.F @ self.cov @ self.F.T + self.G @ self.Q @ self.G.T
        self.cov[0:3, 0:3] = self.F @ self.cov[0:3, 0:3] @ self.F.T + self.G @ self.Q @ self.G.T

        self.xyt[0:2] = p
        self.xyt[2] = psi
        
        self.robot_system.xyt = self.xyt[0:3]
        self.robot_system.cov = self.cov[0:3, 0:3]
            
        end_time = time.time()
        
        self.running_time += (end_time - start_time)
    
    def absolute_observation(self, t):
        dataset = self.dataset
        
        start_time = time.time()

        idx = int(t / self.dt)
      
        # print(idx, len(dataset['measurement']))  
        landmark_observation = dataset['measurement'][idx]
        
        # observation
        observation_num = len(landmark_observation)

        if observation_num == 0:
            return
        
        self.landmark_seq_in_state = []
        self.landmark_seq_no_in_state = []

        landmark_observation_id = []
                
        for i in range(len(landmark_observation)):
            if landmark_observation[i]['id'] in self.landmark_seq:
                self.landmark_seq_in_state.append(landmark_observation[i])
            else:
                self.landmark_seq_no_in_state.append(landmark_observation[i])

            landmark_observation_id.append(landmark_observation[i]['id'])

        # print(self.landmark_seq, landmark_observation_id)
            
        self.landmark_seq_no_obs = []

        for i in range(len(self.landmark_seq)):
            if self.landmark_seq[i] in landmark_observation_id:
                pass
            else:
                self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
        dim = self.xyt.shape[0]
        
        # state update
        if len(self.landmark_seq_in_state) > 0:
            observation_num = len(self.landmark_seq_in_state)
            
            SEQ_UPDATE = True

            if SEQ_UPDATE:
                # sequential update
                for i in range(observation_num):
                    idx = self.landmark_seq_in_state[i]['id']
                    ids = self.landmark_seq.index(idx)

                    z_hat = rot_mtx(self.xyt[2]).T @ (self.xyt[ids*2+3:(ids+1)*2+3] - self.xyt[0:2])
                    z = self.landmark_seq_in_state[i]['obs']
                    dz = z - z_hat
                    
                    Hx = np.zeros((2, 5))
                    Hx[:, 2] = - self.J @ (self.xyt[2*ids+3:2*(ids+1)+3] - self.xyt[0:2])
                    Hx[:, 0:2] = - np.identity(2)
                    Hx[:, 3:5] = np.identity(2)
                    Hx = rot_mtx(self.xyt[2]).T @ Hx 
                    
                    info_m = Hx.T @ np.linalg.inv(LIDAR_SIGMA**2 * np.identity(2)) @ (dz + Hx[:, 0:3] @ self.xyt[0:3] + Hx[:, 3:5] @ self.xyt[ids*2+3:(ids+1)*2+3])
                    info_cov_m = Hx.T @ np.linalg.inv(LIDAR_SIGMA**2 * np.identity(2)) @ Hx

                    cov_x = self.cov[0:3, 0:3]
                    cov_f = self.cov[ids*2+3:(ids+1)*2+3, ids*2+3:(ids+1)*2+3]

                    info_cov_x = np.zeros((5, 5))
                    info_cov_x[0:3, 0:3] = np.linalg.inv(cov_x)
                    info_x = np.zeros((5, ))
                    info_x[0:3] = np.linalg.inv(cov_x) @ self.xyt[0:3]
                    
                    info_cov_f = np.zeros((5, 5))
                    info_cov_f[3:5, 3:5] = np.linalg.inv(cov_f)
                    info_f = np.zeros((5, ))    
                    info_f[3:5] = np.linalg.inv(cov_f) @ self.xyt[ids*2+3:(ids+1)*2+3]
                    
                    info_xm = info_x + info_m
                    info_cov_xm = info_cov_x + info_cov_m
                    
                    info_fm = info_f + info_m
                    info_cov_fm = info_cov_f + info_cov_m
                    
                    omega = self.optimize_omega('trace', info_cov_xm, info_cov_fm)
                    # omega = 0.1
                    # print(omega)
                    
                    cov_joint = np.linalg.inv(omega * info_cov_xm + (1-omega) * info_cov_fm)
                    state_joint = cov_joint @ (omega * info_xm + (1-omega) * info_fm)
                    
                    self.xyt[0:3] = state_joint[0:3]
                    self.cov[0:3, 0:3] = cov_joint[0:3, 0:3]
            else:
                # batch update
                pass
            
        # state augment    
        if len(self.landmark_seq_no_in_state) > 0:
            n_observation_num = len(self.landmark_seq_no_in_state)
            
            bX = self.xyt
            self.xyt = np.zeros((dim + 2*n_observation_num, ))
            self.xyt[0:dim] = bX
             
            bP = self.cov
            self.cov = np.identity(dim + 2*n_observation_num)
            self.cov[0:dim, 0:dim] = bP

            for i in range(len(self.landmark_seq_no_in_state)):
                self.landmark_seq.append(self.landmark_seq_no_in_state[i]['id'])
                
                obs_p = self.landmark_seq_no_in_state[i]['obs']
                
                self.xyt[i*2+dim:(i+1)*2+dim] = rot_mtx(self.xyt[2]) @ obs_p + self.xyt[0:2]

                HL = np.hstack((np.identity(2), (self.J @ rot_mtx(self.xyt[2]) @ obs_p).reshape(2, 1)))
                HR = rot_mtx(self.xyt[2])

                self.cov[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = HL @ self.cov[0:3, 0:3] @ HL.T + \
                                                                HR @ (LIDAR_SIGMA **2 * np.identity(2)) @ HR.T
                
                # self.cov[i*2+dim:(i+1)*2+dim, 0:3] = HL @ self.cov[0:3, 0:3]
                # self.cov[0:3, i*2+dim:(i+1)*2+dim] = self.cov[i*2+dim:(i+1)*2+dim, 0:3].T

                # self.cov[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = HL @ self.cov[0:3, 3:i*2+dim]
                # self.cov[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.cov[i*2+dim:(i+1)*2+dim, 3:i*2+dim].T

        if SLAM_SIMULATION:
            pass
        else:
            # state marginalize
            if len(self.landmark_seq_no_obs) > 0:
                dim = self.xyt.shape[0]
                
                t_m = np.zeros((dim - len(self.landmark_seq_no_obs)*2, dim))
                t_m[0:3, 0:3] = np.identity(3)
                id_obs = 0
                landmark_seq_back = []

                for i in range(len(self.landmark_seq)):
                    if self.landmark_seq[i] in self.landmark_seq_no_obs:
                        pass
                    else:
                        t_m[3+2*id_obs:3+2*(id_obs+1), 3+2*i:3+2*(i+1)] = np.identity(2)
                        id_obs = id_obs + 1
                        landmark_seq_back.append(self.landmark_seq[i])
                
                self.landmark_seq = landmark_seq_back
                
                self.xyt = t_m @ self.xyt
                self.cov = t_m @ self.cov @ t_m.T
        
        self.robot_system.xyt = self.xyt[0:3]        
        self.robot_system.cov = self.cov[0:3, 0:3]

        #print(self.robot_system.xyt)
        #print(self.robot_system.cov)
        #print('----end----')
        
        end_time = time.time()
        self.running_time += (end_time - start_time)
    
    def optimize_omega(self, criterion, info_cov_xm, info_cov_fm):
        def optimize_fn(omega):
                    
            cov_joint = np.linalg.inv(omega * info_cov_xm + (1-omega) * info_cov_fm)
        
            P = cov_joint[0:3, 0:3]
            
            if criterion == 'det':
                return np.log(np.linalg.det(P))
            elif criterion == 'trace':
                return np.trace(P)
            else:
                pass

        return fminbound(optimize_fn, 0, 1)

    def save_est(self, t):
        px = self.robot_system.xyt[0]
        py = self.robot_system.xyt[1]
        
        psi = self.robot_system.xyt[2]

        cov = self.robot_system.cov

        self.robot_system.history.append({'px': px, 'py': py, 'psi': psi, 'cov': np.copy(cov)})

    def run(self):
        # initialize time
        t = 0

        while t < self.duration: 

          # prediction (time propagation) step
          self.prediction(t)
          
          self.absolute_observation(t)

          # save the estimate
          self.save_est(t)

          # update the time
          t = t + self.dt
        
        if PRINT_TIME:
          print('cieif duration: {} \n'.format(self.running_time / self.duration))
