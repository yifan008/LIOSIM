#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import random
import math
import os

from datetime import datetime

M_PI = 3.1415926

def _set_axis(ax):
      for direction in ['left', 'right', 'top', 'bottom']:
          ax.spines[direction].set_linewidth(1.5)

      # ax.xaxis.set_major_locator(ticker.AutoLocator())
      # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
      # ax.yaxis.set_major_locator(ticker.AutoLocator())
      # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

      ax.tick_params(axis='both',
                      which='major',
                      direction='in',
                      length=6,
                      width=1.5,
                      colors='k')

      ax.tick_params(axis='both',
                      which='minor',
                      direction='in',
                      length=3,
                      width=1.5,
                      colors='k')

      # if self.grid_args == 'on':
      ax.grid(color="gray", linestyle=':', linewidth=1)

      # Tweak spacing to prevent clipping of ylabel
      # plt.subplots_adjust(left=0.15)

def rot(psi):
  return np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
  # return np.array([[math.cos(psi), -math.sin(psi), 0.0], [math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])

def normalize_angle(psi):
  return math.atan2(math.sin(psi), math.cos(psi))

def normalize_angle_np(np_psi):
  return np.arctan2(np.sin(np_psi), np.cos(np_psi))

class robot_sim_2D:
  '''
    robot simulation
  '''
  def __init__(self, P0, YAW0, V_sigma, W_sigma):
    '''
      initialization
        :P   2x1
        :YAW 1x1
    '''
    self.P = P0
    self.YAW = YAW0

    self.last_P = self.P
    self.last_YAW = self.YAW

    self.V_sigma = V_sigma
    self.W_sigma = W_sigma

    self.M_PI = 3.1415926

  def motion(self, V, W, delta_T):
    """
      motion model
        :V 2x1
        :W 1x1
    """
    self.V = V
    self.W = W
    self.last_P = self.P
    self.last_YAW = self.YAW
    self.P = self.P + np.matmul(rot(self.YAW), self.V) * delta_T
    self.YAW = self.YAW + self.W * delta_T
      
  def odometry(self):
    """
      odometry
    """
    self.Vs = self.V + np.random.normal(0, self.V_sigma, (2, 1))
    self.Ws = self.W + self.W_sigma * np.random.randn()

class robots_mag:
  '''
    robots team simulation
  '''
  def __init__(self, V_sigma, W_sigma, p_sigma):
    '''
      initialization
        :V_sigma 2x1
        :W_sigma 1x1
    '''
    self.Number = 1
    self.Num = 0

    self.T = 1200
    self.delta_T = 1

    self.V_sigma = V_sigma # m/s
    self.W_sigma = W_sigma # rad/s
    self.p_sigma = p_sigma # m

    self.P0 = np.array([[2.0], [0.0]]) # np.random.uniform(-5, 5, (2, 1))

    self.YAW0 = 0 # np.random.uniform(-2, 2) 

    self.V = np.array([[0.0], [0.2]])

    self.W = 0.08 # np.random.uniform(-0.05, 0.05)
    
    if self.V[1, 0] / self.W > 3.0:
      self.V[1, 0] = 3.0 * self.W

    self.log_W = np.zeros((1, int(self.T / self.delta_T)))

    self.log_V = np.zeros((2, int(self.T / self.delta_T)))

    self.P = np.zeros((2, int(self.T / self.delta_T)))
    self.YAW = np.zeros((1, int(self.T / self.delta_T)))

    self.landmark_num = 15

    self.PL = np.zeros((self.landmark_num, 2))

    r = self.V[1, 0] / self.W
    rmin = 0.5
    rmax = 4.0

    for i in range(self.landmark_num):
      
      if True:
        if np.random.uniform(0, 1) >.5:
          rho = rmin + (rmax-rmin) * np.random.uniform(0, 1)
          th = 2 * M_PI * np.random.uniform(0, 1)
        else:
          rho = - rmax + (rmax-rmin) * np.random.uniform(0, 1)
          th = 2 * M_PI * np.random.uniform(0, 1)

        self.PL[i, 0] = rho * math.cos(th)
        self.PL[i, 1] = rho * math.sin(th)
      else:
        self.PL[i, :] = np.random.uniform(-8, 8, (2, ))
    
    # print(self.PL)

  def initialize(self):
    self.Num = self.Num + 1

    self.robot = robot_sim_2D(self.P0, self.YAW0, self.V_sigma, self.W_sigma)
    
  def motion_mag(self, k):
    """
      motion process
    """
    if self.Num == 1:
      # robot simulation
      self.robot.motion(self.V, self.W, self.delta_T)

      self.robot.odometry()

      self.P[..., k] = self.robot.P[..., 0]
      self.YAW[..., k] = self.robot.YAW

      self.log_W[..., k] = self.W

      self.log_V[..., k:k+1] = self.V
    else:
      self.W = self.log_W[..., k]

      self.V = self.log_V[..., k:k+1]

      # robot simulation
      self.robot.motion(self.V, self.W, self.delta_T)

      self.robot.odometry()
      
      self.P[..., k] = self.robot.P[..., 0]
      self.YAW[..., k] = self.robot.YAW

  def measurement_mag(self):
    """
      relative measurement
    """
    P_mutual_sigma = self.p_sigma # m

    self.landmark_observation = []
    self.landmark_observation_id = []

    R = rot(self.robot.YAW)

    for i in range(self.landmark_num):
      dist = np.sqrt(np.sum(np.square(self.PL[i, :] - self.robot.P.transpose())))
      if dist < 50.0:
        PL = np.array([[self.PL[i, 0]], [self.PL[i, 1]]])
        P = np.array([[self.robot.P[0, 0]], [self.robot.P[1, 0]]])

        p = np.matmul(R.transpose(), (PL - P)) + np.random.normal(0, P_mutual_sigma, (2, 1))

        self.landmark_observation.append(p)
        self.landmark_observation_id.append(i)

class Ideal_EKF:
  def __init__(self, robots_sim):
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2] = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2, 2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.P = 0.00000001 * np.identity(3)

    self.log_X[..., 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.P[i, i])
    
    self.last_Xk = np.zeros((3, 1))

  def predict(self, robots_sim, k):
    self.last_Xk[0:2,  0] = robots_sim.robot.last_P[..., 0]
    self.last_Xk[2,  0] = robots_sim.robot.last_YAW

    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws
    
    self.J = np.array([[0, -1], [1, 0]])

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.last_Xk[2, 0])),  robots_sim.V[:, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.last_Xk[2, 0]) 

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    self.P = np.matmul(self.F, np.matmul(self.P, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T
    
    self.P = 0.5 * (self.P + self.P.transpose())
    
  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    # print(self.landmark_seq)
    # print(dim)
    
    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()

    # observation
    observation_num = len(robots_sim.landmark_observation)
  
    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])

    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))
      delta = 0.001
      for i in range(dim):
        Delta = np.zeros((dim, 1))
        Delta[i, 0] = delta
        Xdelta = self.Xk - Delta
        col = (self.hx(robots_sim, self.Xk) - self.hx(robots_sim, Xdelta)) / delta
        self.H[..., i] = col[..., 0]
        
      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.X = self.X + np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)

      # print(self.P)
      
    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]

      bXk = self.Xk
      self.Xk = np.zeros((dim + 2*n_observation_num, 1))
      self.Xk[0:dim, 0] = bXk[:, 0]

      bP = self.P
      self.P = np.identity(dim + 2*n_observation_num)
      self.P[0:dim, 0:dim] = bP

      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        self.Xk[i*2+dim:(i+1)*2+dim, 0] = robots_sim.PL[self.landmark_seq_no_in_state[i], :].transpose()

        HR = - rot(self.Xk[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.Xk[i*2+dim:(i+1)*2+dim, :] - self.Xk[0:2, :])))
        HL = rot(self.Xk[2, 0]).transpose()

        self.P[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.P[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3]
        self.P[0:3, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 3:i*2+dim]
        self.P[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.P = 0.5 * (self.P + self.P.transpose())

    self.P = np.diag(np.diag(self.P))

    # print(self.P)

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass

    M = np.zeros((3, 3))

    self.log_X[..., k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.P[i, i])
      M[i, i] = self.P[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.P[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number

  def hx(self, robot_sim, X):
    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))
  
        Z[2*i:2*(i+1), 0] = p
      
      return Z
    
    return Z

class Std_EKF:
  def __init__(self, robots_sim):
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2]  = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2,  2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.P = 0.00000001 * np.identity(3)

    self.log_X[..., 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.P[i, i])
    
  def predict(self, robots_sim, k):
    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws
    
    self.J = np.array([[0, -1], [1, 0]])

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.X[2, 0])

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    self.P = np.matmul(self.F, np.matmul(self.P, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T

    self.P = 0.5 * (self.P + self.P.transpose())
  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()
    
    # observation
    observation_num = len(robots_sim.landmark_observation)
  
    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))
      delta = 0.001
      for i in range(dim):
        Delta = np.zeros((dim, 1))
        Delta[i, 0] = delta
        Xdelta = self.X - Delta
        col = (self.hx(robots_sim, self.X) - self.hx(robots_sim, Xdelta)) / delta
        self.H[..., i] = col[..., 0]
        
      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.X = self.X + np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)

    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]

      bP = self.P
      self.P = np.identity(dim + 2*n_observation_num)
      self.P[0:dim, 0:dim] = bP

      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        HR = - rot(self.X[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.X[i*2+dim:(i+1)*2+dim, :] - self.X[0:2, :])))
        HL = rot(self.X[2, 0]).transpose()

        self.P[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.P[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3]
        self.P[0:3, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 3:i*2+dim]
        self.P[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.P = 0.5 * (self.P + self.P.transpose())

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass

    # print(len(self.landmark_seq))

    M = np.zeros((3, 3))

    self.log_X[0:3, k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.P[i, i])
      M[i, i] = self.P[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.P[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number
        
  def hx(self, robot_sim, X):

    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))
  
        Z[2*i:2*(i+1), 0] = p
      
      return Z

class OC_EKF:
  def __init__(self, robots_sim):
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2]  = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2,  2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.P = 0.00000001 * np.identity(3)

    self.log_X[..., 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.P[i, i])
    
    self.X_last = np.zeros((3, 1))
    
    self.flag = False

  def predict(self, robots_sim, k):
    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws
    
    self.J = np.array([[0, -1], [1, 0]])

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.X[2, 0])

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    if not self.flag : 
      self.flag = True   
      self.X_last = self.X[0:3, :]
    else:
      self.F[0:2, 2] = self.J @ (self.X[0:2, 0] - self.X_last[0:2, 0])
      
      self.X_last = self.X[0:3, :]
    
    self.P = np.matmul(self.F, np.matmul(self.P, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T

    self.P = 0.5 * (self.P + self.P.transpose())

    # print(self.log_X_P[2, k])

  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()
    
    # observation
    observation_num = len(robots_sim.landmark_observation)

    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))

      # delta = 0.001
      # for i in range(dim):
      #   Delta = np.zeros((dim, 1))
      #   Delta[i, 0] = delta
      #   Xdelta = self.X - Delta
      #   col = (self.hx(robots_sim, self.X) - self.hx(robots_sim, Xdelta)) / delta
      #   self.H[..., i] = col[..., 0]
      
      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]
        ids = self.landmark_seq.index(idx)
        
        self.H[i*2:(i+1)*2, 2] = - rot(self.X[2, 0]).transpose() @ self.J @ (self.L0[2*ids:2*(ids+1), 0] - self.X[0:2, 0])
        self.H[i*2:(i+1)*2, 0:2] = - rot(self.X[2, 0]).transpose()
        self.H[i*2:(i+1)*2, 2*ids+3:2*(ids+1)+3] = rot(self.X[2, 0]).transpose()

      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.X = self.X + np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)

    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]

      bP = self.P
      self.P = np.identity(dim + 2*n_observation_num)
      self.P[0:dim, 0:dim] = bP

      if len(self.landmark_seq) == 0:
        self.L0 = np.zeros((2*n_observation_num, 1))
      else:
        bL0 = self.L0
        self.L0 = np.zeros((2*n_observation_num + (dim-3), 1))
        self.L0[0:(dim-3), 0] = bL0[:, 0]

      # print(bL0)
      # print(dim)
      # print(len(self.landmark_seq))
      # print(self.L0.shape)

      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        self.L0[i*2+(dim-3):(i+1)*2+(dim-3), :] = self.X[i*2+dim:(i+1)*2+dim, :]

        HR = - rot(self.X[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.X[i*2+dim:(i+1)*2+dim, :] - self.X[0:2, :])))
        HL = rot(self.X[2, 0]).transpose()

        self.P[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.P[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3]
        self.P[0:3, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 3:i*2+dim]
        self.P[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.P = 0.5 * (self.P + self.P.transpose())

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass

    # print(len(self.landmark_seq))
    M = np.zeros((3, 3))

    self.log_X[0:3, k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.P[i, i])
      M[i, i] = self.P[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.P[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number

    # print(self.log_X_P[2, k+1])

  def hx(self, robot_sim, X):
    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))

        Z[2*i:2*(i+1), 0] = p
      
      return Z

  
class OC1_EKF:
  def __init__(self, robots_sim):   
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2]  = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2,  2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.P = 0.00000001 * np.identity(3)

    self.log_X[0:3, 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.P[i, i])

    self.F0 = np.identity(3)
    
    self.J = np.array([[0, -1], [1, 0]])

    self.N = np.zeros((3, 3))
    
    self.N[0:2, 0:2] = np.identity(2)
    self.N[0:2, 2] = self.J @ self.X[0:2, 0]
    self.N[2, 2] = 1

  def predict(self, robots_sim, k):
    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.X[2, 0])

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    self.P = np.matmul(self.F, np.matmul(self.P, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T

    self.P = 0.5 * (self.P + self.P.transpose())

    self.F0 = np.matmul(self.F, self.F0)

  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()
    
    # observation
    observation_num = len(robots_sim.landmark_observation)
  
    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))
      delta = 0.001
      for i in range(dim):
        Delta = np.zeros((dim, 1))
        Delta[i, 0] = delta
        Xdelta = self.X - Delta
        col = (self.hx(robots_sim, self.X) - self.hx(robots_sim, Xdelta)) / delta
        self.H[..., i] = col[..., 0]

      V = np.matmul(self.F0, self.N)

      self.H = self.H @ (np.identity(dim) - V @ np.linalg.inv(V.transpose() @ V) @ V.transpose())

      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.X = self.X + np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)

    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bF0 = self.F0
      self.F0 = np.identity(dim + 2*n_observation_num)
      self.F0[0:dim, 0:dim] = bF0

      bN = self.N
      self.N = np.zeros((dim + 2*n_observation_num, 3))
      self.N[0:dim, 0:3] = bN
      
      for i in range(n_observation_num):
        self.N[i*2+dim:(i+1)*2+dim, 0:2] = np.identity(2)

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]

      bP = self.P
      self.P = np.identity(dim + 2*n_observation_num)
      self.P[0:dim, 0:dim] = bP

      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        self.N[i*2+dim:(i+1)*2+dim, 2] = self.J @ self.X[i*2+dim:(i+1)*2+dim, 0]

        HR = - rot(self.X[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.X[i*2+dim:(i+1)*2+dim, :] - self.X[0:2, :])))
        HL = rot(self.X[2, 0]).transpose()

        self.P[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.P[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 0:3]
        self.P[0:3, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.P[0:3, 3:i*2+dim]
        self.P[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.P[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.P = 0.5 * (self.P + self.P.transpose())

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass
   
    M = np.zeros((3, 3))

    self.log_X[0:3, k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.P[i, i])
      M[i, i] = self.P[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.P[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number

  def hx(self, robot_sim, X):
    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))
  
        Z[2*i:2*(i+1), 0] = p
      
      return Z

class KF_EKF:
  def __init__(self, robots_sim):
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2]  = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2,  2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.Px = 0.00000001 * np.identity(3)

    self.log_X[..., 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.Px[i, i])

    self.J = np.array([[0, -1], [1, 0]])

    self.Fz = np.identity(3)

    self.T = np.zeros((3, 3))

    self.T[0, 2] = 1
    self.T[1:3, 0:2] = np.identity(2)

    self.P = self.T @ self.Px @ self.T.transpose()

  def predict(self, robots_sim, k):
    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    dim = self.X.shape[0]

    self.Fz[dim-2:dim, dim-3] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.X[2, 0])

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    self.Px = np.matmul(self.F, np.matmul(self.Px, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T

    self.Px = 0.5 * (self.Px + self.Px.transpose())

    self.T = np.zeros((dim, dim))
    self.T[dim-2:dim, 0:2] = np.identity(2)
    self.T[dim-3, 2] = 1

    if dim > 3:
      self.T[0:dim-3, 3:dim] = - np.identity(dim-3)
      for i in range(len(self.landmark_seq)):
        idx = i
        self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

    Gz = self.T @ self.G 

    self.P = np.matmul(self.Fz, np.matmul(self.P, self.Fz.transpose())) + Gz @ self.Q @ Gz.transpose() * robots_sim.delta_T * robots_sim.delta_T

    self.P = 0.5 * (self.P + self.P.transpose())

    # invT = np.linalg.inv(self.T)
    # self.Px = invT @ self.P @ invT.transpose()

  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()
    
    # observation
    observation_num = len(robots_sim.landmark_observation)
  
    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))

      for i in range(observation_num):
        idx = self.landmark_seq.index(self.landmark_seq_in_state[i])
        self.H[i*2:(i+1)*2, idx*2:(idx+1)*2] = -rot(self.X[2, 0]).transpose()

      # self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
      # self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.Z = np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)
      
      self.X = self.X + np.linalg.inv(self.T) @ self.Z

      for i in range(len(self.landmark_seq)):
        idx = i
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      invT = np.linalg.inv(self.T)
      self.Px = invT @ self.P @ invT.transpose()

    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bFz = self.Fz
      self.Fz = np.identity(dim + 2*n_observation_num)
      self.Fz[2*n_observation_num:dim + 2*n_observation_num, 2*n_observation_num:dim + 2*n_observation_num] = bFz

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bPx = self.Px
      self.Px = np.identity(dim + 2*n_observation_num)
      self.Px[0:dim, 0:dim] = bPx

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]
  
      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        HR = - rot(self.X[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.X[i*2+dim:(i+1)*2+dim, :] - self.X[0:2, :])))
        HL = rot(self.X[2, 0]).transpose()

        self.Px[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.Px[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.Px[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.Px[0:3, 0:3]
        self.Px[0:3, i*2+dim:(i+1)*2+dim] = self.Px[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.Px[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.Px[0:3, 3:i*2+dim]
        self.Px[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.Px[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.Px = 0.5 * (self.Px + self.Px.transpose())

      # TODO: self.T
      dim = self.X.shape[0]
      self.T = np.zeros((dim, dim))

      self.T[dim-2:dim, 0:2] = np.identity(2)
      self.T[dim-3, 2] = 1

      self.T[0:dim-3, 3:dim] = - np.identity(dim-3)

      for i in range(len(self.landmark_seq)):
        idx = i

        self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      self.P = self.T @ self.Px @ self.T.transpose()

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass

    M = np.zeros((3, 3))

    self.log_X[0:3, k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.Px[i, i])
      M[i, i] = self.Px[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.Px[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number

  def hx(self, robot_sim, X):
    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))
  
        Z[2*i:2*(i+1), 0] = p
      
      return Z


class T_EKF:
  def __init__(self, robots_sim):
    self.log_X =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_P =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_RMS =  np.zeros((3, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))
    self.log_X_NEES_M = np.zeros((1, int(robots_sim.T / robots_sim.delta_T) + 1))

  def initialize(self, robots_sim):
    self.F = np.identity(3)
    self.B = np.identity(3)
    self.G = np.identity(3)

    self.Q = np.identity(3)
    
    self.Q[0:2,  0:2]  = robots_sim.V_sigma * robots_sim.V_sigma * self.Q[0:2,  0:2]
    self.Q[2,  2] = robots_sim.W_sigma * robots_sim.W_sigma * self.Q[2, 2]

    self.landmark_seq = []

    self.X = np.zeros((3, 1))

    self.X[0:2,  0] = robots_sim.robot.P[:, 0]
    self.X[2,  0]  = robots_sim.robot.YAW

    self.Px = 0.00000001 * np.identity(3)

    self.log_X[..., 0] = self.X[0:3, 0]

    for i in range(3):
      self.log_X_P[i, 0] = math.sqrt(self.Px[i, i])

    self.J = np.array([[0, -1], [1, 0]])

    self.Fz = np.identity(3)

    self.T = np.zeros((3, 3))

    self.T[0, 2] = 1
    self.T[1:3, 0:2] = np.identity(2)

    self.P = self.T @ self.Px @ self.T.transpose()

  def predict(self, robots_sim, k):
    Uk = np.zeros((3, 1))

    Uk[0:2,  0] = robots_sim.robot.Vs[..., 0]
    Uk[2,  0] = robots_sim.robot.Ws

    self.F[0:2, 2] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    dim = self.X.shape[0]

    self.Fz[dim-2:dim, dim-3] = np.matmul(np.matmul(self.J, rot(self.X[2, 0])), Uk[0:2, 0]) * robots_sim.delta_T

    self.G[0:2, 0:2] = rot(self.X[2, 0])

    self.B[0:2, 0:2] = rot(self.X[2, 0])

    self.X = self.X + np.matmul(self.B, Uk) * robots_sim.delta_T

    self.Px = np.matmul(self.F, np.matmul(self.Px, self.F.transpose())) + np.matmul(np.matmul(self.G, self.Q), self.G.transpose()) * robots_sim.delta_T * robots_sim.delta_T

    self.Px = 0.5 * (self.Px + self.Px.transpose())

    self.T = np.zeros((dim, dim))
    self.T[dim-2:dim, 0:2] = np.identity(2)
    self.T[dim-3, 2] = 1

    if dim > 3:
      self.T[0:dim-3, 3:dim] = - np.identity(dim-3)
      for i in range(len(self.landmark_seq)):
        idx = i
        self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

    Gz = self.T @ self.G 

    self.P = np.matmul(self.Fz, np.matmul(self.P, self.Fz.transpose())) + Gz @ self.Q @ Gz.transpose() * robots_sim.delta_T * robots_sim.delta_T

    self.P = 0.5 * (self.P + self.P.transpose())

    # invT = np.linalg.inv(self.T)
    # self.Px = invT @ self.P @ invT.transpose()

  def update(self, robots_sim, k):
    dim = self.X.shape[0]

    self.Xk = np.zeros((dim, 1))

    self.Xk[0:2,  0] = robots_sim.robot.P[..., 0] 
    self.Xk[2,  0] = robots_sim.robot.YAW
    
    for i in range(len(self.landmark_seq)):
      self.Xk[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[self.landmark_seq[i], :].transpose()
    
    # observation
    observation_num = len(robots_sim.landmark_observation)
  
    if observation_num == 0:
      return
    
    self.landmark_seq_in_state = []
    self.landmark_seq_no_in_state = []

    for i in range(len(robots_sim.landmark_observation_id)):
      if robots_sim.landmark_observation_id[i] in self.landmark_seq:
        self.landmark_seq_in_state.append(robots_sim.landmark_observation_id[i])
      else:
        self.landmark_seq_no_in_state.append(robots_sim.landmark_observation_id[i])

    self.landmark_seq_no_obs = []

    for i in range(len(self.landmark_seq)):
      if self.landmark_seq[i] in robots_sim.landmark_observation_id:
        pass
      else:
        self.landmark_seq_no_obs.append(self.landmark_seq[i])
        
    # state update
    if len(self.landmark_seq_in_state) > 0:
      observation_num = len(self.landmark_seq_in_state)
      Yk = np.zeros((2*observation_num, 1))
      
      for i in range(observation_num):
        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_in_state[i])
        Yk[i*2:(i+1)*2, 0] = robots_sim.landmark_observation[idx][:, 0]

      self.R = robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2*observation_num)

      self.H = np.zeros((2*observation_num, dim))

      for i in range(observation_num):
        idx = self.landmark_seq.index(self.landmark_seq_in_state[i])
        self.H[i*2:(i+1)*2, idx*2:(idx+1)*2] = -rot(self.X[2, 0]).transpose()

      # self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
      # self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R))
      
      self.Y = Yk - self.hx(robots_sim, self.X)
      self.Z = np.matmul(K, self.Y)
      self.P = np.matmul((np.identity(dim) - np.matmul(K, self.H)), self.P)
      
      self.X = self.X + np.linalg.inv(self.T) @ self.Z

      for i in range(len(self.landmark_seq)):
        idx = i
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      invT = np.linalg.inv(self.T)
      self.Px = invT @ self.P @ invT.transpose()

    # state augment    
    if len(self.landmark_seq_no_in_state) > 0:
      n_observation_num = len(self.landmark_seq_no_in_state)
      bF = self.F
      self.F = np.identity(dim + 2*n_observation_num)
      self.F[0:dim, 0:dim] = bF

      bFz = self.Fz
      self.Fz = np.identity(dim + 2*n_observation_num)
      self.Fz[2*n_observation_num:dim + 2*n_observation_num, 2*n_observation_num:dim + 2*n_observation_num] = bFz

      bB = self.B      
      self.B = np.zeros((dim + 2*n_observation_num, 3))
      self.B[0:dim, 0:3] = bB

      bG = self.G
      self.G = np.zeros((dim + 2*n_observation_num, 3))
      self.G[0:dim, 0:3] = bG

      bPx = self.Px
      self.Px = np.identity(dim + 2*n_observation_num)
      self.Px[0:dim, 0:dim] = bPx

      bX = self.X
      self.X = np.zeros((dim + 2*n_observation_num, 1))
      self.X[0:dim, 0] = bX[:, 0]
  
      for i in range(len(self.landmark_seq_no_in_state)):
        self.landmark_seq.append(self.landmark_seq_no_in_state[i])

        idx = robots_sim.landmark_observation_id.index(self.landmark_seq_no_in_state[i])

        obs_p = robots_sim.landmark_observation[idx]
        
        self.X[i*2+dim:(i+1)*2+dim, :] = rot(self.X[2, 0]) @ obs_p + self.X[0:2, :]

        HR = - rot(self.X[2, 0]).transpose() @ np.hstack((np.identity(2), self.J @ (self.X[i*2+dim:(i+1)*2+dim, :] - self.X[0:2, :])))
        HL = rot(self.X[2, 0]).transpose()

        self.Px[i*2+dim:(i+1)*2+dim, i*2+dim:(i+1)*2+dim] = np.linalg.inv(HL) @ HR @ self.Px[0:3, 0:3] @ HR.transpose() @ np.linalg.inv(HL).transpose() + \
                                                           np.linalg.inv(HL) @ (robots_sim.p_sigma * robots_sim.p_sigma * np.identity(2)) @ np.linalg.inv(HL).transpose()
        
        self.Px[i*2+dim:(i+1)*2+dim, 0:3] = -np.linalg.inv(HL) @ HR @ self.Px[0:3, 0:3]
        self.Px[0:3, i*2+dim:(i+1)*2+dim] = self.Px[i*2+dim:(i+1)*2+dim, 0:3].transpose()

        self.Px[i*2+dim:(i+1)*2+dim, 3:i*2+dim] = -np.linalg.inv(HL) @ HR @ self.Px[0:3, 3:i*2+dim]
        self.Px[3:i*2+dim, i*2+dim:(i+1)*2+dim] = self.Px[i*2+dim:(i+1)*2+dim, 3:i*2+dim].transpose()

        self.Px = 0.5 * (self.Px + self.Px.transpose())

      # TODO: self.T
      dim = self.X.shape[0]
      self.T = np.zeros((dim, dim))

      self.T[dim-2:dim, 0:2] = np.identity(2)
      self.T[dim-3, 2] = 1

      self.T[0:dim-3, 3:dim] = - np.identity(dim-3)

      for i in range(len(self.landmark_seq)):
        idx = i

        self.T[idx*2:(idx+1)*2, 0:2] = np.identity(2)
        self.T[idx*2:(idx+1)*2, 2] = self.J @ (self.X[idx*2+3:(idx+1)*2+3, 0] - self.X[0:2, 0])

      self.P = self.T @ self.Px @ self.T.transpose()

    # state marginalize
    if len(self.landmark_seq_no_obs) > 0:
      pass

    M = np.zeros((3, 3))

    self.log_X[0:3, k+1] = self.X[0:3, 0]
    
    for i in range(3):
      self.log_X_P[i, k+1] = math.sqrt(self.Px[i, i])
      M[i, i] = self.Px[i, i]
    
    X_e = self.X[0:3, 0] - self.Xk[0:3, 0]

    X_e[2] = normalize_angle(X_e[2])

    self.log_X_RMS[..., k+1] = self.log_X_RMS[..., k+1] + np.fabs(X_e) / robots_sim.Number

    P_c = np.zeros((3, 3))
    
    P_c[0:3, 0:3] = self.Px[0:3, 0:3]

    X_e_c = X_e

    self.log_X_NEES[0, k+1] = self.log_X_NEES[0, k+1] + X_e_c.transpose() @ np.linalg.inv(P_c) @ X_e_c / robots_sim.Number

    self.log_X_NEES_M[0, k+1] = self.log_X_NEES_M[0, k+1] + X_e.transpose() @ np.linalg.inv(M) @ X_e / robots_sim.Number

  def hx(self, robot_sim, X):
    observation_num = len(self.landmark_seq_in_state)

    if observation_num > 0:
      Z = np.zeros((2*observation_num, 1))
      
      R1 = rot(X[2, 0]) 

      for i in range(observation_num):
        idx = self.landmark_seq_in_state[i]

        ids = self.landmark_seq.index(idx)

        p = np.matmul(R1.transpose(), (X[ids*2+3:(ids+1)*2+3, 0] - X[0:2, 0]))
  
        Z[2*i:2*(i+1), 0] = p
      
      return Z


def save(robots_sim, ideal_ekf, std_ekf, oc_ekf, oc1_ekf, kf_ekf, t_ekf):
  x_true = np.zeros((2*robots_sim.landmark_num+3, int(robots_sim.T / robots_sim.delta_T)))
  
  x_true[0:2, :] = robots_sim.P[0:2, :]
  x_true[2, :] = robots_sim.YAW

  for i in range(robots_sim.landmark_num):
    x_true[i*2+3:(i+1)*2+3, 0] = robots_sim.PL[i, :].transpose()

  num = 2
  np.savez('Datum_ideal_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = ideal_ekf.log_X, P = ideal_ekf.log_X_P, rms = ideal_ekf.log_X_RMS, nees = ideal_ekf.log_X_NEES, nees_m = ideal_ekf.log_X_NEES_M)
  np.savez('Datum_std_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = std_ekf.log_X, P = std_ekf.log_X_P, rms = std_ekf.log_X_RMS, nees = std_ekf.log_X_NEES, nees_m = std_ekf.log_X_NEES_M)
  np.savez('Datum_oc_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = oc_ekf.log_X, P = oc_ekf.log_X_P, rms = oc_ekf.log_X_RMS, nees = oc_ekf.log_X_NEES, nees_m = oc_ekf.log_X_NEES_M)
  np.savez('Datum_oc1_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = oc1_ekf.log_X, P = oc1_ekf.log_X_P, rms = oc1_ekf.log_X_RMS, nees = oc1_ekf.log_X_NEES, nees_m = oc1_ekf.log_X_NEES_M)
  np.savez('Datum_kf_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = kf_ekf.log_X, P = kf_ekf.log_X_P, rms = kf_ekf.log_X_RMS, nees = kf_ekf.log_X_NEES, nees_m = kf_ekf.log_X_NEES_M)
  np.savez('Datum_t_ekf_kf_' + str(num) + '.npz', t = robots_sim.T, delta_t = robots_sim.delta_T, x_t = x_true, x = t_ekf.log_X, P = t_ekf.log_X_P, rms = t_ekf.log_X_RMS, nees = t_ekf.log_X_NEES, nees_m = t_ekf.log_X_NEES_M)

#################################################### MAIN ###########################################################################
# SLAM Monte Carlo Simulation (NEES) $Ideal-Std-OC-OC1-KF$
# Kalman Decomposition Experiment
# 2D simulation
# NSF-EKF
# KF-EKF
# Circular Trajectories
if __name__ == '__main__':
  V_sigma = 0.1 # m/s
  W_sigma = 0.08 # rad/s
  # V_sigma = 0.1 # m/s
  # W_sigma = 0.2 # rad/s
  p_sigma = 0.1

  robots_sim = robots_mag(V_sigma, W_sigma, p_sigma)

  oc_ekf = OC_EKF(robots_sim)
  ideal_ekf = Ideal_EKF(robots_sim)
  std_ekf = Std_EKF(robots_sim)
  oc1_ekf = OC1_EKF(robots_sim)
  kf_ekf = KF_EKF(robots_sim)
  t_ekf =  T_EKF(robots_sim)

  while robots_sim.Num < robots_sim.Number:
    robots_sim.initialize()
    
    # ideal_ekf.initialize(robots_sim)
    std_ekf.initialize(robots_sim)
    # oc_ekf.initialize(robots_sim)
    # oc1_ekf.initialize(robots_sim)
    kf_ekf.initialize(robots_sim)
    t_ekf.initialize(robots_sim)

    for i in range(0, int(robots_sim.T / robots_sim.delta_T)):
      robots_sim.motion_mag(i)
      robots_sim.measurement_mag()

      std_ekf.predict(robots_sim, i)
      std_ekf.update(robots_sim, i)

      # ideal_ekf.predict(robots_sim, i)
      # ideal_ekf.update(robots_sim, i)

      # oc_ekf.predict(robots_sim, i)
      # oc_ekf.update(robots_sim, i)

      # oc1_ekf.predict(robots_sim, i)
      # oc1_ekf.update(robots_sim, i)

      kf_ekf.predict(robots_sim, i)
      kf_ekf.update(robots_sim, i)

      t_ekf.predict(robots_sim, i)
      t_ekf.update(robots_sim, i)

    print("\033[0;36m Monte Carlo Number: \033[0m", robots_sim.Num)

  save(robots_sim, ideal_ekf, std_ekf, oc_ekf, oc1_ekf, kf_ekf, t_ekf)
