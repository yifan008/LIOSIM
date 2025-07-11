#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm

from pathlib import Path

from ideal import Ideal_EKF
from ekf import Centralized_EKF
from tekf import T_EKF
from msc import MSC_EKF
from msc_ideal import MSC_Ideal

from robot_system import *

import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _set_axis(ax):
      for direction in ['left', 'right', 'top', 'bottom']:
          ax.spines[direction].set_linewidth(2.5)

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

      ax.tick_params(labelsize=11) 

      ax.grid(color="gray", linestyle=':', linewidth=1)

class sim_mag:
    def __init__(self, team_settings):
        self.dt = team_settings['dt']
        self.duration = team_settings['duration']
        self.iter_num = team_settings['iter_num']

        self.at_sigma = team_settings['at_sigma']

        self.bearing_sigma = team_settings['bearing_sigma']
        
        self.N = 1

        self.history = dict()

        self.history['p_t'] = list()
        self.history['v_t'] = list()

        self.history['p'] = list() 
        self.history['v'] = list()

        self.history['a_t'] = list()
        self.history['a'] = list()

        if SIM_FLAG == 1: # PNG pursuing
            self.p_t = np.array([150.0, -150.0])
            self.p = np.array([0.0, 0.0])
            self.v_t = np.array([3.0, 4.0])
            self.v = np.array([5.3, 0.0])
        elif SIM_FLAG == 2: # PID intercept
            self.p_t = np.array([-15000.0, -1500.0])
            self.p = np.array([12000.0, 2000.0])
            self.v_t = np.array([3.0, 1.0])
            self.v = np.array([-5.0, -3.0])
        elif SIM_FLAG == 3: # cross (relative motion along a line, no relative acceleration)
            self.p_t = np.array([150.0, -150.0])
            self.p = np.array([0.0, 0.0])
            self.v_t = np.array([3.0, 3.0])
            self.v = np.array([0, 3.0])
        elif SIM_FLAG == 4: # cross (relative motion along a line, no relative acceleration)
            self.p_t = np.array([1000.0, -500.0])
            self.p = np.array([0.0, 0.0])
            self.v_t = np.array([3.0, 0.0])
            self.v = np.array([3.0, 3.0])
        elif SIM_FLAG == 5: # parallel (relative motion along a line, no relative acceleration)
            self.p_t = np.array([1500.0, -500.0])
            self.p = np.array([0.0, 0.0])
            self.v_t = np.array([4.0, 4.0])
            self.v = np.array([3.0, 3.0])
        elif SIM_FLAG == 6: # parallel (no relative motion, no relative acceleration)
            self.p_t = np.array([150.0, -150.0])
            self.p = np.array([0.0, 0.0])
            self.v_t = np.array([3.0, 0.0])
            self.v = np.array([3.0, 0.0])
        else:
            pass

        self.history['p_t'].append(np.copy(self.p_t))
        self.history['v_t'].append(np.copy(self.v_t))
        
        self.history['p'].append(np.copy(self.p))    
        self.history['v'].append(np.copy(self.v))
            
    def motion_trajectory(self):
        # motion trajectories generation
        t = self.dt

        while t <= self.duration:
            p_r = self.p_t - self.p
            rho = np.linalg.norm(p_r)
            e_r = p_r / rho
            
            v_r = self.v_t - self.v
            
            # q_rate = np.cross(vm / rho * e_v, e_r)
            q_rate = (p_r[0] * v_r[1] - p_r[1] * v_r[0]) / (rho**2)
            
            vm = np.linalg.norm(self.v)
            e_v = self.v / vm
            e_v_n = np.array(([[0, -1], [1, 0]])) @ e_v
    
            if SIM_FLAG == 1:
                self.a = self.N * np.linalg.norm(self.v) * q_rate * e_v_n    
            elif SIM_FLAG == 2:
                self.a = self.N * np.linalg.norm(self.v) * q_rate * e_v_n    
            elif SIM_FLAG == 3:
                self.a = np.array((0, 0))
            elif SIM_FLAG == 4:
                self.a = np.array((0, 0))
                # self.a = v_r * np.random.uniform(-0.4, 0.4)
            elif SIM_FLAG == 5:
                self.a = np.array((0, 0))
                # self.a = v_r * np.random.uniform(-0.4, 0.4)
            elif SIM_FLAG == 6:
                self.a = np.array((0, 0))
            else:
                pass

            self.p += self.v * self.dt + 0.5* self.a * self.dt**2
            self.v += self.a * self.dt
            
            self.a_t = np.zeros((2, ))
            
            self.p_t += self.v_t * self.dt + 0.5* self.a_t * self.dt**2
            self.v_t += self.a_t * self.dt

            self.history['p'].append(np.copy(self.p))
            self.history['v'].append(np.copy(self.v))
            self.history['a'].append(np.copy(self.a))
            
            self.history['p_t'].append(np.copy(self.p_t))
            self.history['v_t'].append(np.copy(self.v_t))
            self.history['a_t'].append(np.copy(self.a_t))
    
            t += self.dt

    def measurement_sim(self):
        # odometry information
        self.odometry = list()

        for k in range(len(self.history['a'])):
            u = self.history['a'][k] # + self.at_sigma * np.random.randn(2, )

            self.odometry.append(u)

        self.measurement = list()

        for k in range(len(self.history['p'])):
            
            p = self.history['p'][k] 
            p_t = self.history['p_t'][k]
            
            p_r = p_t - p
            # theta = np.arctan2(p_r[1], p_r[0]) + self.bearing_sigma * np.random.randn()
            theta = np.arctan2(p_r[0], p_r[1]) + self.bearing_sigma * np.random.randn()

            self.measurement.append(theta)
        
        return self.odometry, self.measurement
        
if __name__ == '__main__':
    dt = STEP
    duration = DURATION
    iter_num = ITER_NUM
    at_sigma = AT_SIGMA
    bearing_sigma = BEARING_SIGMA

    # sim settings
    sim_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'at_sigma': at_sigma, 'bearing_sigma': bearing_sigma}

    sim = sim_mag(sim_settings)

    sim.motion_trajectory()

    '''
    length = len(sim.history['p'])
    
    # print(sim.history['p'][1])
    # print(length)
    
    px = np.zeros((length, ))
    py = np.zeros((length, ))
    
    px_t = np.zeros((length, ))
    py_t = np.zeros((length, ))
  
    for i in range(length):
      px[i] = sim.history['p'][i][0]
      py[i] = sim.history['p'][i][1]
      
      px_t[i] = sim.history['p_t'][i][0]
      py_t[i] = sim.history['p_t'][i][1]
    
    plt.figure()
    ax1 = plt.gca()
    _set_axis(ax1)
    
    plt.plot(px_t, py_t, color='green', label='target', linestyle='-') 
    plt.plot(px_t[0:1], py_t[0:1], color='green', marker='*')

    plt.plot(px, py, color='blue', label='pursuer', linestyle='--')         
    plt.plot(px[0:1], py[0:1], color='blue', marker='*')

    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    
    plt.legend(loc = 'upper right', frameon=True, ncol = 2, prop = {'size':6})

    plt.show()

    '''
    
    xyt_0 = np.zeros((4))

    gt = dict()

    xyt_0[0:2] = sim.history['p_t'][0] - sim.history['p'][0]
    xyt_0[2:4] = sim.history['v_t'][0] - sim.history['v'][0]

    gt = dict()

    gt['p_t'] = sim.history['p_t']
    gt['v_t'] = sim.history['v_t']
    gt['p'] = sim.history['p']
    gt['v'] = sim.history['v']
    
    team_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'at_sigma': at_sigma, 'bearing_sigma': bearing_sigma}

    results = list()

    with tqdm(total=(iter_num), leave=False) as pbar:
        for i in range(iter_num):
            odometry, measurement = sim.measurement_sim()

            print('algorithm running ...')

            dataset = dict()
            dataset['odometry'] = odometry
            dataset['measurement'] = measurement

            dataset['gt'] = gt

            result_alg = dict()

            for alg in algorithms:
                if alg == 'ekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    CENEKF = Centralized_EKF(robot_system, dataset)
                    CENEKF.run()
                    robot_system = CENEKF.robot_system
                elif alg == 'tekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    TEKF = T_EKF(robot_system, dataset)
                    TEKF.run()
                    robot_system = TEKF.robot_system
                elif alg == 'msc':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    MSCEKF = MSC_EKF(robot_system, dataset)
                    MSCEKF.run()
                    robot_system = MSCEKF.robot_system
                elif alg == 'ideal':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    IDEALEKF = Ideal_EKF(robot_system, dataset)
                    IDEALEKF.run()
                    robot_system = IDEALEKF.robot_system
                elif alg == 'msc_ideal':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    MSCIDEAL = MSC_Ideal(robot_system, dataset)
                    MSCIDEAL.run()
                    robot_system = MSCIDEAL.robot_system
                else:
                    sys.exit('Invalid algorithm input!')
            
                result_alg[alg] = robot_system

            results.append(result_alg)

            pbar.update()

    # individual RMSE plots
    rmse_pos = dict()
    rmse_vel = dict()
    nees_avg_pos = dict()
    nees_avg_vel = dict()
        
    for alg in algorithms:
        rmse_pos[alg] = 0
        rmse_vel[alg] = 0
        nees_avg_pos[alg] = 0
        nees_avg_vel[alg] = 0

    t = time.strftime("%Y-%m-%d %H:%M:%S")

    print('TIME: {}'.format(t))

    # num = math.ceil(duration / dt + 1)
  
    num = len(sim.history['p'])
  
    px_gt = np.squeeze([sim.history['p'][k][0] for k in range(len(sim.history['p']))])
    py_gt = np.squeeze([sim.history['p'][k][1] for k in range(len(sim.history['p']))])

    vx_gt = np.squeeze([sim.history['v'][k][0] for k in range(len(sim.history['v']))])
    vy_gt = np.squeeze([sim.history['v'][k][1] for k in range(len(sim.history['v']))])
    
    px_t_gt = np.squeeze([sim.history['p_t'][k][0] for k in range(len(sim.history['p_t']))])
    py_t_gt = np.squeeze([sim.history['p_t'][k][1] for k in range(len(sim.history['p_t']))])
    
    vx_t_gt = np.squeeze([sim.history['v_t'][k][0] for k in range(len(sim.history['v_t']))])
    vy_t_gt = np.squeeze([sim.history['v_t'][k][1] for k in range(len(sim.history['v_t']))])
    
    time_arr = np.array([k * dt for k in range(num)])

    for alg in algorithms:

        pos_error = np.zeros((num, ))
        vel_error = np.zeros((num, ))
      
        nees_pos = np.zeros((num, ))
        nees_vel = np.zeros((num, ))

        for i in range(iter_num):
            s_nees = np.zeros((num, ))
            s_nees_pos = np.zeros((num, ))
            s_nees_vel = np.zeros((num, ))
            
            rx_est = np.squeeze([results[i][alg].history[k]['x'] for k in range(len(results[i][alg].history))])
            ry_est = np.squeeze([results[i][alg].history[k]['y'] for k in range(len(results[i][alg].history))])

            rvx_est = np.squeeze([results[i][alg].history[k]['vx'] for k in range(len(results[i][alg].history))])
            rvy_est = np.squeeze([results[i][alg].history[k]['vy'] for k in range(len(results[i][alg].history))])
            
            cov_p_est = np.squeeze([results[i][alg].history[k]['cov'][0:2, 0:2] for k in range(len(results[i][alg].history))])
            cov_v_est = np.squeeze([results[i][alg].history[k]['cov'][2:4, 2:4] for k in range(len(results[i][alg].history))])

            epos = (np.array(rx_est) - np.array(px_t_gt - px_gt)) ** 2 + (np.array(ry_est) - np.array(py_t_gt - py_gt)) ** 2
            pos_error += epos
            evel = (np.array(rvx_est) - np.array(vx_t_gt - vx_gt)) ** 2 + (np.array(rvy_est) - np.array(vy_t_gt - vy_gt)) ** 2
            vel_error += evel

            for k in range(len(results[i][alg].history)):
                cov = results[i][alg].history[k]['cov']

                dp = np.array((rx_est[k] - (px_t_gt[k] - px_gt[k]), ry_est[k] - (py_t_gt[k] - py_gt[k])))
                dv = np.array((rvx_est[k] - (vx_t_gt[k] - vx_gt[k]), rvy_est[k] - (vy_t_gt[k] - vy_gt[k])))

                nees_pos[k] += dp.T @ np.linalg.inv(cov[0:2, 0:2]) @ dp
                nees_vel[k] += dv.T @ np.linalg.inv(cov[2:4, 2:4]) @ dv

                ds = np.hstack((dp, dv))
                s_nees[k] = ds.T @ np.linalg.inv(cov) @ ds
                s_nees_pos[k] = dp.T @ np.linalg.inv(cov[0:2, 0:2]) @ dp
                s_nees_vel[k] = dv.T @ np.linalg.inv(cov[2:4, 2:4]) @ dv
                
            save_path = '../sim_results' + '/' + t + '/BearingTT' + str(i+1) + '/' 

            Path(save_path).mkdir(parents=True, exist_ok=True)

            file_name = alg + '.npz'

            np.savez(save_path + file_name, t = time_arr, rx_est = rx_est, ry_est = ry_est, rvx_est = rvx_est, 
                    rvy_est = rvy_est, px_gt = px_gt, py_gt = py_gt, px_t_gt = px_t_gt, py_t_gt = py_t_gt, 
                    vx_gt = vx_gt, vy_gt = vy_gt, vx_t_gt = vx_t_gt, vy_t_gt = vy_t_gt,
                    cov_p_est = cov_p_est, cov_v_est = cov_v_est, epos = epos, evel = evel, nees = s_nees, 
                    nees_vel = s_nees_vel, nees_pos = s_nees_pos)

        rmse_pos[alg] += np.sum(pos_error)
        rmse_vel[alg] += np.sum(vel_error)

        nees_avg_pos[alg] += np.sum(nees_pos)
        nees_avg_vel[alg] += np.sum(nees_vel)
        
        N = time_arr.shape[0]

    data_num = iter_num * N

    print('ALG: RMSE_POS             RMSE_VEL             NEES_POS               NEES_VEL')
  
    for alg in algorithms:

        print('data[\'{}_{}\'] = np.array([{}, {}, {}, {}])'.format(alg, num, np.sqrt(rmse_pos[alg] / data_num), np.sqrt(rmse_vel[alg] / data_num), nees_avg_pos[alg] / data_num, nees_avg_vel[alg] / data_num))

