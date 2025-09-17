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

        self.vt_sigma = team_settings['vt_sigma']
        self.wt_sigma = team_settings['wt_sigma']

        self.bearing_sigma = team_settings['lidar_sigma']
        
        self.history = dict()

        self.history['p'] = list() 
        self.history['psi'] = list()
        self.history['pt'] = list()

        self.history['v'] = list()
        self.history['w'] = list()

        if SIM_FLAG == 1: # circle
            self.p = np.array([20.0, 0.0])
            self.psi = m.pi / 2.0
            self.pt = np.array([20, -20])
        elif SIM_FLAG == 2: # towards the landmark
            self.p = np.array([200.0, 0.0])
            self.psi = m.pi / 2.0
            self.pt = np.array([20, -20])
        else:
            pass

        self.history['p'].append(np.copy(self.p))    
        self.history['psi'].append(self.psi)
        self.history['pt'].append(np.copy(self.pt))

    def motion_trajectory(self):
        # motion trajectories generation
        t = self.dt

        while t <= self.duration:
            if SIM_FLAG == 1: # circle
                self.v = np.array([0, 1.0])
                self.w = 0.05
            elif SIM_FLAG == 2: # towards the landmark
                self.v = 0.5 * rot_mtx(self.psi).T @ (self.pt - self.p) / np.linalg.norm(self.pt - self.p)
                self.w = 0.01
            else:
                pass

            self.p += rot_mtx(self.psi) @ self.v * self.dt
            self.psi += self.w * self.dt

            # self.p += rot_mtx(self.psi) @ (self.v * self.dt + self.vt_sigma * np.random.randn(2, ))
            # self.psi += self.w * self.dt + self.wt_sigma * np.random.randn()

            self.pt = self.pt

            self.history['p'].append(np.copy(self.p))
            self.history['psi'].append(self.psi)            
            self.history['pt'].append(np.copy(self.pt))

            self.history['v'].append(np.copy(self.v))
            self.history['w'].append(self.w)
    
            t += self.dt

    def measurement_sim(self):
        # odometry information
        self.odometry = list()

        for k in range(len(self.history['v'])):
            # v_ct = self.history['v'][k] + self.vt_sigma * np.random.randn(2, )
            # w_ct = self.history['w'][k] + self.wt_sigma * np.random.randn()

            v_ct = self.history['v'][k]
            w_ct = self.history['w'][k]

            u = {'v': v_ct, 'w': w_ct}

            self.odometry.append(u)

        self.measurement = list()

        for k in range(len(self.history['p'])):
            
            p = self.history['p'][k] 
            pt = self.history['pt'][k]
            
            p_r = pt - p

            theta = np.arctan2(p_r[1], p_r[0]) + self.bearing_sigma * np.random.randn()

            self.measurement.append(theta)
        
        return self.odometry, self.measurement
        
if __name__ == '__main__':
    dt = STEP
    duration = DURATION
    iter_num = ITER_NUM
    vt_sigma = VT_SIGMA
    wt_sigma = WT_SIGMA
    bearing_sigma = BEARING_SIGMA

    # sim settings
    sim_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'vt_sigma': vt_sigma, 'wt_sigma': wt_sigma, 'bearing_sigma': bearing_sigma}

    sim = sim_mag(sim_settings)

    sim.motion_trajectory()

    '''
    
    length = len(sim.history['p'])
    
    # print(sim.history['p'][1])
    # print(length)
    
    px = np.zeros((length, ))
    py = np.zeros((length, ))
    
    for i in range(length):
      px[i] = sim.history['p'][i][0]
      py[i] = sim.history['p'][i][1]
    
    plt.figure()
    ax1 = plt.gca()
    _set_axis(ax1)
    
    plt.plot(sim.history['pt'][0][0], sim.history['pt'][0][1], color='green', marker='*')

    plt.plot(px, py, color='blue', label='pursuer', linestyle='--')         
    plt.plot(px[0:1], py[0:1], color='blue', marker='*')

    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    
    plt.legend(loc = 'upper right', frameon=True, ncol = 2, prop = {'size':6})

    plt.show()

   '''

    xyt_0 = np.zeros((5))

    gt = dict()

    xyt_0[0:2] = sim.history['p'][0]
    xyt_0[2] = sim.history['psi'][0]
    xyt_0[3:5] = sim.history['pt'][0]

    gt = dict()

    gt['p'] = sim.history['p']
    gt['psi'] = sim.history['psi']
    gt['pt'] = sim.history['pt']

    team_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'vt_sigma': vt_sigma, 'wt_sigma': wt_sigma, 'bearing_sigma': bearing_sigma}

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
                if alg == 'ideal':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    IDEALEKF = Ideal_EKF(robot_system, dataset)
                    IDEALEKF.run()
                    robot_system = IDEALEKF.robot_system
                elif alg == 'ekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    CENEKF = Centralized_EKF(robot_system, dataset)
                    CENEKF.run()
                    robot_system = CENEKF.robot_system
                elif alg == 'tekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    TEKF = T_EKF(robot_system, dataset)
                    TEKF.run()
                    robot_system = TEKF.robot_system
                else:
                    sys.exit('Invalid algorithm input!')
            
                result_alg[alg] = robot_system

            results.append(result_alg)

            pbar.update()

    # individual RMSE plots
    rmse_pos = dict()
    rmse_psi = dict()
    nees_avg_pos = dict()
    nees_avg_psi = dict()
        
    for alg in algorithms:
        rmse_pos[alg] = 0
        rmse_psi[alg] = 0
        nees_avg_pos[alg] = 0
        nees_avg_psi[alg] = 0

    t = time.strftime("%Y-%m-%d %H:%M:%S")

    print('TIME: {}'.format(t))

    # num = math.ceil(duration / dt + 1)
  
    num = len(sim.history['p'])
  
    px_gt = np.squeeze([sim.history['p'][k][0] for k in range(len(sim.history['p']))])
    py_gt = np.squeeze([sim.history['p'][k][1] for k in range(len(sim.history['p']))])

    psi_gt = np.squeeze([sim.history['psi'][k] for k in range(len(sim.history['psi']))])
    
    ptx_gt = np.squeeze([sim.history['pt'][k][0] for k in range(len(sim.history['pt']))])
    pty_gt = np.squeeze([sim.history['pt'][k][1] for k in range(len(sim.history['pt']))])
    
    time_arr = np.array([k * dt for k in range(num)])

    for alg in algorithms:

        pos_error = np.zeros((num, ))
        psi_error = np.zeros((num, ))
      
        nees_pos = np.zeros((num, ))
        nees_psi = np.zeros((num, ))

        for i in range(iter_num):
            s_nees = np.zeros((num, ))
            s_nees_pos = np.zeros((num, ))
            s_nees_psi = np.zeros((num, ))
            
            px_est = np.squeeze([results[i][alg].history[k]['px'] for k in range(len(results[i][alg].history))])
            py_est = np.squeeze([results[i][alg].history[k]['py'] for k in range(len(results[i][alg].history))])

            psi_est = np.squeeze([results[i][alg].history[k]['psi'] for k in range(len(results[i][alg].history))])

            ptx_est = np.squeeze([results[i][alg].history[k]['ptx'] for k in range(len(results[i][alg].history))])
            pty_est = np.squeeze([results[i][alg].history[k]['pty'] for k in range(len(results[i][alg].history))])
            
            cov_p_est = np.squeeze([results[i][alg].history[k]['cov'][0:2, 0:2] for k in range(len(results[i][alg].history))])
            cov_psi_est = np.squeeze([results[i][alg].history[k]['cov'][2, 2] for k in range(len(results[i][alg].history))])
            cov_pt_est = np.squeeze([results[i][alg].history[k]['cov'][3:5, 3:5] for k in range(len(results[i][alg].history))])

            epos = (np.array(px_est) - np.array(px_gt)) ** 2 + (np.array(py_est) - np.array(py_gt)) ** 2
            pos_error += epos
            epsi = (np.array(psi_est) - np.array(psi_gt)) ** 2 
            psi_error += epsi

            for k in range(len(results[i][alg].history)):
                cov = results[i][alg].history[k]['cov']

                dp = np.array((px_est[k] - (px_gt[k]), py_est[k] - (py_gt[k])))
                dpsi = (psi_est[k] - psi_gt[k])

                nees_pos[k] += dp.T @ np.linalg.inv(cov[0:2, 0:2]) @ dp
                nees_psi[k] += dpsi **2 / cov[2, 2]

                ds = np.hstack((dp, dpsi))
                s_nees[k] = ds.T @ np.linalg.inv(cov[0:3, 0:3]) @ ds
                s_nees_pos[k] = dp.T @ np.linalg.inv(cov[0:2, 0:2]) @ dp
                s_nees_psi[k] = dpsi **2 / cov[2, 2]
                
            save_path = '../sim_results' + '/' + t + '/BearingVIO' + str(i+1) + '/' 

            Path(save_path).mkdir(parents=True, exist_ok=True)

            file_name = alg + '.npz'

            np.savez(save_path + file_name, t = time_arr, px_est = px_est, py_est = py_est, psi_est = psi_est, ptx_est = ptx_est, pty_est = pty_est,
                    px_gt = px_gt, py_gt = py_gt, ptx_gt = ptx_gt, pty_gt = pty_gt, psi_gt = psi_gt, 
                    cov_p_est = cov_p_est, cov_psi_est = cov_psi_est, epos = epos, epsi = epsi, 
                    nees = s_nees, nees_psi = s_nees_psi, nees_pos = s_nees_pos)

        rmse_pos[alg] += np.sum(pos_error)
        rmse_psi[alg] += np.sum(psi_error)

        nees_avg_pos[alg] += np.sum(nees_pos)
        nees_avg_psi[alg] += np.sum(nees_psi)
        
        N = time_arr.shape[0]

    data_num = iter_num * N

    print('ALG: RMSE_POS             RMSE_PSI             NEES_POS               NEES_PSI')
  
    for alg in algorithms:

        print('data[\'{}_{}\'] = np.array([{}, {}, {}, {}])'.format(alg, num, np.sqrt(rmse_pos[alg] / data_num), np.sqrt(rmse_psi[alg] / data_num), nees_avg_pos[alg] / data_num, nees_avg_psi[alg] / data_num))

