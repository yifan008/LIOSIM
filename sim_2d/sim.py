#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm

from pathlib import Path

from bekf import B_EKF
from sekf import S_EKF
from tekf import T_EKF
from nekf import N_EKF
from idekf import ID_EKF
from schur import SCHUR_EKF
from ciekf import CI_EKF
from iciekf import ICI_EKF
from schmidt import SCHMIDT_EKF

from robot_system import *
from params import *

import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

M_PI = 3.1415926

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

        self.lidar_sigma = team_settings['lidar_sigma']
        
        FLAG_LANDMARK = 0
        
        self.history = dict()

        self.history['p'] = list() 
        self.history['psi'] = list()
        self.history['pt'] = list()

        self.history['v'] = list()
        self.history['w'] = list()

        self.p = np.array([2.0, 0.0]) # np.random.uniform(-5, 5, (2, 1))
        self.psi = 0.0 # np.random.uniform(-2, 2) 
        
        self.v = np.array([0.0, 0.2])
        self.w = 0.02 # np.random.uniform(-0.05, 0.05)
    
        if self.v[1] / self.w > 3.0:
            self.v[1] = 3.0 * self.w
      
        self.history['p'].append(np.copy(self.p))
        self.history['psi'].append(self.psi)

        self.landmark_num = 20

        r = self.v[1] / self.w
        rmin = 0.5
        rmax = 4.0

        for i in range(self.landmark_num):
            if FLAG_LANDMARK == 0: # limit range
                if np.random.uniform(0, 1) >.5:
                    rho = rmin + (rmax - rmin) * np.random.uniform(0, 1)
                    th = 2 * M_PI * np.random.uniform(0, 1)
                else:
                    rho = - rmax + (rmax - rmin) * np.random.uniform(0, 1)
                    th = 2 * M_PI * np.random.uniform(0, 1)
                    
                landmark_pt = np.array([rho * math.cos(th), rho * math.sin(th)])
                self.history['pt'].append(np.copy(landmark_pt))

            elif FLAG_LANDMARK == 1: # randomly
                landmark_pt = np.random.uniform(-8, 8, (2, ))
                self.history['pt'].append(np.copy(landmark_pt))
             
                # self.pt[i, :] = np.random.uniform(-8, 8, (2, ))
            else:
                pass
        
    def motion_trajectory(self):
        # motion trajectories generation
        t = self.dt

        while t <= self.duration:

            self.p += rot_mtx(self.psi) @ self.v * self.dt
            self.psi += self.w * self.dt
    
            # self.p += rot_mtx(self.psi) @ (self.v * self.dt + self.vt_sigma * np.random.randn(2, ))
            # self.psi += self.w * self.dt + self.wt_sigma * np.random.randn()

            self.history['p'].append(np.copy(self.p))
            self.history['psi'].append(self.psi)            

            self.history['v'].append(np.copy(self.v))
            self.history['w'].append(self.w)
    
            t += self.dt

    def measurement_sim(self):
        # odometry information
        self.odometry = list()

        # print('pv: ', len(self.history['p']), len(self.history['v']))

        for k in range(len(self.history['v'])):
            v_ct = self.history['v'][k] + self.vt_sigma * np.random.randn(2, )
            w_ct = self.history['w'][k] + self.wt_sigma * np.random.randn()

            u = {'v': v_ct, 'w': w_ct}

            self.odometry.append(u)

        self.measurement = list()

        for k in range(len(self.history['p'])):
            
            landmark_observation = list()

            for i in range(self.landmark_num):
                p = self.history['p'][k]
                psi = self.history['psi'][k]
                
                pt = self.history['pt'][i]

                dist = np.linalg.norm(pt - p)
                
                if dist < 1.2:
                    p_r = rot_mtx(psi).T @ (pt - p) + self.lidar_sigma * np.random.randn(2, )
                    landmark_observation.append({'obs': p_r, 'id': i})
            
            # print(landmark_observation)
            # print(len(landmark_observation))
            self.measurement.append(landmark_observation)
                    
        return self.odometry, self.measurement

def plot_traj(sim):
    length = len(sim.history['p'])
    
    px = np.zeros((length, ))
    py = np.zeros((length, ))
    
    for i in range(length):
      px[i] = sim.history['p'][i][0]
      py[i] = sim.history['p'][i][1]
    
    plt.figure()
    ax1 = plt.gca()
    _set_axis(ax1)
    
    plt.plot(px, py, color='blue', label='pursuer', linestyle='--')         
    plt.plot(px[0:1], py[0:1], color='blue', marker='*')

    for i in range(sim.landmark_num):
        plt.plot(sim.history['pt'][i][0], sim.history['pt'][i][1], color='green', marker='*')

    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    
    plt.legend(loc = 'upper right', frameon=True, ncol = 2, prop = {'size':6})

    print('Plotting...')
    plt.show()
    print('Done plotting.')

    # print('simulation ends ...')
    
if __name__ == '__main__':
    dt = STEP
    duration = DURATION
    iter_num = ITER_NUM
    vt_sigma = VT_SIGMA
    wt_sigma = WT_SIGMA
    lidar_sigma = LIDAR_SIGMA
    
    # sim settings
    sim_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'vt_sigma': vt_sigma, 'wt_sigma': wt_sigma, 'lidar_sigma': lidar_sigma}

    print('simulation starts ...')
    
    sim = sim_mag(sim_settings)

    sim.motion_trajectory()
    
    if DRAW_TRAJECTORY:
        plot_traj(sim)
    
    xyt_0 = np.zeros((3, ))
    xyt_0[0:2] = sim.history['p'][0]
    xyt_0[2] = sim.history['psi'][0]

    gt = dict()
    gt['p'] = sim.history['p']
    gt['psi'] = sim.history['psi']
    gt['pt'] = sim.history['pt']

    team_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'vt_sigma': vt_sigma, 'wt_sigma': wt_sigma, 'lidar_sigma': lidar_sigma}

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
                if alg == 'bekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    BEKF = B_EKF(robot_system, dataset)
                    BEKF.run()
                    robot_system = BEKF.robot_system
                elif alg == 'sekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    SEKF = S_EKF(robot_system, dataset)
                    SEKF.run()
                    robot_system = SEKF.robot_system
                elif alg == 'idekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    IDEKF = ID_EKF(robot_system, dataset)
                    IDEKF.run()
                    robot_system = IDEKF.robot_system
                elif alg == 'tekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    TEKF = T_EKF(robot_system, dataset)
                    TEKF.run()
                    robot_system = TEKF.robot_system
                elif alg == 'nekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    NEKF = N_EKF(robot_system, dataset)
                    NEKF.run()
                    robot_system = NEKF.robot_system
                elif alg == 'schur':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    SCHUREKF = SCHUR_EKF(robot_system, dataset)
                    SCHUREKF.run()
                    robot_system = SCHUREKF.robot_system
                elif alg == 'ciekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    CIEKF = CI_EKF(robot_system, dataset)
                    CIEKF.run()
                    robot_system = CIEKF.robot_system
                elif alg == 'iciekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    ICIEKF = ICI_EKF(robot_system, dataset)
                    ICIEKF.run()
                    robot_system = ICIEKF.robot_system
                elif alg == 'schmidt':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    SCHMIDT_EKF = SCHMIDT_EKF(robot_system, dataset)
                    SCHMIDT_EKF.run()
                    robot_system = SCHMIDT_EKF.robot_system
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

    # t = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Use underscores or dashes instead of colons
    t = time.strftime("%Y-%m-%d_%H-%M-%S")

    print('TIME: {}'.format(t))

    # num = math.ceil(duration / dt + 1)
  
    num = len(sim.history['p'])
  
    px_gt = np.squeeze([sim.history['p'][k][0] for k in range(len(sim.history['p']))])
    py_gt = np.squeeze([sim.history['p'][k][1] for k in range(len(sim.history['p']))])
    psi_gt = np.squeeze([sim.history['psi'][k] for k in range(len(sim.history['psi']))])
    
    pt_gt = np.squeeze([sim.history['pt'][i] for i in range(len(sim.history['pt']))])

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

            cov_p_est = np.squeeze([results[i][alg].history[k]['cov'][0:2, 0:2] for k in range(len(results[i][alg].history))])
            cov_psi_est = np.squeeze([results[i][alg].history[k]['cov'][2, 2] for k in range(len(results[i][alg].history))])

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
            
            # Get the current script's directory
            current_dir = Path(__file__).parent
            
            # Get the parent directory (upper level)
            parent_dir = current_dir.parent

            # Define your save_path under the parent directory
            save_path = parent_dir / 'sim_results' / t / f'LIO{i+1}'

            # Create the directory
            save_path.mkdir(parents=True, exist_ok=True)

            file_name = alg + '.npz'

            np.savez(str(save_path / file_name), t = time_arr, px_est = px_est, py_est = py_est, psi_est = psi_est,
                    px_gt = px_gt, py_gt = py_gt, psi_gt = psi_gt, pt_gt = pt_gt, epos = epos, epsi = epsi,
                    cov_p_est = cov_p_est, cov_psi_est = cov_psi_est,
                    nees = s_nees, nees_psi = s_nees_psi, nees_pos = s_nees_pos)

        rmse_pos[alg] += np.sum(pos_error)
        rmse_psi[alg] += np.sum(psi_error)

        nees_avg_pos[alg] += np.sum(nees_pos)
        nees_avg_psi[alg] += np.sum(nees_psi)
        
        N = time_arr.shape[0]

    data_num = iter_num * N

    print('ALG: RMSE_POS              RMSE_PSI               NEES_POS               NEES_PSI')
  
    for alg in algorithms:

        print('{}_{}: {}, {}, {}, {}'.format(alg, num, np.sqrt(rmse_pos[alg] / data_num), np.sqrt(rmse_psi[alg] / data_num), nees_avg_pos[alg] / data_num, nees_avg_psi[alg] / data_num))

    print('simulation ends ...')
    