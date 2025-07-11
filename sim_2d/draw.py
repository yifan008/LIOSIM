#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
from tqdm import tqdm

from pathlib import Path

from ekf import Centralized_EKF

from robot_system import *

import random

from matplotlib import markers, pyplot as plt

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

      # ax.grid(color="gray", linestyle=':', linewidth=1)

if __name__ == '__main__':
    # t = time.strftime("%Y-%m-%d %H:%M:%S")
    t = TIME_MARK

    # algorithms = ['ekf', 'fej', 'inv', 'kd', 'kdg', 'kdg2', 'kdl', 'kdl2']
    # algorithms = ['ukf', 'ekf', 'fej', 'inv', 'kdl']
    # algorithms = ['ukf', 'ekf', 'fej', 'inv', 'kdp', 'kdl']
    # algorithms = ['ideal', 'ekf', 'tekf', 'msc']
    
    # algorithms = ['ekf', 'fej', 'inv', 'kd', 'kdg', 'kdp', 'kdl']
    color_tables = {'ekf':'blue', 'ideal':'green', 'gt_t_p':'cyan', 'msc':'purple', 'tekf':'red', 'msc_ideal':'yellow', 'gt_t':'hotpink', 'odom':'yellow', 'gt':'purple', 'kdl2':'Moccasin', 'kdg2':'LavenderBlush', 'gt_p':'navy'}
    marker_tables = {'ekf':'o', 'ideal':'h', 'inv':'s', 'msc':'^', 'tekf':'p', 'msc_ideal':'3', 'kdp':'*', 'gt':'2', 'odom':'o', 'kdg2':'s', 'ukf':'*', 'kdl2': 's'}
    label_tables = {'ekf':'EKF', 'ideal':'Ideal', 'inv':'I-EKF', 'msc':'MSC', 'tekf':'T-EKF', 'ukf':'UKF', 'kdp':'T-EKF (T1)', 'odom':'ODOM', 'msc_ideal':'MSC_I'} 
    style_table = {'ekf':'-', 'ideal':'--', 'inv':'-.', 'ukf':'-', 'tekf':':', 'msc':':', 'msc_ideal':':', 'kdp':':', 'odom':':'}

    iter_num = ITER_NUM

    # individual RMSE plots
    rmse_pos = dict()
    rmse_vel = dict()
    nees_avg = dict()
    nees_pos = dict()
    nees_vel = dict()

    for i in range(iter_num):
      if DRAW_BOUNDS:
        plt_p = plt.figure(figsize=(12, 4))
        ax_px = plt.subplot(121)
        plt.xlabel('t (s)', fontsize=12)
        plt.ylabel(r'$\rm x \ (m)$', fontsize=12)
        ax_px.tick_params(axis='both', labelsize=12)

        ax_py = plt.subplot(122)
        plt.xlabel('t (s)', fontsize=12)
        plt.ylabel(r'$\rm y \ (m)$', fontsize=12)
        ax_py.tick_params(axis='both', labelsize=12)

        plt_v = plt.figure(figsize=(12, 4))
        ax_vx = plt.subplot(121)
        plt.xlabel('t (s)', fontsize=12)
        plt.ylabel(r'$\rm vx \ (m/s)$', fontsize=12)
        ax_vx.tick_params(axis='both', labelsize=12)

        ax_vy = plt.subplot(122)
        plt.xlabel('t (s)', fontsize=12)
        plt.ylabel(r'$\rm vy \ (m/s)$', fontsize=12)
        ax_vy.tick_params(axis='both', labelsize=12)

      for alg in algorithms:
        save_path = '../sim_results' + '/' + t + '/BearingTT' + str(i+1) + '/'
        file_name = alg + '.npz'

        data = np.load(save_path + file_name)

        time_arr = data['t']

        pos_error = data['epos']
        vel_error = data['evel']
        s_nees = data['nees']
        s_nees_pos = data['nees_pos']
        s_nees_vel = data['nees_vel']

        px_gt = data['px_gt']
        py_gt = data['py_gt']
        
        vx_gt = data['vx_gt']
        vy_gt = data['vy_gt']

        px_t_gt = data['px_t_gt']
        py_t_gt = data['py_t_gt']

        vx_t_gt = data['vx_t_gt']
        vy_t_gt = data['vy_t_gt']

        rx_est = data['rx_est']
        ry_est = data['ry_est']
        rvx_est = data['rvx_est']
        rvy_est = data['rvy_est']

        cov_p_est = data['cov_p_est']
        cov_v_est = data['cov_v_est']

        cov_px_est = cov_p_est[:, 0, 0]
        cov_py_est = cov_p_est[:, 1, 1]

        cov_vx_est = cov_v_est[:, 0, 0]
        cov_vy_est = cov_v_est[:, 1, 1]

        dp_x = rx_est - (px_t_gt - px_gt)
        dp_y = ry_est - (py_t_gt - py_gt)
        
        dv_x = rvx_est - (vx_t_gt - vx_gt)
        dv_y = rvy_est - (vy_t_gt - vy_gt)

        N = time_arr.shape[0]

        if alg not in rmse_pos and alg not in rmse_vel:
          rmse_pos[alg] = pos_error
          rmse_vel[alg] = vel_error
          nees_avg[alg] = s_nees
          nees_pos[alg] = s_nees_pos
          nees_vel[alg] = s_nees_vel
        else:
          rmse_pos[alg] += pos_error
          rmse_vel[alg] += vel_error
          nees_avg[alg] += s_nees
          nees_pos[alg] += s_nees_pos
          nees_vel[alg] += s_nees_vel

        N_step = int(N / 1000)

        ax_px.plot(time_arr[range(0, N, N_step)], dp_x[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_px.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_px_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_px.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_px_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_py.plot(time_arr[range(0, N, N_step)], dp_y[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_py.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_py_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_py.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_py_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_vx.plot(time_arr[range(0, N, N_step)], dv_x[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_vx.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_vx_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_vx.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_vx_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_vy.plot(time_arr[range(0, N, N_step)], dv_y[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_vy.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_vy_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_vy.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_vy_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_px.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
        ax_py.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
        ax_vx.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
        ax_vy.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})

        current_path = os.getcwd()
        Path(current_path + "/figures/bounds").mkdir(parents=True, exist_ok=True)

        fig_name = str(i+1) + '_p_xy' + '.png'
        plt_p.savefig(current_path + "/figures/bounds" + fig_name, dpi=600, bbox_inches='tight')

        fig_name = str(i+1) + '_v_xy' + '.png'
        plt_v.savefig(current_path + "/figures/bounds" + fig_name, dpi=600, bbox_inches='tight')

    data_num = iter_num * N

    print('ALG:  RMSE_POS[m]       RMSE_VEL[rad]        NEES_POS       NEES_VEL')

    for alg in algorithms:
      print('{}: {} {} {} {}'.format(alg, np.sqrt(np.sum(rmse_pos[alg]) / data_num), np.sqrt(np.sum(rmse_vel[alg]) / data_num), np.sum(nees_pos[alg]) / data_num, np.sum(nees_vel[alg]) / data_num))

    # N_step = int(N / 1000)
    N_step = int(N / 1000)
 
    plt_rmse = plt.figure(figsize=(8, 4))

    plt_rmse_pos = plt.subplot(211)
    
    plt.ylabel(r'$\rm Pos. \ RMSE \ (m)$', fontsize=12)
    plt_rmse_pos.tick_params(axis='both', labelsize=12)

    plt_rmse_vel = plt.subplot(212)
    plt.xlabel('t (s)', fontsize=12)
    plt.ylabel(r'$\rm Vel. \ RMSE \ (rad)$', fontsize=12)
    plt_rmse_vel.tick_params(axis='both', labelsize=12)

    for alg in algorithms:
      plt_rmse_pos.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_pos[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)
      plt_rmse_vel.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_vel[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)        
      
    plt_rmse_pos.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})

    # plt_rmse_pos.set_ylim(0, 3)

    current_path = os.getcwd()
    Path(current_path + "/figures").mkdir(parents=True, exist_ok=True)
    plt_rmse.savefig(current_path + "/figures/rmse_tt" + '.png', dpi=600, bbox_inches='tight')
      
    plt_nees = plt.figure(figsize=(8,4))
 
    plt_nees_pos = plt.subplot(211)

    plt.ylabel(r'$\rm Pos. \ NEES$', fontsize=12)
    plt_nees_pos.tick_params(axis='both', labelsize=12)   
    plt_nees_vel = plt.subplot(212)
    
    plt.xlabel('t (s)', fontsize=12)
    plt.ylabel(r'$\rm Vel. \ NEES$', fontsize=12)
    plt_nees_vel.tick_params(axis='both', labelsize=12)

    ylim0 = ylim1 = 1.0
        
    for alg in algorithms:
      nees_pos_ = (nees_pos[alg] / iter_num)[range(0, N, N_step)]
      
      plt_nees_pos.plot(time_arr[range(0, N, N_step)], nees_pos_, color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)        
      plt_nees_vel.plot(time_arr[range(0, N, N_step)], (nees_vel[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)         

    plt_nees_pos.axhline(2, color='k', linestyle='--', linewidth = 0.5)
    plt_nees_vel.axhline(2, color='k', linestyle='--', linewidth = 0.5)
        
    plt_nees_pos.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
    # plt_nees_pos.set_ylim(0, 11)
    
    current_path = os.getcwd()

    plt_nees.savefig(current_path + "/figures/nees_tt" + '.png', dpi=600, bbox_inches='tight')
  
    # box-plot (position and orientation rmse)
    plt_rmse3 = plt.figure(figsize=(6, 4))
    plt_rmse_ax3 = plt.subplot(211)
    plt.ylabel('Pos. RMSE (m)', fontsize=14)
    plt_rmse_ax3.tick_params(axis='both', labelsize=14)

    plt_rmse_ax4 = plt.subplot(212)
    plt.ylabel(r'$\rm Vel. RMSE \ (rad)$', fontsize=14)
    plt_rmse_ax4.tick_params(axis='both', labelsize=14)

    data_pos = []
    data_vel = []
    labels = []
    colors = []

    for alg in algorithms:
        pos_rmse = np.sqrt(rmse_pos[alg] / iter_num)
        vel_rmse = np.sqrt(rmse_vel[alg] / iter_num)

        print('{}: {:.4f}/{:.4f}'.format(alg, np.sum(pos_rmse) / pos_rmse.shape[0], np.sum(vel_rmse) / vel_rmse.shape[0]))

        data_pos.append(pos_rmse)
        data_vel.append(vel_rmse)
        labels.append(label_tables[alg])
        colors.append(color_tables[alg])

    alg = 'ekf'

    # color_tables[alg]
    mean = {'linestyle':'-','color':color_tables[alg]}

    median = {'linestyle':'--','color':'purple'}

    showfilter = False
    shownortch = True

    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)
    bplot_ori = plt_rmse_ax4.boxplot(data_vel, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)

    for alg in algorithms:
      # bplot_pos['boxes'][algorithms.index(alg)].set_facecolor(color_tables[alg])
      bplot_pos['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      bplot_ori['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      
      for i in range(2):
        bplot_pos['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
      
        bplot_pos['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])

    bplot_ori['means'][algorithms.index(alg)].set_color(color_tables[alg])

    current_path = os.getcwd()
    plt_rmse3.savefig(current_path + "/figures/rmse_box_tt" + '.png', dpi=600, bbox_inches='tight')
  
    plt_traj = plt.figure(figsize=(6, 4))
    plt_traj_ax = plt.gca()
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('y (m)', fontsize=14)
    plt_traj_ax.tick_params(axis='both', labelsize=14)

    save_path = '../sim_results' + '/' + t + '/BearingTT' + str(1) + '/'

    style_list = ['-', '--', '-.', ':', ':', ':']
    color_list = ['orange', 'blue', 'red', 'lime', 'Magenta', 'hotpink']

    file_name = alg + '.npz'
    data = np.load(save_path + file_name)

    px_gt = data['px_gt']
    py_gt = data['py_gt']

    px_t_gt = data['px_t_gt']
    py_t_gt = data['py_t_gt']

    plt_traj_ax.plot(px_gt, py_gt, label='pursuer', linewidth=1, color=color_tables['gt_p'])
    plt_traj_ax.plot(px_t_gt, py_t_gt, label='target', linewidth=1, color=color_tables['gt_t'])
    plt_traj_ax.plot(px_t_gt - px_gt, py_t_gt - py_gt, label='target-pursuer', linewidth=1, color=color_tables['gt_t_p'])
    
    plt_traj_ax.plot(px_gt[0], py_gt[0], color=color_tables['gt_p'], marker = 'o')
    plt_traj_ax.plot(px_t_gt[0], py_t_gt[0], color=color_tables['gt_t'], marker = 'o')
    plt_traj_ax.plot(px_t_gt[0] - px_gt[0], py_t_gt[0] - py_gt[0], color=color_tables['gt_t_p'], marker = 'o')

    for alg in algorithms:
      file_name = alg + '.npz'
      data = np.load(save_path + file_name)
    
      rx_est = data['rx_est']
      ry_est = data['ry_est']

      plt_traj_ax.plot((rx_est + px_gt)[range(0, N, 5)], (ry_est + py_gt)[range(0, N, 5)], label=label_tables[alg], linewidth=1, color=color_tables[alg], linestyle = style_table[alg])

      plt_traj_ax.plot((rx_est + px_gt)[0], (ry_est + py_gt)[0], color=color_tables[alg], marker = 'o')
        
    plt_traj_ax.legend(loc = 'upper left', frameon=True, ncol = 4, prop = {'size':10})

    plt.show()
