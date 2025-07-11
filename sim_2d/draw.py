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

    color_tables = {'ekf':'blue', 'ideal':'green', 'gt_t_p':'cyan', 'msc':'purple', 'tekf':'red', 'msc_ideal':'yellow', 'gt_t':'hotpink', 'odom':'yellow', 'gt':'purple', 'kdl2':'Moccasin', 'kdg2':'LavenderBlush', 'gt_p':'navy'}
    marker_tables = {'ekf':'o', 'ideal':'h', 'inv':'s', 'msc':'^', 'tekf':'p', 'msc_ideal':'3', 'kdp':'*', 'gt':'2', 'odom':'o', 'kdg2':'s', 'ukf':'*', 'kdl2': 's'}
    label_tables = {'ekf':'EKF', 'ideal':'Ideal', 'inv':'I-EKF', 'msc':'MSC', 'tekf':'T-EKF', 'ukf':'UKF', 'kdp':'T-EKF (T1)', 'odom':'ODOM', 'msc_ideal':'MSC_I'} 
    style_table = {'ekf':'-', 'ideal':'--', 'inv':'-.', 'ukf':'-', 'tekf':':', 'msc':':', 'msc_ideal':':', 'kdp':':', 'odom':':'}

    iter_num = ITER_NUM

    # individual RMSE plots
    rmse_pos = dict()
    rmse_psi = dict()
    nees_avg = dict()
    nees_pos = dict()
    nees_psi = dict()

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

        plt_psi = plt.figure(figsize=(12, 4))
        ax_psi = plt.subplot(121)
        plt.xlabel('t (s)', fontsize=12)
        plt.ylabel(r'$\psi \ \rm (m/s)$', fontsize=12)
        ax_psi.tick_params(axis='both', labelsize=12)

      for alg in algorithms:
        save_path = '../sim_results' + '/' + t + '/BearingVIO' + str(i+1) + '/'
        file_name = alg + '.npz'

        data = np.load(save_path + file_name)

        time_arr = data['t']

        pos_error = data['epos']
        psi_error = data['epsi']
        s_nees = data['nees']
        s_nees_pos = data['nees_pos']
        s_nees_psi = data['nees_psi']

        px_gt = data['px_gt']
        py_gt = data['py_gt']
        psi_gt = data['psi_gt']
        ptx_gt = data['ptx_gt']
        pty_gt = data['pty_gt']

        px_est = data['px_est']
        py_est = data['py_est']
        psi_est = data['psi_est']
        ptx_est = data['ptx_est']
        pty_est = data['pty_est']

        cov_p_est = data['cov_p_est']
        cov_psi_est = data['cov_psi_est']

        cov_px_est = cov_p_est[:, 0, 0]
        cov_py_est = cov_p_est[:, 1, 1]

        dp_x = px_est - px_gt
        dp_y = py_est - py_gt
        dpsi = psi_est - psi_gt

        N = time_arr.shape[0]

        if alg not in rmse_pos and alg not in rmse_psi:
          rmse_pos[alg] = pos_error
          rmse_psi[alg] = psi_error
          nees_avg[alg] = s_nees
          nees_pos[alg] = s_nees_pos
          nees_psi[alg] = s_nees_psi
        else:
          rmse_pos[alg] += pos_error
          rmse_psi[alg] += psi_error
          nees_avg[alg] += s_nees
          nees_pos[alg] += s_nees_pos
          nees_psi[alg] += s_nees_psi

        N_step = int(N / 1000)

        ax_px.plot(time_arr[range(0, N, N_step)], dp_x[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_px.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_px_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_px.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_px_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_py.plot(time_arr[range(0, N, N_step)], dp_y[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_py.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_py_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_py.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_py_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_psi.plot(time_arr[range(0, N, N_step)], dpsi[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linewidth=1.0, marker = marker_tables[alg], markerfacecolor='white', markersize=4.0)
        ax_psi.plot(time_arr[range(0, N, N_step)], 3*np.sqrt(cov_psi_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)
        ax_psi.plot(time_arr[range(0, N, N_step)], -3*np.sqrt(cov_psi_est[range(0, N, N_step)]), color=color_tables[alg], linestyle='-', linewidth=1.2)

        ax_px.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
        ax_py.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
        ax_psi.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})

        current_path = os.getcwd()
        Path(current_path + "/figures/bounds").mkdir(parents=True, exist_ok=True)

        fig_name = str(i+1) + '_p_xy' + '.png'
        plt_p.savefig(current_path + "/figures/bounds" + fig_name, dpi=600, bbox_inches='tight')

        fig_name = str(i+1) + '_psi' + '.png'
        plt_psi.savefig(current_path + "/figures/bounds" + fig_name, dpi=600, bbox_inches='tight')

    data_num = iter_num * N

    print('ALG:  RMSE_POS[m]       RMSE_VEL[rad]        NEES_POS       NEES_VEL')

    for alg in algorithms:
      print('{}: {} {} {} {}'.format(alg, np.sqrt(np.sum(rmse_pos[alg]) / data_num), np.sqrt(np.sum(rmse_psi[alg]) / data_num), np.sum(nees_pos[alg]) / data_num, np.sum(nees_psi[alg]) / data_num))

    # N_step = int(N / 1000)
    N_step = int(N / 1000)
 
    plt_rmse = plt.figure(figsize=(8, 4))

    plt_rmse_pos = plt.subplot(211)
    plt.ylabel(r'$\rm Pos. \ RMSE \ (m)$', fontsize=12)
    plt_rmse_pos.tick_params(axis='both', labelsize=12)

    plt_rmse_psi = plt.subplot(212)
    plt.xlabel('t (s)', fontsize=12)
    plt.ylabel(r'$\rm Ori. \ RMSE \ (rad)$', fontsize=12)
    plt_rmse_psi.tick_params(axis='both', labelsize=12)

    for alg in algorithms:
      plt_rmse_pos.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_pos[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)
      plt_rmse_psi.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_psi[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)        
      
    plt_rmse_pos.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})

    # plt_rmse_pos.set_ylim(0, 3)

    current_path = os.getcwd()
    Path(current_path + "/figures").mkdir(parents=True, exist_ok=True)
    plt_rmse.savefig(current_path + "/figures/rmse_vio" + '.png', dpi=600, bbox_inches='tight')
      
    plt_nees = plt.figure(figsize=(8,4))
 
    plt_nees_pos = plt.subplot(211)
    plt.ylabel(r'$\rm Pos. \ NEES$', fontsize=12)
    plt_nees_pos.tick_params(axis='both', labelsize=12)   
    
    plt_nees_psi = plt.subplot(212)
    plt.xlabel('t (s)', fontsize=12)
    plt.ylabel(r'$\rm Ori. \ NEES$', fontsize=12)
    plt_nees_psi.tick_params(axis='both', labelsize=12)

    ylim0 = ylim1 = 1.0
        
    for alg in algorithms:
      nees_pos_ = (nees_pos[alg] / iter_num)[range(0, N, N_step)]
      
      plt_nees_pos.plot(time_arr[range(0, N, N_step)], nees_pos_, color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)        
      plt_nees_psi.plot(time_arr[range(0, N, N_step)], (nees_psi[alg] / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.2, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=2)         

    plt_nees_pos.axhline(2, color='k', linestyle='--', linewidth = 0.5)
    plt_nees_psi.axhline(2, color='k', linestyle='--', linewidth = 0.5)
        
    plt_nees_pos.legend(loc = 'upper left', frameon=False, ncol = 4, prop={'family' : 'DejaVu Sans', 'size':10, 'weight':'medium'})
    # plt_nees_pos.set_ylim(0, 11)
    
    current_path = os.getcwd()

    plt_nees.savefig(current_path + "/figures/nees_vio" + '.png', dpi=600, bbox_inches='tight')
  
    # box-plot (position and orientation rmse)
    plt_rmse3 = plt.figure(figsize=(6, 4))
    plt_rmse_ax3 = plt.subplot(211)
    plt.ylabel('Pos. RMSE (m)', fontsize=14)
    plt_rmse_ax3.tick_params(axis='both', labelsize=14)

    plt_rmse_ax4 = plt.subplot(212)
    plt.ylabel(r'$\rm Ori. RMSE \ (rad)$', fontsize=14)
    plt_rmse_ax4.tick_params(axis='both', labelsize=14)

    data_pos = []
    data_psi = []
    labels = []
    colors = []

    for alg in algorithms:
        pos_rmse = np.sqrt(rmse_pos[alg] / iter_num)
        psi_rmse = np.sqrt(rmse_psi[alg] / iter_num)

        print('{}: {:.4f}/{:.4f}'.format(alg, np.sum(pos_rmse) / pos_rmse.shape[0], np.sum(psi_rmse) / psi_rmse.shape[0]))

        data_pos.append(pos_rmse)
        data_psi.append(psi_rmse)
        labels.append(label_tables[alg])
        colors.append(color_tables[alg])

    alg = 'ideal'

    # color_tables[alg]
    mean = {'linestyle':'-','color':color_tables[alg]}

    median = {'linestyle':'--','color':'purple'}

    showfilter = False
    shownortch = True

    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)
    bplot_ori = plt_rmse_ax4.boxplot(data_psi, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
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
    plt_rmse3.savefig(current_path + "/figures/rmse_box_vio" + '.png', dpi=600, bbox_inches='tight')
  
    plt_traj = plt.figure(figsize=(6, 4))
    plt_traj_ax = plt.gca()
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('y (m)', fontsize=14)
    plt_traj_ax.tick_params(axis='both', labelsize=14)

    save_path = '../sim_results' + '/' + t + '/BearingVIO' + str(1) + '/'

    style_list = ['-', '--', '-.', ':', ':', ':']
    color_list = ['orange', 'blue', 'red', 'lime', 'Magenta', 'hotpink']

    file_name = alg + '.npz'
    data = np.load(save_path + file_name)

    px_gt = data['px_gt']
    py_gt = data['py_gt']

    plt_traj_ax.plot(px_gt, py_gt, label='robot', linewidth=1, color=color_tables['gt_p'])
    plt_traj_ax.plot(px_gt[0], py_gt[0], color=color_tables['gt_p'], marker = 'o')

    for alg in algorithms:
      file_name = alg + '.npz'
      data = np.load(save_path + file_name)
    
      px_est = data['px_est']
      py_est = data['py_est']

      plt_traj_ax.plot(px_est[range(0, N, 5)], py_est[range(0, N, 5)], label=label_tables[alg], linewidth=1, color=color_tables[alg], linestyle = style_table[alg])
      plt_traj_ax.plot(px_est[0], py_est[0], color=color_tables[alg], marker = 'o')
        
    plt_traj_ax.legend(loc = 'upper left', frameon=True, ncol = 4, prop = {'size':10})

    plt.show()
