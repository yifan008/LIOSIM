#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as m

import random

from params import *

def rot_mtx(theta):
    mtx = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

    return mtx

class CentralizedSystem():

    def __init__(self, dataset, team_settings):

        self.dt = team_settings['dt']
        self.duration = team_settings['duration']

        self.history = list()

        self.dim = 5
        
        self.xyt = np.zeros(shape=(self.dim), dtype=float)
        self.cov =  np.zeros(shape=(self.dim, self.dim), dtype=float)

        self.init_team(dataset)

    def init_team(self, dataset):

        px = dataset[0] + 0.2
        py = dataset[1] + 0.2
        
        psi = dataset[2] + 0.01

        ptx = dataset[3] + 0.1
        pty = dataset[4] + 0.1

        # print(dataset)

        self.xyt = np.array((px, py, psi, ptx, pty))

        self.cov[0:2, 0:2] = np.identity(2) * 0.2**2
        self.cov[2, 2] = 0.01**2
        self.cov[3:5, 3:5] = 0.1**2

        self.history.append({'px': px, 'py': py, 'psi': psi, 'ptx': ptx, 'pty': pty, 'cov': self.cov})
