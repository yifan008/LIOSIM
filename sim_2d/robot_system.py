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

        self.dim = 3
        
        self.xyt = np.zeros(shape=(self.dim), dtype=float)
        self.cov =  np.zeros(shape=(self.dim, self.dim), dtype=float)

        self.init_team(dataset)

    def init_team(self, dataset):

        px = dataset[0] + 0.02
        py = dataset[1] + 0.02
        psi = dataset[2] + 0.001

        self.xyt = np.array((px, py, psi))

        self.cov[0:2, 0:2] = np.identity(2) * 0.02**2
        self.cov[2, 2] = 0.001**2

        self.history.append({'px': px, 'py': py, 'psi': psi, 'cov': self.cov})
