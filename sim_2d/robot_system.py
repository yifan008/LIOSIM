#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as m

import random

from params import *

class CentralizedSystem():

    def __init__(self, dataset, team_settings):

        self.dt = team_settings['dt']
        self.duration = team_settings['duration']

        self.history = list()

        self.dim = 4
        
        self.xyt = np.zeros(shape=(self.dim), dtype=float)
        self.cov =  np.zeros(shape=(self.dim, self.dim), dtype=float)

        self.init_team(dataset)

    def init_team(self, dataset):

        x = dataset[0] + 1
        y = dataset[1] + 1

        vx = dataset[2] + 0.2
        vy = dataset[3] + 0.5

        # print(dataset)

        self.xyt = np.array((x, y, vx, vy))

        self.cov[0:2, 0:2] = np.identity(2) * 1
        self.cov[2, 2] = 0.2**2
        self.cov[3, 3] = 0.5**2
        # self.cov[2:4, 2:4] = np.identity(2) * 0.5**2

        self.history.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'cov': self.cov})
