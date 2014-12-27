#!/usr/bin/env python
"""
This file contains all the simulation experiments
"""

from __future__ import absolute_import, division

__author__ = "Jing Zhang"
__email__ = "jingzbu@gmail.com"
__status__ = "Development"

import numpy as np
from numpy import linalg as LA
from math import sqrt
from matplotlib.mlab import prctile

import sys
import os

ROOT = os.environ.get('tahtma_ROOT')
if ROOT is None:
    print('Please set <tahtma_ROOT> variable in bash.')
    sys.exit()
if not ROOT.endswith('TAHTMA'):
    print('Please set <tahtma_ROOT> path variable correctly. '
          'Please change the name of the <ROOT> folder to TAHTMA. Other name(s) '
          'will cause import problem.')
    sys.exit()
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT.rstrip('TAHTMA'))

from TAHTMA.util.HoeffdingRuleModelBased import mu_ini, probability_matrix_Q
from TAHTMA.util.HoeffdingRuleModelBased import P_est, chain, mu_est


N = 3
mu = mu_ini(N**2)
print mu
n = 10000
Q = probability_matrix_Q(N)
P = P_est(Q)
PP = LA.matrix_power(P, 1000)
print PP[0, :]

x = chain(mu, P, n)
mu_1 = mu_est(x, N)
print mu_1  

