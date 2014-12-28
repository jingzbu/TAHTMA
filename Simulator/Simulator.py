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
from math import log
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

from TAHTMA.util.HoeffdingRuleModelBased import mu_ini, probability_matrix_Q, chain, HoeffdingRuleMarkov
from TAHTMA.util.HoeffdingRuleModelBased import P_est, Q_est, G_est, H_est, Sigma_est, W_est, mu_est, KL_est


N = 8
mu_0 = mu_ini(N**2)  # the initial distribution
Q = probability_matrix_Q(N)  # the original transition matrix
P = P_est(Q)  # the new transition matrix
PP = LA.matrix_power(P, 1000)
mu = PP[0, :]  # the actual stationary distribution; 1 x (N**2)


n_1 = 100 * N * N  # the length of a sample path

# Get a sample path of the Markov chain with length n_1; this path is used to estimate the stationary distribution
x_1 = chain(mu, P, n_1)

mu_1 = mu_est(x_1, N)  # Get the estimated stationary distribution

n = 100

SampNum = 1000
beta = 0.001

KL = []
for i in range(0, SampNum):
    x = chain(mu, P, n)
    mu = np.reshape(mu, (N, N))
    KL.append(KL_est(x, mu))  # Get the actual relative entropy (K-L divergence)
eta_actual = prctile(KL, 100 * (1 - beta))
print eta_actual

Q_1 = Q_est(mu_1)  # Get the estimate of Q
P_1 = P_est(Q_1)  # Get the estimate of P
G_1 = G_est(Q_1)  # Get the estimate of the gradient
H_1 = H_est(mu_1)  # Get the estimate of the Hessian
Sigma = Sigma_est(P_1, mu_1)  # Get the estimate of the covariance matrix
W = W_est(Sigma, SampNum)  # Get a sample path of W

eta = HoeffdingRuleMarkov(beta, G_1, H_1, W, n)
print eta

eta_Sanov = - log(beta) / n
print eta_Sanov