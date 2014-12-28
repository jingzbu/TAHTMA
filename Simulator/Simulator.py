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
import matplotlib.pyplot as plt
from pylab import *
import pylab

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

from TAHTMA.util.HoeffdingRuleModelBased import *

# N = 2
# N = 3
N = 12

mu_0 = mu_ini(N**2)  # the initial distribution
Q = probability_matrix_Q(N)  # the original transition matrix
P = P_est(Q)  # the new transition matrix
PP = LA.matrix_power(P, 1000)
mu = PP[0, :]  # the actual stationary distribution; 1 x (N**2)

n_1 = 1000 * N * N  # the length of a sample path

# Get a sample path of the Markov chain with length n_1; this path is used to estimate the stationary distribution
x_1 = chain(mu, P, n_1)

mu_1 = mu_est(x_1, N)  # Get the estimated stationary distribution

Q_1 = Q_est(mu_1)  # Get the estimate of Q
P_1 = P_est(Q_1)  # Get the estimate of P
G_1 = G_est(Q_1)  # Get the estimate of the gradient
H_1 = H_est(mu_1)  # Get the estimate of the Hessian
Sigma = Sigma_est(P_1, mu_1)  # Get the estimate of the covariance matrix
SampNum = 1000
W = W_est(Sigma, SampNum)  # Get a sample path of W

beta = 0.001

eta_actual = []
eta = []
eta_Sanov = []
# n_range = range(2 * N * N, 20 * N * N + 5, N * N)
n_range = range(2 * N * N, 5 * N * N + 5, int(0.2 * N * N + 1))
# n_range = range(2 * N * N, 2 * N * N + 205, N * N)
for n in n_range:
    KL = []
    for i in range(0, SampNum):
        x = chain(mu, P, n)
        mu = np.reshape(mu, (N, N))
        KL.append(KL_est(x, mu))  # Get the actual relative entropy (K-L divergence)
    eta_actual.append(prctile(KL, 100 * (1 - beta)))
    # print eta_actual

    eta.append(HoeffdingRuleMarkov(beta, G_1, H_1, W, n))
    # print eta

    eta_Sanov.append(- log(beta) / n)
    # print eta_Sanov


eta_actual, = plt.plot(n_range, eta_actual, "r.-")
eta_wc, = plt.plot(n_range, eta, "bs-")
eta_Sanov, = plt.plot(n_range, eta_Sanov, "go-")

plt.legend([eta_actual, eta_wc, eta_Sanov], ["theoretical (actual) value", \
                                             "estimated by weak convergence analysis", \
                                             "estimated by Sanov's theorem"])
plt.xlabel('$n$ (number of samples)')
plt.ylabel('$\eta$ (threshold)')
# pylab.xlim(2 * N * N - 2, 20 * N * N + 5)
pylab.xlim(2 * N * N - 5, 5 * N * N + 10)
# savefig('/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results/N_2.eps')
# pylab.xlim(2 * N * N - 5, 2 * N * N + 205)
# savefig('/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results/N_3.eps')
savefig('/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results/N_12.eps')

plt.show()