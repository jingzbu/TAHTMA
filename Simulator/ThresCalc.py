#!/usr/bin/env python
"""
This file contains all the threshold calculation methods
"""
from __future__ import absolute_import, division

__author__ = "Jing Zhang"
__email__ = "jingzbu@gmail.com"
__status__ = "Development"


import numpy as np
from numpy import linalg as LA
from math import log
from matplotlib.mlab import prctile

from TAHTMA.util.util import mu_ini, probability_matrix_Q, P_est, chain, mu_est, Q_est, G_est
from TAHTMA.util.util import H_est, Sigma_est, W_est, KL_est, HoeffdingRuleMarkov


class ThresBase(object):
    def __init__(self, N, beta, n, mu_0, mu, mu_1, P, G_1, H_1, W_1):
        self.N = N  # N is the row dimension of the original transition matrix Q
        self.beta = beta  # beta is the false alarm rate
        self.n = n  # n is the number of samples
        self.mu_0 = mu_0
        self.mu = mu
        self.mu_1 = mu_1
        self.P = P
        self.G_1 = G_1
        self.H_1 = H_1
        self.W_1 = W_1

class ThresActual(ThresBase):
    """ Computing the actual (theoretical) K-L divergence and threshold
    """
    def ThresCal(self):
        SampNum = 1000
        self.KL = []
        for i in range(0, SampNum):
            x = chain(self.mu_0, self.P, self.n)
            mu = np.reshape(self.mu, (self.N, self.N))
            self.KL.append(KL_est(x, mu))  # Get the actual relative entropy (K-L divergence)
        self.eta = prctile(self.KL, 100 * (1 - self.beta))
        KL = self.KL
        eta = self.eta
        return KL, eta

class ThresWeakConv(ThresBase):
    """ Estimating the K-L divergence and threshold by use of weak convergence
    """
    def ThresCal(self):
        self.KL, self.eta = HoeffdingRuleMarkov(self.beta, self.G_1, self.H_1, self.W_1, self.n)
        KL = self.KL
        eta = self.eta
        return KL, eta

class ThresSanov(ThresBase):
    """ Estimating the threshold by use of Sanov's theorem
    """
    def ThresCal(self):
        self.eta = - log(self.beta) / self.n
        eta = self.eta
        return eta