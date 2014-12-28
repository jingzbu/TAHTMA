#!/usr/bin/env python
""" A library of utility functions that will be used by Simulator
"""
from __future__ import absolute_import, division

import numpy as np
from numpy import linalg as LA
from math import sqrt, log
from matplotlib.mlab import prctile


def rand_x(p):
    """
    Generate a random variable in 0, 1, ..., (N-1) given a distribution vector p
    N is the dimension of p
    ----------------
    Example
    ----------------
    p = [0.5, 0.5]
    x = []
    for j in range(0, 20):
        x.append(rand_x(p))
    print x
    ----------------
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
    """
    p = np.array(p)
    N = p.shape
    #assert(abs(sum(p) - 1) < 1e-5)
    u = np.random.rand(1, 1)
    i = 0
    s = p[0]
    while (u > s).all() and i < N[0]-1:
        i = i + 1
        s = s + p[i]
    index = i

    return index

def chain(mu_0, Q, n):
    """
    Simulate a Markov chain on {0, 1, ..., n-1} given an initial
    distribution and a transition matrix
    ----------------
    The program assumes that the states are labeled 0, 1, ..., N-1
    ----------------
    mu_0: the initial distribution, a 1 x N vector
    Q: the transition matrix, N x N
    ----------------
    n: the number of samples
    N: the total number of states of the chain
    ----------------
    Example
    ----------------
    """
    #assert(abs(sum(mu_0) - 1) < 1e-5)
    Q = np.array(Q)
    N, _ = Q.shape
    assert(N == _)
    x = [0] * n
    x[0] = rand_x(mu_0)
    for i in range(1, n-1):
       x[i+1] = rand_x(Q[x[i], :])

    return x

def probability_matrix_Q(dimQ):
    """
    Randomly generate the original transition matrix Q
    dimQ: the row dimension of Q
    Example
    ----------------
    """
    weight = []
    for i in range(0, dimQ):
        t = np.random.randint(1, high=dimQ, size=dimQ)
        weight.append(t)
    Q = np.random.rand(dimQ, dimQ)
    for i in range(0, dimQ):
        Q[i, :] = weight[i] * Q[i, :]
        Q[i, :] = Q[i, :] / (sum(Q[i,:]))
    N, _ = Q.shape
    assert(N == _)

    return Q

def mu_ini(dim_mu):
    """
    Randomly generate the initial distribution mu
    dim_mu: the dimension of mu
    Example
    ----------------
    """
    weight = []
    for i in range(0, dim_mu):
        t = np.random.randint(1, high=dim_mu, size=1)
        weight.append(t)
    mu = np.random.rand(1, dim_mu)
    for i in range(0, dim_mu):
        mu[0, i] = (weight[i][0]) * mu[0, i]
    mu = mu / (sum(mu[0, :]))
    assert(abs(sum(mu[0, :]) - 1) < 1e-5)

    return mu

def mu_est(x, N):
    """
    Estimate the stationary distribution mu
    x: a sample path of the chain
    N: the obtained mu should be an N x N matrix
    Example
    ----------------
    >>> x = [0, 1, 2, 2, 1, 2, 1,0, 1, 2, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1]
    >>> N = 2
    >>> mu_1 = mu_est(x, N)
    >>> print mu_1
    [[ 0.11111111  0.40740741]
     [ 0.33333333  0.14814815]]
    """
    eps = 1e-8
    gama = [0] * N * N
    for j in range(0, N**2):
        gama[j] = (x.count(j)) / (len(x))
        if gama[j] < eps:
            gama[j] = eps
    for j in range(0, N**2):
        gama[j] = gama[j] / sum(gama)  # Normalize the estimated probability law
    gama = np.array(gama)
    mu = gama.reshape(N, N)

    return mu

def KL_est(x, mu):
    """
    Estimate the relative entropy (K-L divergence)
    x: a sample path of the chain
    mu: the true stationary distribution; an N x N matrix
    """
    mu = np.array(mu)
    N, _ = mu.shape
    assert(N == _)

    # Compute the empirical distribution
    eps = 1e-8
    gama = [0] * N * N
    for j in range(0, N**2):
        gama[j] = (x.count(j)) / (len(x))
        if gama[j] < eps:
            gama[j] = eps
    for j in range(0, N**2):
        gama[j] = gama[j] / sum(gama)  # Normalize the estimated probability law
    gama = np.array(gama)
    gama = np.reshape(gama, (N, N))

    # Compute the relative entropy (K-L divergence)
    d = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            d[i, j] = gama[i, j] * (log(gama[i, j] / (sum(gama[i, :]))) - log(mu[i, j] / (sum(mu[i, :]))))
    KL = sum(sum(d))

    return KL

def Q_est(mu):
    """
    Estimate the original transition matrix Q
    Example
    ----------------
    >>> mu = [[0.625,  0.125],  [0.125,  0.125]]
    >>> print Q_est(mu)
    [[ 0.83333333  0.16666667]
     [ 0.5         0.5       ]]
    """

    mu = np.array(mu)
    N, _ = mu.shape
    assert(N == _)

    pi = np.sum(mu, axis=1)

    Q = mu / np.dot( pi.reshape(-1, 1), np.ones((1, N)) )

    return Q

def P_est(Q):
    """
    Estimate the new transition matrix P
    """
    Q = np.array(Q)
    N, _ = Q.shape
    assert(N == _)

    P1 = np.zeros((N, N**2))

    for j in range(0, N):
        for i in range(j * N, (j + 1) * N):
            P1[j, i] = Q[j, i - j * N]

    P = np.tile(P1, (N, 1))

    return P

def G_est(Q):
    """
    Estimate the Gradient
    Example
    ----------------
    >>> mu = [[0.625,  0.125],  [0.125,  0.125]]
    >>> print G_est(mu)
    [[ 0.16666667  0.83333333  0.5         0.5       ]]
    """
    N, _ = Q.shape
    assert(N == _)
    alpha = 1 - Q
    G = alpha.reshape(1, N**2)

    return G

def H_est(mu):
    """
    Estimate the Hessian
    Example
    ----------------
    >>> mu = [[0.625,  0.125],  [0.125,  0.125]]
    >>> print H_est(mu)
    [[ 0.04444444 -0.22222222  0.          0.        ]
     [-1.11111111  5.55555556  0.          0.        ]
     [ 0.          0.          2.         -2.        ]
     [ 0.          0.         -2.          2.        ]]
    """
    mu = np.array(mu)
    N, _ = mu.shape
    assert(N == _)

    H = np.zeros((N, N, N, N))

    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                for l in range(0, N):
                    if k != i:
                        H[i, j, k, l] = 0
                    elif l == j:
                        H[i, j, k, l] = 1 / mu[i, j] - 1 / (sum(mu[i, :])) - \
                        ((sum(mu[i, :])) - mu[i, j]) / ((sum(mu[i, :]))**2)
                    else:
                        H[i, j, k, l] = - ((sum(mu[i, :])) - mu[i, j]) / \
                            ((sum(mu[i, :]))**2)

    H = np.reshape(H, (N**2, N**2))

    return H

def Sigma_est(P, mu):
    """
    Estimate the covariance matrix of the empirical measure
    Example
    ----------------
    >>> mu = [[0.625,  0.125],  [0.125,  0.125]]
    >>> print Sigma_est(mu)
    [[ 0.625   -0.15625 -0.15625 -0.3125 ]
     [-0.15625  0.0625   0.0625   0.03125]
     [-0.15625  0.0625   0.0625   0.03125]
     [-0.3125   0.03125  0.03125  0.25   ]]
    """
    mu = np.array(mu)
    N, _ = mu.shape
    assert(N == _)

    muVec = np.reshape(mu, (1, N**2))

    I = np.matrix(np.identity(N**2))

    M = 1000

    PP = np.zeros((M, N**2, N**2))
    for m in range(1, M):
        PP[m] = LA.matrix_power(P, m)

    series = np.zeros((1, M))

    Sigma = np.zeros((N**2, N**2))
    for i in range(0, N**2):
        for j in range(0, N**2):
            for m in range(1, M):
                series[0, m] = muVec[0, i] * (PP[m][i, j] - muVec[0, j]) + \
                                muVec[0, j] * (PP[m][j, i] - muVec[0, i])
            Sigma[i, j] = muVec[0, i] * (I[i, j] - muVec[0, j]) + \
                            sum(series[0, :])

    # Ensure Sigma to be symmetric
    Sigma = (1.0 / 2) * (Sigma + np.transpose(Sigma))

    # Ensure Sigma to be positive semi-definite
    D, V = LA.eig(Sigma)
    D = np.diag(D)
    Q, R = LA.qr(V)
    for i in range(0, N**2):
        if D[i, i] < 0:
            D[i, i] = 0
    Sigma = np.dot(np.dot(Q, D), LA.inv(Q))
    return Sigma

def W_est(Sigma, SampNum):
    """
    Generate samples of W
    ----------------
    Sigma: the covariance matrix; an N x N matrix
    SampNum: the length of the sample path
    """
    N, _ = Sigma.shape
    assert(N == _)
    W_mean = np.zeros((1, N))
    W = np.random.multivariate_normal(W_mean[0, :], Sigma, (1, SampNum))

    return W

def HoeffdingRuleMarkov(beta, G, H, W, FlowNum):
    """
    Estimate the covariance matrix of the empirical measure
    ----------------
    beta: the false alarm rate
    G: the gradient
    H: the Hessian
    W: a sample path of the Gaussian empirical measure
    FlowNum: the number of flows
    ----------------
    """
    _, SampNum, _ = W.shape

    # Estimate K-L divergence using 2nd-order Taylor expansion
    KL = []
    for j in range(0, SampNum):
        t = (1.0 / sqrt(FlowNum)) * np.dot(G, W[0, j, :]) + \
                (1.0 / 2) * (1.0 / FlowNum) * \
                    np.dot(np.dot(W[0, j, :], H), W[0, j, :])
        KL.append(t)
    eta = prctile(KL, 100 * (1 - beta))

    return eta