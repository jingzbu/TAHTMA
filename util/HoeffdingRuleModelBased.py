#!/usr/bin/env python
""" A library of utility functions that will be used by Simulator
"""
from __future__ import absolute_import, division

import numpy as np
from numpy import linalg as LA
from math import sqrt
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

# #p = [0.5, 0.5]
# p = np.array([0.2, 0.7, 0.1])
# x = []
# for j in range(0, 20):
#    x.append(rand_x(p))
# print x

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
    
#mu_0 = np.array([0.2, 0.4, 0.4])
#
#Q = np.array([[ 0.18210804,  0.55936188,  0.25853007],
#              [ 0.41009471,  0.0436163,   0.54628898],
#              [ 0.46396811,  0.09889201,  0.43713988]])
#n = 100
#x = chain(mu_0, Q, n)
#print x
    
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

#dimQ = 4
#Q = probability_matrix_Q(dimQ)
#print Q

#mu = [[0.0775, 0.0338, 0.0281, 0.0974],
#      [0.0269, 0.0595, 0.0281, 0.1098],
#      [0.0437, 0.0251, 0.0231, 0.0618],
#      [0.0887, 0.1059, 0.0744, 0.1161]]
      
#mu = [[0.0790, 0.0260, 0.0461, 0.0890],
#     [0.0341, 0.0580, 0.0240, 0.1011],
#     [0.0307, 0.0279, 0.0314, 0.0724],
#     [0.0964, 0.1052, 0.0609, 0.1181]]

# mu = [[0.625, 0.125], [0.125, 0.125]]
#mu = [[0.2138, 0.2411], [0.2411, 0.3039]]
#mu = [[0.245000000000008, 0.210000000000007], [0.210000000000007, 0.335000000000010]]
#mu = [[0.2329, 0.1660], [0.1660, 0.4351]]

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
    
#dim_mu = 3
#mu = mu_ini(dim_mu)
#print mu

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
    for j in range(0, N):
        gama[j] = gama[j] / sum(gama)  # Normalize the estimated probability law
    gama = np.array(gama)
    # mu = gama.reshape(N, N, order='F')
    mu = gama.reshape(N, N)
    
    return mu

# x = [0, 1, 2, 2, 1, 2, 1,0, 1, 2, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1]
# N = 2
# mu_1 = mu_est(x, N)
# print mu_1

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


# Q = Q_est(mu)
# print Q
#print np.linalg.matrix_power(Q, 100)

def P_est(Q):
    """
    Estimate the new transition matrix P
    Example
    ----------------
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
    
# P = P_est(Q)
# print P
#print np.linalg.matrix_power(P, 1000)
#
#mu = np.array(mu)
#n, _ = mu.shape
#mu = mu.reshape(1, n**2)
#print mu

#N = 3
#mu = mu_ini(N**2)
#print mu
#n = 10000
#Q = probability_matrix_Q(N)
#P = P_est(Q)
#
#x = chain(mu, P, n)
#mu_1 = mu_est(x, N)
#print mu_1  

def G_est(mu):
    """
    Estimate the Gradient
    Example
    ----------------
    >>> mu = [[0.625,  0.125],  [0.125,  0.125]]
    >>> print G_est(mu)
    [[ 0.16666667  0.83333333  0.5         0.5       ]]
    """
    Q = Q_est(mu)  
    N, _ = Q.shape
    assert(N == _)
    alpha = 1 - Q
    G = alpha.reshape(1, N**2)
    
    return G

# G = G_est(mu)
# print G
    

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

# H = H_est(mu)
# print H
    

 
def Sigma_est(mu):
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

    Q = Q_est(mu)
    P = P_est(Q)
    
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
            
    # Essure Sigma to be symmetric
    Sigma = (1.0 / 2) * (Sigma + np.transpose(Sigma))  
    
    # Essure Sigma to be positive semi-definite
    D, V = LA.eig(Sigma)
    D = np.diag(D)
    Q, R = LA.qr(V)  
    for i in range(0, N**2):
        if D[i, i] < 0:
            D[i, i] = 0
    Sigma = np.dot(np.dot(Q, D), LA.inv(Q))
    return Sigma
 
#Sigma = Sigma_est(mu)
#print Sigma 
    

def HeffdingRuleMarkov(beta, SampNum, mu, FlowNum):
    """
    Estimate the covariance matrix of the empirical measure
    Example
    ----------------
    """  
    mu = np.array(mu) 
    N, _ = mu.shape
    assert(N == _)
    
    G = G_est(mu)  # Get the gradient estimate
    
    H = H_est(mu)  # Get the Hessian estimate
    
    Sigma = Sigma_est(mu)  # Get the covariance matrix estimate
    
    # Generate samples of W
    W_mean = np.zeros((1, N**2))
    W = np.random.multivariate_normal(W_mean[0,:], Sigma, (1, SampNum))
    
    # Estimate K-L divergence using 2nd-order Taylor expansion
    KL = []
    for j in range(0, SampNum):
        t = (1.0 / sqrt(FlowNum)) * np.dot(G, W[0, j, :]) + \
            (1.0 / 2) * (1.0 / FlowNum) * \
             np.dot(np.dot(W[0, j, :], H), W[0, j, :])
        KL.append(t)
    eta = prctile(KL, 100 * (1 - beta))
    return eta
 
    
#beta = 0.001
#SampNum = 1000
#FlowNum = 40
#eta = HeffdingRuleMarkov(beta, SampNum, mu, FlowNum)
#print eta
#
#sanov = -1.0 / FlowNum * log(beta) 
#print sanov







#N = 2
#n = 100
#Q = probability_matrix_Q(N)
#P = P_est(Q)
#mu = mu_ini(N**2)
#x = chain(mu, P, n)
    
