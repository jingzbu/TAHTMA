import numpy as np
from matplotlib.mlab import prctile
import matplotlib.pyplot as plt
import pylab
from pylab import *

dir = '/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results'

N = 5

eta = np.load(dir + '/N_%d/eta_KL.npz'%N)


n_range = eta['n_range']
eta_actual = eta['eta_actual']
eta_wc = eta['eta_wc']
eta_Sanov = eta['eta_Sanov']

eta.close()

print n_range
print eta_actual
print eta_wc
print eta_Sanov

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 26}

plt.figure(figsize=(11,9))

matplotlib.rc('font', **font)

eta_actual, = plt.plot(n_range, eta_actual, "r-", linewidth=3.5)
eta_wc, = plt.plot(n_range, eta_wc, "b-", linewidth=3.5)
eta_Sanov, = plt.plot(n_range, eta_Sanov, "g-", linewidth=3.5)

plt.legend([eta_actual, eta_wc, eta_Sanov], ["a proxy of actual value", \
                                                "estimated by WC result", \
                                                "estimated by Sanov's theorem"])
plt.xlabel('$n$ (number of samples)')
plt.ylabel('$\eta$ (threshold)')
# plt.title('Threshold ($\eta$) versus Number of samples ($n$)')
pylab.xlim(np.amin(n_range) - 2, np.amax(n_range) + 2)
pylab.ylim(0.0, 0.7)
# plt.grid()
savefig(dir + '/N_%d/eta_comp_N_%d.eps'%(N,N))
plt.show()
