import numpy as np
from matplotlib.mlab import prctile
import matplotlib.pyplot as plt
import pylab
from pylab import *

dir = '/home/jzh/Dropbox/Research/Anomaly_Detection/Experimental_Results'

eta = np.load(dir + '/N_7/eta_KL.npz')


n_range = eta['n_range']
eta_actual = eta['eta_actual']
eta_wc = eta['eta_wc']
eta_Sanov = eta['eta_Sanov']

eta.close()

print n_range
print eta_actual
print eta_wc
print eta_Sanov

eta_actual, = plt.plot(n_range, eta_actual, "ro-")
eta_wc, = plt.plot(n_range, eta_wc, "bs-")
eta_Sanov, = plt.plot(n_range, eta_Sanov, "g^-")

plt.legend([eta_actual, eta_wc, eta_Sanov], ["$\eta_n$ (actual value)", \
                                                "$\eta_n^{wc}$ (estimated by WC result)", \
                                                "$\eta_n^{sv}$ (estimated by Sanov's theorem)"])
plt.xlabel('$n$ (number of samples)')
plt.ylabel('$\eta$ (threshold)')
# plt.title('Threshold ($\eta$) versus Number of samples ($n$)')
pylab.xlim(np.amin(n_range) - 3, np.amax(n_range) + 3)
# pylab.ylim(0, 0.4)
savefig(dir + '/N_7/eta_comp_N_7.eps')
plt.show()