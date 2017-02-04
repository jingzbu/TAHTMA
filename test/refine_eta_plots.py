import numpy as np
from matplotlib.mlab import prctile
import matplotlib.pyplot as plt
import pylab
from pylab import *

fig_dir = '/home/jzh/Dropbox/tsp_rev_refs/figs/'

N_list = [2, 3, 4, 5, 6, 7, 8]
beta_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
k_list = range(21)[1:]

# N_list = [2]
# beta_list = [0.1]

save_data_dict = {}

for N in N_list:
    for beta in beta_list:
        for k in k_list:
            eta = np.load(fig_dir + 'eta_KL_mb_N_%s_beta_%s_k_%s.eps.npz'%(N, beta, k))

            n_range = eta['n_range']
            eta_actual = eta['eta_actual']
            eta_wc_1 = eta['eta_wc_1']
            eta_wc_2 = eta['eta_wc_2']
            eta_Sanov = eta['eta_Sanov']

            eta.close()

            # print n_range
            # print eta_actual
            # print eta_wc
            # print eta_Sanov

            font = {'family' : 'normal',
                    'weight' : 'normal',
                    'size'   : 28}

            plt.figure(figsize=(11.5, 9))

            matplotlib.rc('font', **font)

            eta_actual, = plt.plot(n_range, eta_actual, "r-o", linewidth=3.5, markersize=8)
            eta_wc_1, = plt.plot(n_range, eta_wc_1, "b-s", linewidth=3.5, markersize=6.5)
            eta_wc_2, = plt.plot(n_range, eta_wc_2, "m-v", linewidth=3.5, markersize=8)
            eta_Sanov, = plt.plot(n_range, eta_Sanov, "g-^", linewidth=3.5, markersize=8)

            plt.legend([eta_actual, eta_wc_1, eta_wc_2, eta_Sanov], [r"$\eta^{*}_{n,\beta}(N)$", \
                                                                     r"$\eta_{n,\beta}^{\mathrm{wc}}(N)$", \
                                                                     r"$\bar{\eta}_{n,\beta}^{\mathrm{wc}}(N)$", \
                                                                     r"$\eta_{n,\beta}^{\mathrm{sv}}(N)$"])
            plt.xlabel('$n$ (sample size)')
            plt.ylabel('$\eta$ (threshold)')

            # plt.grid()
            # plt.title('Threshold ($\eta$) versus Number of samples ($n$)')
            # pylab.xlim(np.amin(n_range) - 1, np.amax(n_range) + 1)
            # pylab.ylim(0, 1)
            pylab.xlim(np.amin(n_range) - (N - 1), np.amax(n_range) + (N - 1))

            # pylab.ylim(0.0, 1.2)
            savefig(fig_dir + 'eta_comp_mb_N_%s_beta_%s_k_%s.eps' % (N, beta, k))
            # if args.show_pic:
            #     print('--> export result to %s' % (fig_fig_dir + 'eta_comp_mb_N_%s_beta_%s_k_%s.eps' % (N, beta, k)))
            #     plt.show()
