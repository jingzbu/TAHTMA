from pylab import *
import numpy as np
import statsmodels.api as sm  # recommended import according to the docs
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace,hstack

mu, sigma = 1, 2
sample = np.random.normal(mu, sigma, 10000)
# sample = np.random.lognormal(mu, sigma, 10000)

# sample.plot(kind="density")

ecdf = sm.distributions.ECDF(sample)

x = np.linspace(min(sample), max(sample))
y = ecdf(x)
plt.plot(x, y)
plt.show()


# samp = hstack([sample])
# # obtaining the pdf (my_pdf is a function!)
# my_pdf = gaussian_kde(samp)
#
# # plotting the result
# x = linspace(-10, 10, 1000)
# plt.plot(x,my_pdf(x),'r')  # distribution function
# # hist(samp,normed=1,alpha=.3)  # histogram
# plt.show()