import matplotlib.pyplot as plt
import numpy as np

import abeec 
from basic_utils import example_prior, example_distance, example_simulator

# Read simulated dataset (you can generate a new test dataset 
# by doing import basic_utils; basic_utils.gen_fake_data()):
data = np.loadtxt('basic_data.dat', unpack = True)

# Perform ABC sampling:
samples = abeec.sampler.sample(example_prior(), example_distance(data), example_simulator(), \
                               M = 10000, N = 200, Delta = 0.1,\
                               verbose = True)

# Extract the 100 posterior samples from the latest particle:
tend = list(samples.keys())[-1]
mus, sigmas = samples[tend]['thetas']

# Plot:
plt.plot(mus, sigmas,'ko', label = 'ABC samples')
plt.plot([50, 50], [-100, 100], '--', label = 'True value', color = 'cornflowerblue')
plt.plot([-100, 100], [10., 10.], '--', color = 'cornflowerblue')
plt.xlabel('Mean', fontsize = 10)
plt.ylabel('$\sigma$', fontsize = 14)
plt.legend(fontsize = 12)
plt.xlim(49,51)
plt.ylim(9,11)
plt.show()
