import matplotlib.pyplot as plt
import numpy as np

from astropy.timeseries import LombScargle
import corner

import abeec 
from tso_utils import tso_prior, tso_distance, tso_simulator

# Read simulated dataset (you can generate a new test dataset 
# by doing import basic_utils; basic_utils.gen_fake_data()):
data = np.loadtxt('tso_data.dat', unpack = True)

# Perform ABC sampling:
samples = abeec.sampler.sample(tso_prior(), tso_distance(data), tso_simulator(), \
                               M = 10000, N = 300, Delta = 0.01,\
                               verbose = True)

# Extract the 300 posterior samples from the latest particle:
tend = list(samples.keys())[-1]
betas, sigma_poissons, sigma_flickers = samples[tend]['thetas']

# Plot corner plot:
stacked_samples = np.vstack((np.vstack((betas, sigma_poissons)), sigma_flickers)).T
figure = corner.corner(stacked_samples, labels = [r"$\beta$ (from 1/f$^{\beta}$)", r"$\sigma_{poisson}$", r"$\sigma_{flicker}$"],\
                       quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

# Plot true values:
true_values = [1., 2., 10.] # beta, sigma_poisson, sigma_flicker
axes = np.array(figure.axes).reshape((3, 3))

# Loop over histograms:
for i in range(3):
    ax = axes[i,i]
    ax.axvline(true_values[i], color = 'cornflowerblue')

# Now loop over 2D surfaces:
for yi in range(3):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(true_values[xi], color = 'cornflowerblue')
        ax.axhline(true_values[yi], color = 'cornflowerblue')
        ax.plot(true_values[xi], true_values[yi], 'o', mfc = 'cornflowerblue', mec = 'cornflowerblue')

# Show corner plot:
plt.show()

# Now plot the PSD. First, get the tso_distance(data) object as well as the tso_simulator() object, which 
# have all the info we need:
distance = tso_distance(data)
simulator = tso_simulator()

# Plot the PSD of the data:
plt.plot(distance.frequencies, distance.data_power, color = 'orangered', zorder = 2)

# Now get simulations for all the ABC samples:
simulations = simulator.several_simulations(samples[tend]['thetas'])
real_simulations = simulator.several_simulations([np.ones(simulations.shape[0])*true_values[0], \
                                                  np.ones(simulations.shape[0])*true_values[1], \
                                                  np.ones(simulations.shape[0])*true_values[2]])

# Get the PSDs of those simulated datasets:
real_psds = np.zeros([simulations.shape[0], len(distance.frequencies)])
psds = np.zeros([simulations.shape[0], len(distance.frequencies)])
for i in range(simulations.shape[0]):

    psds[i,:] =  LombScargle(distance.times, simulations[i,:], normalization = 'psd').power(distance.frequencies)
    real_psds[i,:] =  LombScargle(distance.times, real_simulations[i,:], normalization = 'psd').power(distance.frequencies)
    # plt.plot(distance.frequencies, psds[i,:], color = 'cornflowerblue', alpha = 0.1, zorder = 1)
    # plt.plot(distance.frequencies, real_psds[i,:], color = 'black', alpha = 0.1, zorder = 2)

# Plot quantiles for both real and the sampled PSDs:
for alpha in [0.68, 0.95]:
    m, u, l = abeec.function_quantiles(psds, alpha = alpha)
    real_m, real_u, real_l = abeec.function_quantiles(real_psds, alpha = alpha)

    plt.fill_between(distance.frequencies, l, u, alpha = 0.1, zorder = 2, color = 'black')
    plt.fill_between(distance.frequencies, real_l, real_u, alpha = 0.3, zorder = 2, color = 'cornflowerblue')

# Plot average PSD:
plt.plot(distance.frequencies, m, color = 'black', alpha = 0.5, zorder = 3, label = 'Posterior PSD')
plt.plot(distance.frequencies, real_m, color = 'cornflowerblue', alpha = 0.5, zorder = 3, label = 'PSDs sampled from real, underlying model')

plt.legend()
# Set yscale to log:
#plt.yscale('log')

# Show plot:
plt.xlabel('Frequency (1/Hz)')
plt.ylabel('PSD')
plt.yscale('log')
plt.show()

# Plot autocorrelation:
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

plt.plot(distance.times, autocorr(data), color = 'orangered', zorder = 2, label = 'Autocorrelation of the input data')
autocorrs = np.zeros([simulations.shape[0], len(data)])
real_autocorrs = np.zeros([simulations.shape[0], len(data)])
for i in range(simulations.shape[0]):

    autocorrs[i,:] = autocorr(simulations[i,:])
    real_autocorrs[i,:] = autocorr(real_simulations[i,:])
    plt.plot(distance.times, autocorrs[i,:], color = 'cornflowerblue', alpha = 0.1, zorder = 1)
    plt.plot(distance.times, real_autocorrs[i,:], color = 'orangered', alpha = 0.1, zorder = 1)

# Plot average autocorrelation:
plt.plot(distance.times, np.median(autocorrs, axis = 0), color = 'blue', lw=3, zorder = 3, label = 'Median posterior autocorrelation')
plt.plot(distance.times, np.median(real_autocorrs, axis = 0), color = 'orangered', lw=3, zorder = 3, label = 'Median from real model')
plt.legend()
plt.show()

# Plot differences:
for i in range(simulations.shape[0]):
    if i == 0:
        plt.plot(distance.times, autocorr(data) - autocorrs[i,:], color = 'cornflowerblue', alpha = 0.1, zorder = 1, label = 'autocorr input - autocorr posteriors')
        plt.plot(distance.times, autocorr(data) - real_autocorrs[i,:], color = 'orangered', lw = 4, alpha = 0.1, zorder = 1, label = 'autocorr input data - autocorr real model')
    else:
        plt.plot(distance.times, autocorr(data) - autocorrs[i,:], color = 'cornflowerblue', alpha = 0.1, zorder = 1)
        plt.plot(distance.times, autocorr(data) - real_autocorrs[i,:], color = 'orangered', lw = 4, alpha = 0.1, zorder = 1)
plt.legend()
plt.show()
