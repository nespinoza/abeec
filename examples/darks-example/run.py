import matplotlib.pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from astropy.timeseries import LombScargle
import corner

import abeec 
from psd_utils import tso_prior, tso_distance, tso_simulator

# Define which detector/subarray is going to be fit:
detector = 'SUBSTRIP96'

if detector == 'SUBSTRIP256':

    frequency_filename = 'data/farray256_otis.npy'
    psd_filename = 'data/median_power256_otis.npy'
    ncolumns = 2048
    nrows = 256
    ngroups = 199

elif detector == 'SUBSTRIP96':

    frequency_filename = 'data/farray96_otis.npy'
    psd_filename = 'data/median_power96_otis.npy'
    ncolumns = 2048
    nrows = 96 
    ngroups = 239  
    
elif detector == 'NRS1':

    frequency_filename = 'data/frequencies.npy'
    psd_filename = 'data/median_NRS1_88.npy'
    indexes_filename = 'data/indexes.npy'
    ncolumns = 2048
    nrows = 512
    ngroups = 88

elif detector == 'NRS2':

    frequency_filename = 'data/frequencies.npy'
    psd_filename = 'data/median_NRS2_88.npy'
    indexes_filename = 'data/indexes.npy'
    ncolumns = 2048
    nrows = 512 
    ngroups = 88

# Perform ABC sampling:
prior = tso_prior()

distance = tso_distance(filename = psd_filename, filename_indexes = indexes_filename)

simulator = tso_simulator(ncolumns = ncolumns, nrows = nrows, ngroups = ngroups, frequency_filename = frequency_filename)

samples = abeec.sampler.sample(prior, distance, simulator, \
                               M = 150, N = 100, Delta = 0.1,\
                               verbose = True, output_file = 'results_'+detector+'.pkl')

# Extract the 300 posterior samples from the latest particle:
tend = list(samples.keys())[-1]
betas, sigma_poissons, sigma_flickers = samples[tend]['thetas']

# Plot corner plot:
stacked_samples = np.vstack((np.vstack((betas, sigma_poissons)), sigma_flickers)).T
figure = corner.corner(stacked_samples, labels = [r"$\beta$ (from 1/f$^{\beta}$)", r"$\sigma_{w}$", r"$\sigma_{flicker}$"],\
                       quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

plt.savefig('corner_'+detector+'.pdf')
