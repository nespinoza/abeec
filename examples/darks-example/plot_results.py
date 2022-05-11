import os
import pickle

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np

import juliet

import psd_utils
from user_options import *

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='cornflowerblue', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

def plot_ellipse(x, y, ax):

    confidence_ellipse(x, y, ax, n_std=1,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.8)
    confidence_ellipse(x, y, ax, n_std=2,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.5)
    confidence_ellipse(x, y, ax, n_std=3,
                       edgecolor='cornflowerblue', facecolor = 'cornflowerblue', alpha = 0.3)

"""
First part is to actually generate simulations given our best-fit parameters for 
1/f + white-noise. So, we generate these instances:
"""

# First, extract posterior distributions:
posteriors = pickle.load(open('results_'+detector+'.pkl', 'rb'))

last_int = list(posteriors.keys())[-1]
beta, sigma_w, sigma_f = posteriors[last_int]['thetas']

# Extract frequencies and the actual data PSD:
frequencies = np.load(frequency_filename)
data_psd = np.load(psd_filename)

# Compute simulated PSDs. First compute times, then psds:
times, _ = psd_utils.generate_detector_ts(1., 1., 1., columns = ncolumns, \
                                         rows = nrows, pixel_time = pixel_time, \
                                         jump_time = jump_time, return_time = True)

print('Generating sample PSD...')

median_beta, median_sigma_f, median_sigma_w = np.median(beta), np.median(sigma_f), np.median(sigma_w)

if not os.path.exists('sims_'+detector+'.npy'):

    nsims = 100
    sims = np.zeros([len(frequencies), nsims])

    for i in range(nsims):

        all_psds = psd_utils.get_sim_psds(beta[i], sigma_w[i], sigma_f[i], \
                                          ncolumns, nrows, pixel_time, \
                                          jump_time, times, frequencies, ngroups)
        print(i,'of',nsims)
        sims[:, i] = np.median(all_psds, axis=0)

    np.save('sims_'+detector, sims)

else:

    sims = np.load('sims_'+detector+'.npy')

"""
Now that this is done, we move forward and perform the actual plotting. First, plot the posterior distribution:
"""

grid = GridSpec(2, 5)
fig = plt.figure(figsize=(10,4))

ax_psd = fig.add_subplot(grid[:,:3])
ax1 = fig.add_subplot(grid[0,3])
ax2 = fig.add_subplot(grid[1,3])
ax3 = fig.add_subplot(grid[1,4])
ax4 = fig.add_subplot(grid[0,4])

# ax1 first:
ax1.plot(beta, sigma_w, '.', alpha = 0.7)
plot_ellipse(beta, sigma_w, ax1)
ax1.set_ylabel(r'$\sigma_w$ (counts)')

# ax2:
ax2.plot(beta, sigma_f, '.', alpha = 0.7)
plot_ellipse(beta, sigma_f, ax2)
ax2.set_xlabel(r'Slope, $\beta$')
ax2.set_ylabel(r'$\sigma_f$ (counts)')

# ax3:
ax3.plot(sigma_w, sigma_f, '.', alpha = 0.7)
plot_ellipse(sigma_w, sigma_f, ax3)
ax3.set_xlabel(r'$\sigma_w$ (counts)')

# ax4:
ax4.hist(sigma_f/sigma_w, bins = 8, color = 'cornflowerblue')
ax4.set_xlabel('$\sigma_f/\sigma_w$')

"""
Now plot the PSD:
"""

ax_psd.plot(frequencies, data_psd, color = 'black', zorder = 1)
ax_psd.set_xscale('log')
ax_psd.set_yscale('log')
ax_psd.set_xlabel('Frequency (Hz)')
ax_psd.set_ylabel('Power')
ax_psd.set_xlim([np.min(frequencies), np.max(frequencies)])

# Add text with parameters:
x_text, y_text = 2e2, 3e6
ax_psd.text(x_text*0.85, y_text/1.5, plot_name+':', fontsize=13)
ax_psd.text(x_text, y_text/6, r'$\beta = {0:.2f} \pm {1:.2f}$'.format(np.median(beta), np.sqrt(np.var(beta))), fontsize=11)
ax_psd.text(x_text, y_text/18, r'$\sigma_f = {0:.2f} \pm {1:.2f}$'.format(np.median(sigma_f), np.sqrt(np.var(sigma_f))), fontsize=11)
ax_psd.text(x_text, y_text/54, r'$\sigma_w = {0:.2f} \pm {1:.2f}$'.format(np.median(sigma_w), np.sqrt(np.var(sigma_w))), fontsize=11)

print('beta, sigma_f, sigma_w:')
print(np.median(beta), np.median(sigma_f), np.median(sigma_w))

model, model_up, model_down = np.zeros(len(frequencies)), np.zeros(len(frequencies)), np.zeros(len(frequencies))

try:

    for i in range(len(frequencies)):

        model[i], model_up[i], model_down[i] = juliet.utils.get_quantiles(sims[i, :])

    ax_psd.fill_between(frequencies, model_down, model_up, zorder = 3, color = 'cornflowerblue', alpha = 0.3)

except:

    for i in range(sims.shape[1]):

        ax_psd.plot(frequencies, sims[:, i], color = 'cornflowerblue', alpha = 0.05, zorder = 3)

    for i in range(len(frequencies)):

        model[i] = np.median(sims[i, :])

    ax_psd.plot(frequencies, model, zorder = 3, color = 'cornflowerblue')

print('...plotting done! Saving...')

ax_psd.plot(frequencies, model, color = 'cornflowerblue', zorder = 2, alpha = 0.9)
plt.tight_layout()

plt.savefig('plot_'+detector+'.pdf')
