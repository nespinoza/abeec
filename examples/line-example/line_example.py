import matplotlib.pyplot as plt
import numpy as np

import corner
import abeec 
from line_utils import example_prior, example_distance, example_simulator

# Read simulated dataset (you can generate a new test dataset 
# by doing import line_utils; line_utils.gen_fake_data()):
data = np.loadtxt('line_data.dat', unpack = True)
xdata = np.linspace(-5, 5, len(data))

# Perform ABC sampling:
samples = abeec.sampler.sample(example_prior(), example_distance(data), example_simulator(), \
                               M = 3000, N = 300, Delta = 0.1,\
                               verbose = True)

# Extract the 300 posterior samples from the latest particle:
tend = list(samples.keys())[-1]
all_as, all_bs, sigmas = samples[tend]['thetas']

# Plot corner plot:
cdata = np.vstack((np.vstack((all_as, all_bs)), sigmas)).T
figure = corner.corner(cdata, labels = [r"$a$ (slope)", r"$b$ (intercept)", r"$\sigma$"],\
                       quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})

# Plot true values:
true_values = [10., 50., 20.] # a, b, sigma
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

plt.show()

# Plot data:
plt.plot(xdata, data, 'o', mfc = 'white', mec = 'cornflowerblue')
for i in range( len(all_as) ) :

    a, b = all_as[i], all_bs[i]

    plt.plot(xdata, xdata*a + b, color = 'black', alpha = 0.01)

plt.xlim(-5,5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('X axis', fontsize = 18)
plt.ylabel('Y axis', fontsize = 18)
plt.show()
