import numpy as np

import sampler
from utils import example_prior, example_distance, example_simulator

# Read simulated dataset:
data = np.loadtxt('data.dat', unpack = True)

# Perform ABC sampling:
samples = sampler.sample(example_prior(), example_distance(data), example_simulator())
