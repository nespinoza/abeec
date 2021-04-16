import numpy as np

import abeec 
from utils1 import example_prior, example_distance, example_simulator

# Read simulated dataset:
data = np.loadtxt('data1.dat', unpack = True)

# Perform ABC sampling:
samples = abeec.sampler.sample(example_prior(), example_distance(data), example_simulator())
