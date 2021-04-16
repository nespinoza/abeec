import numpy as np
from scipy.stats import loguniform

from astropy.timeseries import LombScargle
from stochastic.processes.noise import ColoredNoise

class example_prior:
    """
    This example class defines an example prior class to handle both evaluations and 
    sampling of the prior
    """

    def sample(self, nsamples = None):
        """
        Function that sample points from the prior
        """
        if nsamples is None:
            nsamples = self.nsamples

        # Evaluate samples:
        beta_samples = np.random.uniform(self.beta1, self.beta2, nsamples)
        sigma_samples = loguniform.rvs(self.sigma1, self.sigma2, size = nsamples)

        # Return them:
        return beta_samples, sigma_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        beta, sigma = theta

        # Validate the uniform prior:
        if beta <= self.beta1 or beta >= self.beta2:
            return False

        # Validate the loguniform prior:
        if sigma <= self.sigma1 or sigma >= self.sigma2:
            return False

        # If all is good, return a nice True:
        return True

    def evaluate(self, theta):
        """
        Given an input vector, evaluate the prior. In this case, this just returns the 
        priors defined by the hyperparameters. For the uniform prior, the value of the 
        prior doesn't depend on the inputs. For the loguniform, that's note the case.
        """

        # Extract current parameters to evaluate the priors on:
        beta, sigma = theta

        # Return the prior evaluated on theta:
        return self.beta_prior * (self.sigma_factor / sigma)


    def __init__(self, beta1 = 0.1, beta2 = 3., sigma1 = 0.1, sigma2 = 100., nsamples = 100):

        # Define hyperparameters of the prior. First for beta (uniform prior):
        self.beta1 = beta1
        self.beta2 = beta2

        # Value of the prior given hyperparameters:
        self.beta_prior = 1. / (beta2 - beta1)

        # Same for sigma (log-uniform):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # Compute factor that will multiply 1/x where x is the input to evaluate the 
        # prior:
        la = np.log(sigma1)
        lb = np.log(sigma2)
        self.sigma_factor = 1./(lb - la)  

        # Define the default number of samples:
        self.nsamples = nsamples

def gen_fake_data(length = 1000):
    beta, sigma = 1., 10.

    cn_simulator = ColoredNoise(beta=beta, t = 10)
    simulated_data = cn_simulator.sample(length - 1)
    simulated_data /= np.sqrt(np.var(simulated_data))
    simulated_data *= sigma
    fout = open('data.dat', 'w')
    for i in range(len(simulated_data)):
        fout.write(str(simulated_data[i])+'\n')
    fout.close()

class example_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        beta, sigma = parameters

        # Generate simulation:
        cn_simulator = ColoredNoise(beta=beta, t = self.tend)
        simulated_data = cn_simulator.sample(self.length-1)

        # Standarize it:
        simulated_data /= np.sqrt(np.var(simulated_data))

        # Return scaled time-series:
        return simulated_data * sigma

    def several_simulations(self, parameters):

        # Extract parameters:
        betas, sigmas = parameters
        nsamples = len(betas)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([betas[i], sigmas[i]])

        return simulations


    def __init__(self, tend = 10, length = 1000):
        self.tend = tend
        self.length = length

class example_distance:
    """
    Example class for distance.
    """
    def single_distance(self, simulation):
        """ Given a dataset and a simulation, this function returns the distance 
            between them. This is defined here as the median absolute sum between their 
            power-spectral densities, with a time-array defined outside of this function """
        simulation_power = LombScargle(self.times, simulation).power(self.frequencies)
        return np.sum(np.abs(self.data_power - simulation_power))

    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)
        for i in range(nsimulations):
            distances[i] = self.single_distance(simulations[i,:])
        return distances

    def __init__(self, data, length = 1000):
        self.times = np.linspace(0.01,10.,length)

        self.min_tscale, self.max_tscale = np.median(np.abs(np.diff(self.times))), \
                                           np.max(self.times) - np.min(self.times)

        self.frequencies = np.linspace(1. / self.max_tscale, \
                                       1. / (2. * self.min_tscale), length)

        self.data_power = LombScargle(self.times, data).power(self.frequencies)
