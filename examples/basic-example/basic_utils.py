import numpy as np
from scipy.stats import loguniform

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
        mu_samples = np.random.uniform(self.mu1, self.mu2, nsamples)
        sigma_samples = loguniform.rvs(self.sigma1, self.sigma2, size = nsamples)

        # Return them:
        return mu_samples, sigma_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        mu, sigma = theta

        # Validate the uniform prior:
        if mu <= self.mu1 or mu >= self.mu2:
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
        mu, sigma = theta

        # Return the prior evaluated on theta:
        return self.mu_prior * (self.sigma_factor / sigma)


    def __init__(self, mu1 = -100, mu2 = 100, sigma1 = 0.1, sigma2 = 100., nsamples = 100):

        # Define hyperparameters of the prior. First for mu (uniform prior):
        self.mu1 = mu1
        self.mu2 = mu2

        # Value of the prior given hyperparameters:
        self.mu_prior = 1. / (mu2 - mu1)

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
    mu, sigma = 50., 10.

    simulated_data = np.random.normal(mu, sigma, length)
    fout = open('basic_data.dat', 'w')
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
        mu, sigma = parameters

        return np.random.normal(mu, sigma, self.length)

    def several_simulations(self, parameters):

        # Extract parameters:
        mus, sigmas = parameters
        nsamples = len(mus)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([mus[i], sigmas[i]])

        return simulations


    def __init__(self, length = 1000):
        self.length = length

class example_distance:
    """
    Example class for distance.
    """
    def single_distance(self, simulation):
        """ Given a dataset and a simulation, this function returns the distance 
            between them. This is defined here as the sum of the absolute deviation between 
            the data and a given simulation """ 

        sim_mean = np.mean(simulation)
        sim_var = np.var(simulation)
        return np.abs( (sim_mean - self.data_mean) / self.data_mean) + \
               np.abs( (sim_var - self.data_variance) / self.data_variance )

    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)
        for i in range(nsimulations):
            distances[i] = self.single_distance(simulations[i,:])
        return distances

    def __init__(self, data, length = 1000):

        self.data = data

        self.data_mean = np.mean(data)
        self.data_variance = np.var(data)
