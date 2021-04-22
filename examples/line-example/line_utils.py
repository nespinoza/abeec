import numpy as np
from scipy.stats import loguniform

class example_prior:
    """
    This example class defines an example prior class to handle both evaluations and 
    sampling of the prior. This samples a slope (a), an intercept (b) and the standard-deviation (sigma) of 
    the data.
    """

    def sample(self, nsamples = None):
        """
        Function that sample points from the prior. Uniform for the slope and intercept --- log-uniform for the standard-deviation.
        """
        if nsamples is None:
            nsamples = self.nsamples

        # Evaluate samples:
        a_samples = np.random.uniform(self.a1, self.a2, nsamples)
        b_samples = np.random.uniform(self.b1, self.b2, nsamples)
        sigma_samples = loguniform.rvs(self.sigma1, self.sigma2, size = nsamples)

        # Return them:
        return a_samples, b_samples, sigma_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        a, b, sigma = theta

        # Validate the uniform priors:
        if a <= self.a1 or a >= self.a2:
            return False

        if b <= self.b1 or b >= self.b2:
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
        a, b, sigma = theta

        # Return the prior evaluated on theta:
        return self.a_prior * self.b_prior *  (self.sigma_factor / sigma)


    def __init__(self, a1 = -100, a2 = 100, b1 = -100, b2 = 100, sigma1 = 0.1, sigma2 = 100., nsamples = 100):

        # Define hyperparameters of the prior. First for slope (a, uniform prior):
        self.a1 = a1
        self.a2 = a2

        # Value of the prior given hyperparameters:
        self.a_prior = 1. / (a2 - a1)

        # Same for intercept:
        self.b1 = b1
        self.b2 = b2

        self.b_prior = 1. / (b2 - b1)

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
    """
    This function saves a file, `line_data.dat` with simulations from a line with slope `a`, intercept `b` and standard-deviation on the 
    data of `sigma` of size `length`.
    """

    # Define parameters:
    a, b, sigma = 10., 50., 20.

    # Simulate random noise:
    simulated_data = np.random.normal(0., sigma, length)

    # Add a line with slope "a" and intercept "b" to that error:
    x = np.linspace(-5, 5, length)
    simulated_data += a*x + b

    fout = open('line_data.dat', 'w')
    for i in range(len(simulated_data)):
        fout.write(str(simulated_data[i])+'\n')
    fout.close()

class example_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations. Simulates same data as the one in the `gen_fake_data()` function.
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        a, b, sigma = parameters

        return a * self.x + b + np.random.normal(0, sigma, self.length)

    def several_simulations(self, parameters):

        # Extract parameters:
        all_as, all_bs, all_sigmas = parameters
        nsamples = len(all_sigmas)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([all_as[i], all_bs[i], all_sigmas[i]])

        return simulations


    def __init__(self, length = 1000):
        self.length = length
        self.x = np.linspace(-5, 5, self.length)
        self.dataset_length = len(self.x)

class example_distance:
    """
    Example class for distance.
    """
    def single_distance(self, simulation):
        """ Given a dataset and a simulation, this function returns the distance 
            between them. This is defined here as the distance to the slope, intercept and 
            total variance of residuals """ 

        sim_a, sim_b = np.polyfit(self.xdata, simulation, 1)
        sim_residual_variance = np.var(simulation - (sim_a * self.xdata + sim_b))
        return np.abs( (sim_a - self.data_a) / self.data_a) + \
               np.abs( (sim_b - self.data_b) / self.data_b ) + \
               np.abs( (sim_residual_variance - self.data_residual_variance) / self.data_residual_variance )

    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)
        for i in range(nsimulations):
            distances[i] = self.single_distance(simulations[i,:])
        return distances

    def __init__(self, ydata, length = 1000):

        self.xdata = np.linspace(-5, 5, length)
        self.ydata = ydata

        self.data_a, self.data_b = np.polyfit(self.xdata, self.ydata, 1)
        self.data_residual_variance = np.var(self.ydata - (self.data_a * self.xdata + self.data_b))
