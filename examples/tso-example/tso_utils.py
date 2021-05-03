import numpy as np
from scipy.stats import loguniform

from astropy.timeseries import LombScargle
from stochastic.processes.noise import ColoredNoise

def generate_flicker(beta, sigma_flicker, length = 1000, tend = 10):

    # Simulate 1/f process first:
    cn_simulator = ColoredNoise(beta=beta, t = tend)

    simulated_data = cn_simulator.sample(length - 1)
    sigma_simulated_data = np.sqrt(np.var(simulated_data))

    # Standarize simulaton; amplify by sigma_flicker:
    return (simulated_data / sigma_simulated_data) * sigma_flicker

class tso_prior:
    """
    This example class defines an example prior class to handle both evaluations and 
    sampling of the prior. This samples the frequency slope (beta), the sigma of the poisson process (sigma_poisson) and 
    the sigma of the flicker process (sigma_flicker).
    """

    def sample(self, nsamples = None):
        """
        Function that sample points from the prior. Uniform for the slope and intercept --- log-uniform for the standard-deviation.
        """
        if nsamples is None:
            nsamples = self.nsamples

        # Evaluate samples:
        beta_samples = np.random.uniform(self.beta1, self.beta2, nsamples)
        sigma_poisson_samples = loguniform.rvs(self.sigma_poisson1, self.sigma_poisson2, size = nsamples)
        sigma_flicker_samples = loguniform.rvs(self.sigma_flicker1, self.sigma_flicker2, size = nsamples)

        # Return them:
        return beta_samples, sigma_poisson_samples, sigma_flicker_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        beta, sigma_poisson, sigma_flicker = theta

        # Validate the uniform priors:
        if beta <= self.beta1 or beta >= self.beta2:
            return False

        if sigma_poisson <= self.sigma_poisson1 or sigma_poisson >= self.sigma_poisson2:
            return False

        # Validate the loguniform prior:
        if sigma_flicker <= self.sigma_flicker1 or sigma_flicker >= self.sigma_flicker2:
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
        beta, sigma_poisson, sigma_flicker = theta

        # Return the prior evaluated on theta:
        return self.beta_prior * (self.sigma_poisson_factor / sigma_poisson) *  (self.sigma_flicker_factor / sigma_flicker)


    def __init__(self, beta1 = 0.0, beta2 = 3, sigma_poisson1 = 1., sigma_poisson2 = 30., sigma_flicker1 = 1., sigma_flicker2 = 30., nsamples = 100):

        # Define hyperparameters of the prior. First for slope of 1/f process (beta, uniform prior):
        self.beta1 = beta1
        self.beta2 = beta2

        # Value of the prior given hyperparameters:
        self.beta_prior = 1. / (beta2 - beta1)

        # Same for sigma_poisson (log-uniform):
        self.sigma_poisson1 = sigma_poisson1
        self.sigma_poisson2 = sigma_poisson2

        # Compute factor that will multiply 1/x where x is the input to evaluate the 
        # prior:
        la = np.log(sigma_poisson1)
        lb = np.log(sigma_poisson2)
        self.sigma_poisson_factor = 1./(lb - la)  

        # Repeat for sigma_flicker:
        self.sigma_flicker1 = sigma_flicker1
        self.sigma_flicker2 = sigma_flicker2

        # Compute factor that will multiply 1/x where x is the input to evaluate the 
        # prior:
        la = np.log(sigma_flicker1)
        lb = np.log(sigma_flicker2)
        self.sigma_flicker_factor = 1./(lb - la) 

        # Define the default number of samples:
        self.nsamples = nsamples

def generate_simulation(beta, sigma_poisson, sigma_flicker, length = 1000, tend = 10):
    # Simulate 1/f process first:
    flicker = generate_flicker(beta, sigma_flicker, length = length, tend = tend)

    # Add the poisson process:
    poisson = np.random.poisson(sigma_poisson**2, length)

    return flicker + poisson

def gen_fake_data(length = 1000, tend = 10):
    """
    This function saves a file, `tso_data.dat`, with simulations from a 1/f ("flicker"-noise) process with added Poisson noise:
    """
    # Note the *square* of the sigma_poisson parameter equals the rate, lambda, of the poission process:
    beta, sigma_poisson, sigma_flicker = 1., 2., 10. 

    # Get data:
    simulated_data = generate_simulation(beta, sigma_poisson, sigma_flicker, length = length, tend = tend)

    # Save the data:
    fout = open('tso_data.dat', 'w')
    for i in range(len(simulated_data)):
        fout.write(str(simulated_data[i])+'\n')
    fout.close()

class tso_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations. Simulates same data as the one in the `gen_fake_data()` function.
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        beta, sigma_poisson, sigma_flicker = parameters

        # Simulate. First 1/f process:
        simulated_data = generate_flicker(beta, sigma_flicker, length = self.length, tend = self.tend)

        # Add both and return:
        simulated_data += np.random.poisson(sigma_poisson**2, self.length)

        return simulated_data

    def several_simulations(self, parameters):

        # Extract parameters:
        betas, sigma_poissons, sigma_flickers = parameters
        nsamples = len(betas)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([betas[i], sigma_poissons[i], sigma_flickers[i]])

        return simulations


    def __init__(self, tend = 10, length = 1000):
        self.tend = tend
        self.length = length

class tso_distance:
    """
    Example class for distance of the TSO example.

    Parameters
    ----------

    ydata : numpy.array
        Array defining the input data.

    tend : float
        End time of the time-array, which goes from 0 to `tend`.

    length : int
        Length of the dataset.

    mode : string
        This defines the mode with which the distance will be calculated. `autocorr` uses the autocorrelation function to measure the distance. `psd` uses 
        the power-spectral density.
    """
    def get_mad(self, data, data_median):
        """This functions gets the median absolute deviation of data given the dataset and its median"""
        return np.median(np.abs(data - data_median))

    def get_autocorrelation(self, data):
        """This function gets the autocorrelation of a dataset"""
        result = np.correlate(data, data, mode='full')
        return result[result.size // 2:]

    def get_LS(self, data):
        """ This function gets the PSD of a dataset. Note the important fact that we define normalization = 'psd', so the periodogram is 
            not normalized (which we want to identify the different sigma-components):
        """
        return LombScargle(self.times, data, normalization = 'psd').power(self.frequencies)

    def autocorr_distance(self, simulation):
        
         # Distance is the absolute value of the median of the difference between the autocorrelation functions:
         differences = self.data_autocorrelation - self.get_autocorrelation(simulation)
         return np.sum(differences**2)#np.abs(med_differences) + np.abs(mad_differences)

    def psd_distance(self, simulation):

        # Distance is the absolute value of the median of the difference between the PSDs:
        differences = self.data_power - self.get_LS(simulation)
        #med = np.median(differences)
        #med_differences = med / self.data_med_power
        #mad_differences = self.get_mad(differences, med) / self.data_mad_power
        return np.sum(differences**2)#np.abs(med_differences) + np.abs(mad_differences)

    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)
        for i in range(nsimulations):
            distances[i] = self.single_distance(simulations[i,:])
        return distances

    def __init__(self, ydata, tend = 10, length = 1000, nbins = 30, mode = 'autocorrelation'):

        # Define times and time-scales:
        self.times = ColoredNoise(beta=1., t = tend).times(length - 1)
        self.min_tscale, self.max_tscale = np.median(np.abs(np.diff(self.times))), \
                                           np.max(self.times) - np.min(self.times)

        # Define frequency-space for PSD:
        self.frequencies = np.linspace(1. / self.max_tscale, \
                1. / (2. * self.min_tscale), length)[:-1]

        # Compute data power-spectral density:
        self.data_power = self.get_LS(ydata)

        # Compute data autocorrelation:
        self.data_autocorrelation = self.get_autocorrelation(ydata)

        if mode.lower() == 'psd':
            # Define the signle-distance function:
            self.single_distance = self.psd_distance
            self.data_med_power = np.median(self.data_power)
            self.data_mad_power = self.get_mad(self.data_power, self.data_med_power)

        elif mode.lower() == 'autocorrelation':
            # Define single-distance function:
            self.single_distance = self.autocorr_distance
            self.data_med_autocorrelation = np.median(self.data_autocorrelation)
            self.data_mad_autocorrelation = self.get_mad(self.data_autocorrelation, self.data_med_autocorrelation)

        else:
            raise Exception("Mode "+mode+" not supported")

        # Define mode:
        self.distance_mode = mode.lower()
