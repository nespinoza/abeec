import numpy as np
import time
from scipy.stats import loguniform

import dask
import ray
from dask.distributed import Client
from astropy.timeseries import LombScargle
from stochastic.processes.noise import ColoredNoise

import data_utils

def generate_detector_ts(beta, sigma_w, sigma_flicker, columns = 2048, rows = 512, pixel_time = 10, jump_time = 120, return_image = False, return_time = False):
    """
    This function simulates a JWST detector image and corresponding time-series of the pixel-reads, assuming the noise follows a $1/f^\beta$ power-law in its 
    power spectrum. This assumes the 1/f pattern (and hence the detector reads) go along the columns of the detector.
    Parameters
    ----------
    beta : float
        Power-law index of the PSD of the noise. 
    sigma_w : boolean
        Square-root of the variance of the added Normal-distributed noise process.
    sigma_flicker : float
        Variance of the power-law process in the time-domain. 
   columns : int
        Number of columns of the detector.
    rows : int
        Number of rows of the detector.
    pixel_time : float
        Time it takes to read a pixel along each column in microseconds. Default is 10 microseconds (i.e., like JWST NIR detectors).
    jump_time : float
        Time it takes to jump from one column to the next once all its pixels have been read, in microseconds. Default is 120 microseconds (i.e., like JWST NIR detectors).
    return_image : boolean
        If True, returns an image with the simulated values. Default is False.
    return_time : boolean 
        If True, returns times as well. Default is False.

    Returns
    -------
    times : `numpy.array`
        The time-stamp of the flux values (i.e., at what time since read-out started were they read).
    time_series : `numpy.array`
        The actual flux values on each time-stamp (i.e., the pixel counts as they were read in time).
    image : `numpy.array` 
        The image corresponding to the `times` and `time_series`, if `return_image` is set to True.
    """
    # This is the number of "fake pixels" not read during the waiting time between jumps:
    nfake = int(jump_time/pixel_time)

    # First, generate a time series assuming uniform sampling (we will chop it later to accomodate the jump_time):
    CN = ColoredNoise(beta = beta, t = (rows * columns * pixel_time) + columns * jump_time)

    # Get the samples and time-indexes:
    nsamples = rows * columns + (nfake * columns)
    y = CN.sample(nsamples)
    t = CN.times(nsamples)

    # Now remove samples not actually read by the detector due to the wait times. Commented 
    # loop below took 10 secs (!). New pythonic way is the same thing, takes millisecs, and 
    # gets image for free:

    if return_time:
        t_image = t[:-1].reshape((columns, rows + nfake))
        time_image = t_image[:, :rows]
        times = time_image.flatten()

    y_image = y[:-1].reshape((columns, rows + nfake))
    image = y_image[:, :rows]
    time_series = image.flatten()

    # Set process standard-deviation to input sigma:
    time_series = sigma_flicker * (time_series / np.sqrt(np.var(time_series)) )

    # Add poisson noise:
    time_series = time_series + np.random.normal(0., sigma_w, len(time_series))

    if not return_image:
        if not return_time:
            return time_series
        else:
            return times, time_series

    else:
        if return_time:
            # Return all:
            return times, time_series, image.transpose()
        else:
            return time_series, image.transpose()

class tso_prior:
    """
    This example class defines an example prior class to handle both evaluations and 
    sampling of the prior. This samples the frequency slope (beta), the sigma of the 
    white-noise process (a gaussian, with standard deviation sigma_w) and the sigma 
    of the flicker process (sigma_flicker).
    """

    def sample(self, nsamples = None):
        """
        Function that sample points from the prior. Uniform for beta (with limits beta1 and beta2); 
        log uniform for the sigmas (with limits sigma1 and sigma2).
        """
        if nsamples is None:
            nsamples = self.nsamples

        # Evaluate samples:
        beta_samples = np.random.uniform(self.beta1, self.beta2, nsamples)
        sigma_w_samples = loguniform.rvs(self.sigma_w1, self.sigma_w2, size = nsamples)
        sigma_flicker_samples = loguniform.rvs(self.sigma_flicker1, self.sigma_flicker2, size = nsamples)

        # Return them:
        return beta_samples, sigma_w_samples, sigma_flicker_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        beta, sigma_w, sigma_flicker = theta

        # Validate the uniform priors:
        if beta <= self.beta1 or beta >= self.beta2:
            return False

        if sigma_w <= self.sigma_w1 or sigma_w >= self.sigma_w2:
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
        prior doesn't depend on the inputs. For the loguniform, that's not the case.
        """

        # Extract current parameters to evaluate the priors on:
        beta, sigma_w, sigma_flicker = theta

        # Return the prior evaluated on theta:
        return self.beta_prior * (self.sigma_w_factor / sigma_w) *  (self.sigma_flicker_factor / sigma_flicker)


    def __init__(self, beta1 = 0.1, beta2 = 3, sigma_w1 = 1., sigma_w2 = 20., sigma_flicker1 = 1., sigma_flicker2 = 20., nsamples = 100):

        # Define hyperparameters of the prior. First for slope of 1/f process (beta, uniform prior):
        self.beta1 = beta1
        self.beta2 = beta2

        # Value of the prior given hyperparameters:
        self.beta_prior = 1. / (beta2 - beta1)

        # Same for sigma_poisson (log-uniform):
        self.sigma_w1 = sigma_w1
        self.sigma_w2 = sigma_w2

        # Compute factor that will multiply 1/x where x is the input to evaluate the 
        # prior:
        la = np.log(sigma_w1)
        lb = np.log(sigma_w2)
        self.sigma_w_factor = 1./(lb - la)  

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

@dask.delayed
def sim_psd(data, times, frequencies):

    # Get LS periodogram in seconds (times in microseconds):
    psd = LombScargle(times * 1e-6, data, normalization = 'psd').power(frequencies)

    return psd

def get_sim_psds(beta, sigma_w, sigma_flicker, columns, rows, pixel_time, jump_time, times, frequencies, ngroups):

    all_data = data_utils.get_sim_data(beta, sigma_w, sigma_flicker, columns, rows, pixel_time, jump_time, ngroups)

    all_results = []

    #tpsds = time.time()
    for i in range(ngroups):

        all_results.append(sim_psd(all_data[i], times, frequencies))

    results = dask.delayed(all_results).compute(num_workers = 30)#, scheduler='processes')

    #print('Generating PSDs took', time.time() - tpsds, 'seconds.')
    return np.array(results)

class tso_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations. Simulates same data as the one in the `gen_fake_data()` function.
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        beta, sigma_w, sigma_flicker = parameters

        # Simulate single simulation that will be compared against the data. Note we simulate 
        # data for several groups and then compute the PSD. The MEDIAN PSD is the observable, 
        # not the time-series itself:

        #print('Generating 88 sims...')
        #t0 = time.time()

        all_psds = get_sim_psds(beta, sigma_w, sigma_flicker, self.ncolumns, self.nrows, self.pixel_time, \
                                self.jump_time, self.times, self.frequencies, self.ngroups)

        #print('Total: took ',time.time() - t0, 'secs.')

        #print(all_psds.shape)
        if self.idx is None:
            return np.median(all_psds, axis=0)
        else:
            return np.median(all_psds, axis=0)[self.idx]

    def several_simulations(self, parameters):

        # Extract parameters:
        betas, sigma_ws, sigma_flickers = parameters
        nsamples = len(betas)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.nfrequencies])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([betas[i], sigma_ws[i], sigma_flickers[i]])

        return simulations


    def __init__(self, ncolumns = 2048, nrows = 512, ngroups = 88, frequency_filename = '', idx = None):

        # Load frequencies (needed to simulate the PSDs):
        self.frequencies = np.load(frequency_filename)
        self.idx = None

        if idx is not None:
            
            self.idx = idx
            self.nfrequencies = len(idx)

        else:

            self.nfrequencies = len(self.frequencies)

        # Define properties of the real and to-be-simulated datasets 
        # (size of the arrays, number of groups used to get the data PSD):
        self.ncolumns = ncolumns
        self.nrows = nrows
        self.ngroups = ngroups
        self.pixel_time = 10 # pixel-to-pixel time in microseconds
        self.jump_time = 120 # column-to-column wait time in microseconds

        # With those properties, generate a mock dataset so as to print some properties out:
        self.times, _ = generate_detector_ts(1., 1., 1., columns = self.ncolumns, \
                                             rows = self.nrows, pixel_time = self.pixel_time, \
                                             jump_time = self.jump_time, return_time = True)

        print('Simulated data properties:')
        print('--------------------------\n')
        print('Columns, rows, ngroups:',self.ncolumns, self.nrows, self.ngroups)
        print('Minimum time: ',self.times[0],' us.','Maximum:',self.times[-1],'us')
        print('Simulated dataset(s) (time-series) length:', len(self.times))
        print('Number of frequencies:', self.nfrequencies)

class tso_distance:
    """
    Example class for distance of the TSO example.

    """

    def single_distance(self, simulation):

        total_sum = 0.

        for i in range(self.nindexes):

            value1 = np.median(self.data_power[self.indexes[i]])
            value2 = np.median(simulation[self.indexes[i]])

            total_sum += np.abs(np.log10(value1) - np.log10(value2))
 
        return total_sum

    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)

        for i in range(nsimulations):
            distances[i] = self.single_distance(simulations[i,:])

        return distances

    def __init__(self, filename = '', filename_indexes = '', idx = None):

        # Define/load input PSD:
        self.data_power = np.load(filename)
        self.indexes = np.load(filename_indexes, allow_pickle=True)
        self.nindexes = len(self.indexes)

        if idx is not None:

            self.data_power = self.data_power[idx]

if __name__ == '__main__':
    client = dask.distributed.Client()
