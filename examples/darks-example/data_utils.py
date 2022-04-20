import numpy as np
import time
from scipy.stats import loguniform

import ray
from stochastic.processes.noise import ColoredNoise

ray.init(num_cpus=30)

@ray.remote
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

def get_sim_data(beta, sigma_w, sigma_flicker, columns, rows, pixel_time, jump_time, ngroups):

    all_data = []

    #tdata = time.time()
    for i in range(ngroups):

        all_data.append(generate_detector_ts.remote(beta, sigma_w, sigma_flicker, columns, rows, pixel_time, jump_time))

    results = ray.get(all_data)

    #print('Generating data took', time.time() - tdata, 'seconds.')

    return results
