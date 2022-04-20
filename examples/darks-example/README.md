# Fitting real power spectral densities with `abeec`
-----------------------------

In this example, we perform a real-world fitting of JWST detector data. Here, the detector shows a clear signature of `1/f**beta` noise plus white-noise which we model as *gaussian* noise. 

## Distance metric
------------------
As a distance metric, we use the power spectral density (PSD). In practice, we (a) bin the PSD at certain frequencies (spaced in log) in order to speed-up convergence, (b) perform the distance in log-space in order to account for the fact that the PSD varies by several orders of magnitude between the low and high-frequency end and (c) compute the distance in absolute value; this guards against outliers.

## Data simulation
------------------
The data is generated via a second script (`data_utils.py`), which couples our detector simulator with `ray`, a library that efficiently parallelizes the calculations to generate many of those simulations; this by default uses 30 cores to significantly speed-up the calculations. Similarly, we use `dask` to parallelize the calculation of the lomb-scargle periodogram for those simulated datasets. We use both `ray` and `dask` beause using either separately for two functions was found to be a rather complicated task.

## Priors
---------
This is a 3-parameter fit, where the parameters are `beta` and `sigma_flicker`, which define the slope and amplitude of the 1/f component, and `sigma_w`, which defines the amplitude of the white-noise component. `beta` is set to be uniform by default between 0.1 and 3. Both amplitudes of the noise are set to log-uniform priors between 1 and 10. 

## Scripts and usage
-------------------

There are two scripts that define the data, prior and distance calculation functions: the `psd_utils.py` and the `data_utils.py` functions. The latter is the simulator, the former is the main function that hold the distance, simulator and prior classes.

The code is then run using the `run.py` script.
