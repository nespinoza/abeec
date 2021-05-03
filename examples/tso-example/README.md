# Fitting a stochastic process with `abeec`
-----------------------------

In this example, we fit the parameters of a realization of a `1/f**beta` noise process plus *poisson* noise. 

As a distance metric, we use the autocorrelation fuction. The `tso_utils.py` contains the simulators to simulate fake data, the distance function and the prior. The priors 
in turn are uniform for the `beta` paramter, and log-uniform for the square-root of the variance of the Poisson process and for the amplitude of the 1/f process. 

The `tso_example.py` script is ready-to-run; it will plot at the end a corner plot of the fitted parameters, along with realizations of the power-spectral density of the 
process and the autocorrelation function.
