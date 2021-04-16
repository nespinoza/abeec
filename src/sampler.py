import numpy as np
from scipy.stats import multivariate_normal

def get_weights(prior, old_weights, old_thetas, old_covariance, new_thetas, N):
    """ Function that calculates the (normalized) weights of each particle """

    weights = np.zeros(N)
    for j in range(N):
        
        theta = np.array([parameter[j] for parameter in new_thetas])
        
        # Evaluate numerator of equation (3) in Ishida et al. (2015):
        numerator = prior.evaluate(theta)

        # Evaluate denominator:
        denominator = 0.
        for i in range(N):
            old_theta = np.array([parameter[i] for parameter in old_thetas])
            denominator += old_weights[i] * multivariate_normal.pdf(theta, \
                                                                    mean = old_theta,\
                                                                    cov = old_covariance)
        
        # Save weight:
        weights[j] = numerator / denominator

    return weights/np.sum(weights)

def sample(prior, distance, simulator, M = 300, N = 30, Delta = 0.01):
    """
    Given a dataset `data`, a function that return parameters sampled from the prior `prior_sampler`, a function 
    that runs simulations given parameters `simulator` and a `distance` function that given two datasets 
    in the format of `data` returns back the distance between two datasets, this function samples from the 
    posterior using the PMC-ABC algorithm outlined in Ishida et al. (2015; https://arxiv.org/abs/1504.06129).

    Attributes
    ----------
    prior : object
        This is an object that has to have three possible functions. (1) A function `prior.sample(nsamples)` to sample from 
        the prior; this needs to return an N-tuple where each element is an array of length `nsamples` storing the samples 
        from the prior. (2) A function `prior.validate(theta)` that validates `theta` falls within the prior range; if 
        `theta` is within that, this has to return `True` and `False` otherwise. (3) A function `prior.evaluate(theta)` 
        which evaluates the prior at a given parameter vector `theta`.
    distance : object
        Object that calculates the distance from simulations to a(n already loaded) dataset. Has to have two functions; 
        `distance.single_distance` which, given a simulation, returns the distance to that simulation and `distance.several_distances`, 
        which calculates distances to `simulations.shape[0]` simulations stored in an array `simulations`.
    simulator : object
        Object that generates simulations. Must have a `simulator.single_simulation(parameter)` that generates a 
        simulation given a parameter vector and a `simulator.several_simulations(parameter)` that receives a N-tuple, 
        representing each of the parameters, and each tuple is an array, whose i-th element is the parameter for 
        the i-th particle. 
    M : int
        Number of "particles" --- i.e., draws from the prior drawn on the very first iteration of the ABC sampling scheme.
    N : int
        Number of particles on each particle system (i.e., number of particles to keep on that first iteration and in 
        each of the subsequent importance sampling iterations). N must be less than M.
    Delta : float
        Convergence criterion for the ABC sampling scheme: minimum ratio between N and the number of draws necessary to 
        construct a particle system, K. 

    """

    # Check weird user inputs:
    if N > M:
        raise Exception('N ('+str(N)+', number of particles) cannot be larger than M ('+str(M)+', number of each sub particle system). ')

    ######################################################################
    # STEP 1: find the best particles on the initial draw from the prior #
    ######################################################################
    print('\t >> 1. Starting ABC sampler...')

    # First, generate set of samples from the prior:
    thetas = prior.sample(nsamples = M)
    nparameters = len(thetas)

    # Simulate initial particle system:
    simulations = simulator.several_simulations(thetas)

    # Get distances between the dataset and the simulations; sort them out from best to worst, select best N ones:
    distances = distance.several_distances(simulations)
    idx = np.argsort(distances)[:N]
     
    # Save the N best ones to S0 (i.e., S at t=0):
    t = 0
    S = {}
    S[t] = {}
    S[t]['thetas'] = [parameter[idx] for parameter in thetas]
    S[t]['distances'] = distances[idx]

    # Get covariance matrix for these thetas:
    S[t]['covariance'] = np.cov(S[0]['thetas'])

    # Calculate and save inital weights:
    S[t]['weights'] = np.ones(N)/N

    print('\t \t Initial N particles successfully generated.')

    ######################################################################
    # STEP 2: iterate to find sub-particle systems of draws for t > 0    #
    ######################################################################

    # Define parameters that are going to be used in the iteration:
    Current_Delta = np.inf
    idx_quantile = int(N * 0.75)
    idx_N = np.arange(N)

    # Start iteration:
    print('\t >> 2. Going to iterative importance sampling. Trying to reach ', Delta)
    while Current_Delta > Delta:

        # Initialize parameters for the current iteration:
        K = 0
        Kstar = 0
        t = t + 1
        S[t] = {}

        # Calculate epsilon as the 75th quantile of distances in S[t-1]:
        epsilon = S[t-1]['distances'][idx_quantile]

        # Sample N new particles:
        counter = 0
        while True:

            Kstar += 1

            # Sample a proposed theta:
            idx_0 = np.random.choice(idx_N, p = S[t-1]['weights'])
            theta_0 = np.array([parameter[idx_0] for parameter in S[t-1]['thetas']])
            current_theta = np.random.multivariate_normal(theta_0, S[t-1]['covariance'])

            # Before running the simulation, validate that the sampled theta falls 
            # within the bounds of the prior:
            if prior.validate(current_theta):
                current_simulation = simulator.single_simulation(current_theta)
                current_distance = distance.single_distance(current_simulation)
            else:
                current_distance = np.inf

            # If it does, save the sampled theta:
            if current_distance < epsilon:
                if counter == 0:
                    thetas = np.copy(current_theta)
                    distances = np.array([current_distance])
                    Kstars = np.array([Kstar])
                else:
                    thetas = np.vstack((thetas, current_theta))
                    distances = np.append(distances, current_distance)
                    Kstars = np.append(Kstars, Kstar)
                K += Kstar
                Kstar = 0
                counter += 1

            if counter == N:
                break

        # Save the N new particles in the S dictionary. First, save the ordered distances 
        # (this ordering is needed on the next iteration):
        idx = np.argsort(distances)
        S[t]['distances'] = np.copy(distances[idx])

        # Save the ordered thetas:
        thetas = tuple(thetas.T)
        S[t]['thetas'] = [parameter[idx] for parameter in thetas]

        # Get weights:
        S[t]['weights'] = get_weights(prior, S[t-1]['weights'], S[t-1]['thetas'], \
                                      S[t-1]['covariance'], S[t]['thetas'], N)

        # Get (weighted) covariance matrix:
        S[t]['covariance'] = np.cov(S[t]['thetas'], aweights = S[t]['weights'])

        # Calculate the current delta to see if we've completed the sampling:
        Current_Delta = np.double(N)/np.double(K)

        print('\t    At t = ',t,', current Delta is ', Current_Delta,' | Target Delta : ',Delta)

    return S
