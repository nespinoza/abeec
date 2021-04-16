# abeec 🐝 --- an ABC sampler
----------------------------------------------------------------

`abeec` is a sampler to perform Approximate Bayesian Computation (ABC) --- i.e., likelihood free posterior inference! It is based on the algorithm presented in [Ishida et al. (2015)](https://arxiv.org/abs/1504.06129).

**Author**: Nestor Espinoza (nespinoza@stsci.edu)

## Statement of need
While for Cosmological applications an ABC sampler has already been published by the team of Ishida et al (`cosmoabc` --- [check their repository!](https://github.com/COINtoolbox/CosmoABC)), 
there was a need to develop a more general scheme to allow some flexibility to the sampler. For instance, doing arbitrary prior distributions (e.g., with priors that might be correlated) 
was not straightforward to implement, as well as have external functions for distances and simulators that could all benefit from a common parallelization scheme. On top of that, I wanted 
a simple sampler that used the most basic `python` libraries (e.g., `numpy` and `scipy`) at its core. That's where `abeec` comes into place.

## Using the library
To perform ABC on a given dataset, you need three ingredients:

1. A `prior` from which to draw points.
2. A `distance` to compute distances from simulated datasets to your dataset.
3. And a `simulator`, to simulate datasets to compare against your dataset.

In `abeec`, it is expected the user will provide **_classes_** defining the `prior`, the `distance` and the `simulator`. All the sampler does it take those and apply the iterative importance 
sampling scheme outlined in [Ishida et al. (2015)](https://arxiv.org/abs/1504.06129), giving back a sample from the posterior. Once those classes are written, one might simply run the 
sampler as:

        import abeec
        from your_script import prior, distance, simulator

        samples = abeec.sample(prior, distance, simulator)

The best is to check the examples under `examples`.

## Installation
Installation is as simple as:

        python setup.py install

Or via PyPi:

        pip install abeec

## Licence and attribution

Read the `LICENCE` file for licencing details on how to use the code. If you make use of this code, please cite [Ishida et al. (2015)](https://arxiv.org/abs/1504.06129) and link back 
to this repository.
