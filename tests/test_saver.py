from __future__ import (print_function, division)
import numpy as np
import dynesty
import os
"""
Run a series of basic tests to check whether anything huge is broken.

"""

# seed the random number generator
np.random.seed(5647)

nlive = 100
printing = False

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


def test_ellipsoids():
    # stress test ellipsoid decompositions
    ndim = 2
    fname = 'xx.h5'
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    save_history=True,
                                    history_filename=fname)
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    assert (os.path.exists(fname))
