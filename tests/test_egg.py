from __future__ import (print_function, division)
from six.moves import range
import numpy as np
from numpy import linalg
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa
import dynesty  # noqa
from dynesty import plotting as dyplot  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

# seed the random number generator
np.random.seed(5647)

nlive = 1000
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
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif')
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def test_ellipsoids_bootstrap():
    # stress test ellipsoid decompositions
    ndim = 2
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    bootstrap=5)
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])
