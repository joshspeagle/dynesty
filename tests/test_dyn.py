import numpy as np
import dynesty
from utils import get_rstate, get_printing
"""
This is a hard test of dynamic sampling with the 2d eggbox
"""

nlive = 500
printing = get_printing()

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


LOGZ_TRUTH = 235.855940


def test_dyn():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    THRESHOLD = 5  # in sigmas
    rstate = get_rstate()
    # this is expected to use unif sampler and multi bound
    sampler = dynesty.DynamicNestedSampler(loglike_egg,
                                           prior_transform_egg,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    assert (abs(LOGZ_TRUTH - sampler.results.logz[-1]) <
            THRESHOLD * sampler.results.logzerr[-1])
    print(sampler.citations)
