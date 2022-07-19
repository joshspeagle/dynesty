import numpy as np
import dynesty
import pytest
import itertools
from utils import get_rstate, get_printing
"""
Run a series of basic tests of the 2d eggbox
"""

nlive = 1000
printing = get_printing()

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


@pytest.mark.parametrize(
    "scale,walks",
    itertools.product([True, False], [True, False])
)
def test_adapt(scale, walks):
    # stress test various boundaries
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound="single",
                                    sample="rwalk",
                                    rstate=rstate,
                                    adapt_scale=scale,
                                    adapt_walks=walks,
                                    )
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])
