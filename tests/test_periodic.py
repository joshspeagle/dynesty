import numpy as np
import dynesty
from scipy.special import erf
from utils import get_rstate

nlive = 100
printing = True
win = 100


def loglike(x):
    return -0.5 * x[1]**2


def prior_transform(x):
    return (2 * x - 1) * win


def test_periodic():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    logz_true = np.log(np.sqrt(2 * np.pi) * erf(win / np.sqrt(2)) / (2 * win))
    thresh = 5
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           periodic=[0],
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    assert (np.abs(sampler.results.logz[-1] - logz_true) <
            thresh * sampler.results.logzerr[-1])
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    periodic=[0],
                                    rstate=rstate)
    sampler.run_nested(dlogz=1, print_progress=printing)
    assert (np.abs(sampler.results.logz[-1] - logz_true) <
            thresh * sampler.results.logzerr[-1])
