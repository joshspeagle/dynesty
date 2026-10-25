import numpy as np
import pytest
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


# simple gaussian for quick tests
def loglike_gau(x):
    return -0.5 * np.sum(x**2)


def prior_transform_gau(u):
    return 20 * u - 10


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
    assert (abs(LOGZ_TRUTH - sampler.results.logz[-1])
            < THRESHOLD * sampler.results.logzerr[-1])
    print(sampler.citations)


def test_add_batch_default_nlive():
    # add_batch() without nlive should use the nlive given at
    # initialization, like run_nested() does
    ndim = 2
    nlive0 = 60
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           ndim,
                                           nlive=nlive0,
                                           rstate=get_rstate())
    sampler.run_nested(maxbatch=0, print_progress=printing)
    sampler.add_batch(print_progress=printing)
    assert sampler.results.batch_nlive[-1] == nlive0


def test_stop_kwargs_target_n_effective():
    # the target_n_effective in stop_kwargs is used and
    # the user's dictionary is not modified
    target = 1500
    stop_kwargs = {'target_n_effective': target}
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           2,
                                           nlive=100,
                                           rstate=get_rstate())
    sampler.run_nested(stop_kwargs=stop_kwargs, print_progress=printing)
    assert stop_kwargs == {'target_n_effective': target}
    # the default target (10000) would need many more samples
    assert target <= sampler.n_effective < 10000


def test_stop_n_effective_conflict():
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           2,
                                           nlive=100,
                                           rstate=get_rstate())
    with pytest.raises(ValueError):
        sampler.run_nested(n_effective=1000,
                           stop_kwargs={'target_n_effective': 2000},
                           print_progress=printing)


def test_stop_n_effective_custom_stop_function():
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           2,
                                           nlive=100,
                                           rstate=get_rstate())
    with pytest.warns(UserWarning, match='n_effective is ignored'):
        sampler.run_nested(n_effective=1000,
                           stop_function=dynesty.dynamicsampler.
                           stopping_function,
                           maxbatch=0,
                           print_progress=printing)
