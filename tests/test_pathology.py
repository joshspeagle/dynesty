from __future__ import (print_function, division)
import numpy as np
import dynesty
import pytest

nlive = 1000
printing = False
alpha = 1e-8


@pytest.fixture(autouse=True)
def set_seed():
    # seed the random number generator
    np.random.seed(5647)


def loglike(x):
    # this is 1/|x} distribution along the x axis
    # it stops rizing near zero at alpha
    # the second dimension is flat
    logl = -np.log(np.maximum(np.abs(x[0]), alpha))

    noplateau = -1e-10 * (x**2).sum()
    # this is to avoid complete plateau

    return logl + noplateau


def prior_transform(x):
    return x * 2 - 1


def test_pathology():
    ndim = 2
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif')
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    logz_truth = np.log(1 - np.log(alpha))
    # this the integral
    logz, logzerr = sampler.results.logz[-1], sampler.results.logzerr[-1]
    thresh = 5
    assert (np.abs(logz - logz_truth) < thresh * logzerr)
