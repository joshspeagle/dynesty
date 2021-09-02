from __future__ import (print_function, division)
import numpy as np
import dynesty
import pytest
import itertools
from utils import get_rstate, get_printing

nlive = 1000
printing = get_printing()
alpha = 1e-8


def loglike(x):
    # this is 1/|x} distribution along the x axis
    # it stops rising near zero at alpha
    # the second dimension is flat
    logl = -np.log(np.maximum(np.abs(x[0]), alpha))

    noplateau = -1e-8 * (x**2).sum()
    # this is to avoid complete plateau

    return logl + noplateau


def prior_transform(x):
    return x * 2 - 1


@pytest.mark.parametrize("bound,sample",
                         itertools.product(['multi'],
                                           ['unif', 'rslice', 'rwalk']))
def test_pathology(bound, sample):
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    logz_truth = np.log(1 - np.log(alpha))
    # this the integral
    logz, logzerr = sampler.results.logz[-1], sampler.results.logzerr[-1]
    thresh = 4
    assert (np.abs(logz - logz_truth) < thresh * logzerr)


@pytest.mark.parametrize("bound,sample",
                         itertools.product(['multi'], ['unif', 'rslice']))
def test_pathology_dynamic(bound, sample):
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           bound=bound,
                                           sample=sample,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    logz_truth = np.log(1 - np.log(alpha))
    # this the integral
    logz, logzerr = sampler.results.logz[-1], sampler.results.logzerr[-1]
    thresh = 4
    assert (np.abs(logz - logz_truth) < thresh * logzerr)
