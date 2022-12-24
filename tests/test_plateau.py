import numpy as np
import dynesty
import dynesty.utils as dyutil
import scipy.special
from utils import get_rstate, get_printing
import pytest

printing = get_printing()


class Plateau:
    # likelihood that has value A1 inside a sphere with the radius R
    # and outside it has velue A0
    def __init__(self, ndim):
        self.ndim = ndim
        self.S = 3
        self.R = 1
        self.A0 = 1
        self.A1 = 10

    def __call__(self, x):
        r = np.sqrt(np.sum(x**2))
        if r < self.R:
            ret = np.log(self.A1)  # - 1e-6 * r
        else:
            ret = np.log(self.A0)  # - 1e-6 * r
        # print(ret, r)
        return ret

    @property
    def logz_true(self):
        # true value of the integral
        logz = np.log(self.A0 + np.pi**(self.ndim / 2.) /
                      scipy.special.gamma(self.ndim / 2. + 1) *
                      self.R**self.ndim * (self.A1 - self.A0) /
                      ((2 * self.S)**self.ndim))
        return logz

    def prior_transform(self, x):
        return (2 * x - 1) * self.S


# here are are trying to test different stages of plateau
# probing with different dlogz's
@pytest.mark.parametrize('sample,dlogz', [('unif', 1), ('rwalk', 1),
                                          ('rslice', 1), ('unif', .01),
                                          ('rwalk', .01), ('rslice', .01)])
def test_static(sample, dlogz):
    nlive = 1000
    rstate = get_rstate()
    ndim = 2
    plateau = Plateau(ndim)
    sampler = dynesty.NestedSampler(plateau,
                                    plateau.prior_transform,
                                    plateau.ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    bound='none',
                                    sample=sample)
    sampler.run_nested(print_progress=printing, dlogz=dlogz)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - plateau.logz_true) < THRESH * res.logzerr[-1]


# here are are trying to test different stages of plateau
# probing with different dlogz's
@pytest.mark.parametrize('sample,', ['unif', 'rslice', 'rwalk'])
def test_dynamic(sample):
    rstate = get_rstate()
    nlive = 100
    ndim = 2
    plateau = Plateau(ndim)
    sampler = dynesty.DynamicNestedSampler(plateau,
                                           plateau.prior_transform,
                                           plateau.ndim,
                                           nlive=nlive,
                                           rstate=rstate,
                                           bound='none',
                                           sample=sample)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - plateau.logz_true) < THRESH * res.logzerr[-1]


# here are are trying to test different stages of plateau
# probing with different dlogz's
def test_merge():
    nlive = 100
    rstate = get_rstate()
    res_list = []
    ndim = 2
    plateau = Plateau(ndim)
    for i in range(3):
        sampler = dynesty.NestedSampler(plateau,
                                        plateau.prior_transform,
                                        plateau.ndim,
                                        nlive=nlive,
                                        rstate=rstate,
                                        bound='none',
                                        sample='unif')
        sampler.run_nested(print_progress=printing)
        res_list.append(sampler.results)
    res = dyutil.merge_runs(res_list)
    THRESH = 3
    assert np.abs(res.logz[-1] - plateau.logz_true) < THRESH * res.logzerr[-1]


class WeddingCake:
    # Wedding cake function from Fowlie 2020
    def __init__(self, ndim, sig=.2, alpha=.7):
        self.ndim = ndim
        self.sig = sig
        self.alpha = alpha

    def __call__(self, x):
        D = len(x)
        r = np.max(np.abs(x - 0.5))
        i = (D * np.log(2 * r) / np.log(self.alpha)).astype(int)
        logp = -(self.alpha**(2 * i / D)) / (8 * self.sig**2)
        return logp

    @property
    def logz_true(self):
        logz = scipy.special.logsumexp(
            -self.alpha**(2 * np.arange(100) / self.ndim) / (8 * self.sig**2) +
            np.arange(100) * np.log(self.alpha) + np.log((1 - self.alpha)))
        return logz

    def prior_transform(self, x):
        return x


# here are are trying to test different stages of plateau
# probing with different dlogz's
@pytest.mark.parametrize('sample', ['unif', 'rwalk', 'rslice'])
def test_cake_static(sample):
    nlive = 1000
    rstate = get_rstate()
    ndim = 5
    cake = WeddingCake(ndim)
    sampler = dynesty.NestedSampler(cake,
                                    cake.prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    sample=sample)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - cake.logz_true) < THRESH * res.logzerr[-1]


@pytest.mark.parametrize('sample,', ['unif', 'rslice', 'rwalk'])
def test_cake_dynamic(sample):
    rstate = get_rstate()
    nlive = 100
    ndim = 5
    cake = WeddingCake(ndim)
    sampler = dynesty.DynamicNestedSampler(cake,
                                           cake.prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate,
                                           sample=sample)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - cake.logz_true) < THRESH * res.logzerr[-1]


class EdgesInf:
    # I'm putting a gaussian and truncating its wings to -inf

    def __init__(self, ndim=2, volfrac=0.0001, r0=5):
        self.size = r0 / volfrac**(1. / ndim)
        self.r0 = r0
        assert (self.size > r0)
        self.ndim = ndim

    def __call__(self, x):
        r2 = np.sum(x**2)
        r = np.sqrt(r2)

        if r > self.r0:
            return -np.inf
        else:
            ret = -0.5 * r2
        lnorm = -self.ndim / 2. * np.log(2 * np.pi) + self.ndim * np.log(
            2 * self.size)
        # first factor is gaussian norm, second is from the prior
        # to integrate to 1
        return ret + lnorm

    def prior_transform(self, x):
        return (2 * x - 1) * self.size


# TODO THIS TEMPORARILY DISABLED
# BEFORE THE INITIAL SAMPLING of -inf is fixed
def test_edge():
    rstate = get_rstate()
    ndim = 2
    ei = EdgesInf(ndim)
    nlive = 100
    sampler = dynesty.NestedSampler(ei,
                                    ei.prior_transform,
                                    ei.ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(print_progress=True)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1]) < THRESH * res.logzerr[-1]
