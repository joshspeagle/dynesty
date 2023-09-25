import numpy as np
import dynesty
import dynesty.utils as dyutil
import scipy.special
from utils import get_rstate, get_printing
import pytest

printing = get_printing()


class Plateau:
    # likelihood that has value As inside a concentric
    # spheres/spherical with the radii Rs
    # the value outside the last sphere is given in As[-1]
    # everything is defined in the box -S<x<S
    def __init__(self, ndim, Rs=[1], As=[10, 1], S=3):
        self.ndim = ndim
        self.S = S
        self.Rs = np.concatenate(([0], np.array(Rs)))
        self.logAs = np.log(np.array(As))
        assert (len(self.Rs) == len(self.logAs))
        assert np.all(np.diff(Rs) > 0)
        assert Rs[-1] < S

    def __call__(self, x):
        r = np.sqrt(np.sum(x**2))
        xid = np.searchsorted(self.Rs, r, 'right')
        ret = self.logAs[xid - 1]
        return ret

    @property
    def logz_true(self):
        # true value of the integral
        n = self.ndim
        logmult = n / 2. * np.log(np.pi) - scipy.special.gammaln(n / 2. + 1)
        logvols = np.zeros(len(self.Rs))

        # volumes = pi^(n/2)/gamma(n/2+1) * (R_{k+1}^n - R_{k}^n)
        logvols[:-1] = logmult + n * np.log(
            self.Rs[1:]) + np.log1p(-(self.Rs[:-1] / self.Rs[1:])**n)
        # the last vol is (2s)^n - pi^(n/2)/gamma(n/2+1) * R[-1]^n
        logvols[-1] = n * np.log(
            2 * self.S) + np.log1p(-np.exp(logmult + n * np.log(self.Rs[-1] /
                                                                (2 * self.S))))
        logprior = -n * np.log(2 * self.S)
        logz = scipy.special.logsumexp(self.logAs + logvols) + logprior

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

    def __init__(self, ndim=2, volfrac=0.001, r0=5):
        self.size = 0.5 * r0 * (
            (np.pi**(ndim / 2.)) /
            (scipy.special.gamma(ndim / 2 + 1) * volfrac))**(1. / ndim)
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
        N = self.ndim
        lnorm = (N / 2.) * np.log(np.pi / 2) - N * np.log(self.size) + np.log(
            scipy.special.gammainc(N / 2., 0.5 * self.r0**2))
        # Assuming[(rr > 0) && (N > 1),  Integrate[Exp[-r^2/2]*r^(N - 1),
        # {r, 0, rr}]*2* Pi^(N/2)/Gamma[N/2] /(2*S)^N]
        # first factor is gaussian norm, second is from the prior
        # to integrate to 1
        return ret - lnorm

    def prior_transform(self, x):
        return (2 * x - 1) * self.size


@pytest.mark.parametrize('dynamic,', [False, True])
def test_edge(dynamic):
    rstate = get_rstate()
    ndim = 2
    ei = EdgesInf(ndim)
    nlive = 100
    if dynamic:
        CL = dynesty.DynamicNestedSampler
    else:
        CL = dynesty.NestedSampler
    sampler = CL(ei, ei.prior_transform, ei.ndim, nlive=nlive, rstate=rstate)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    THRESH = 4
    assert np.abs(res.logz[-1]) < THRESH * res.logzerr[-1]


def test_exc_small():
    rstate = get_rstate()
    ndim = 2
    ei = EdgesInf(ndim, 1e-6)
    nlive = 10
    with pytest.raises(RuntimeError):
        sampler = dynesty.NestedSampler(ei,
                                        ei.prior_transform,
                                        ei.ndim,
                                        nlive=nlive,
                                        rstate=rstate)
        sampler.run_nested(print_progress=printing)


# probe the uniform distribution
@pytest.mark.parametrize('dyn', [False, True])
def test_uniform(dyn):
    rstate = get_rstate()
    nlive = 100
    ndim = 2

    def like(x):
        return 0.

    def prior(x):
        return x

    if dyn:
        CL = dynesty.DynamicNestedSampler
    else:
        CL = dynesty.NestedSampler
    sampler = CL(
        like,
        prior,
        ndim,
        nlive=nlive,
        rstate=rstate,
    )
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - 0) < THRESH * res.logzerr[-1]


# test uniform distribution with very low
# likelihood
@pytest.mark.parametrize('dyn', [False, True])
def test_uniform1(dyn):
    rstate = get_rstate()
    nlive = 100
    ndim = 2

    def like(x):
        return -1e100

    def prior(x):
        return x

    if dyn:
        CL = dynesty.DynamicNestedSampler
    else:
        CL = dynesty.NestedSampler
    sampler = CL(
        like,
        prior,
        ndim,
        nlive=nlive,
        rstate=rstate,
    )
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    assert res is not None
