import numpy as np
import dynesty
import pytest
from scipy.special import erf
from scipy import stats
from utils import get_rstate, get_printing

nlive = 100
printing = get_printing()
win = 100
ndim = 2


def loglike(x):
    return -0.5 * x[1]**2


def prior_transform(x):
    return (2 * x - 1) * win


@pytest.mark.parametrize("sampler,dynamic", [('rwalk', True), ('unif', True),
                                             ('rslice', True),
                                             ('unif', False)])
def test_periodic(sampler, dynamic):
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    logz_true = np.log(np.sqrt(2 * np.pi) * erf(win / np.sqrt(2)) / (2 * win))
    thresh = 8
    # This is set up to higher level
    # becasue of failures at ~5ssigma level
    # this needs to be investigated
    rstate = get_rstate()
    if dynamic:
        dns = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           periodic=[0],
                                           rstate=rstate,
                                           sample=sampler)
        dns.run_nested(dlogz_init=1, print_progress=printing)
    else:
        dns = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    periodic=[0],
                                    rstate=rstate,
                                    sample=sampler)
        dns.run_nested(dlogz=1, print_progress=printing)
    assert (np.abs(dns.results['logz'][-1] - logz_true)
            < thresh * dns.results['logzerr'][-1])


def test_error():
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[22],
                                     rstate=rstate)


def test_reflective_ignored():
    # reflective option is not supported anymore, it should warn
    rstate = get_rstate()
    with pytest.warns(UserWarning, match='Reflective'):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[1],
                                     reflective=[1],
                                     rstate=rstate)


def wrapdist(x):
    # distance from the periodic boundary x=0 (==1), in (-0.5, 0.5]
    return np.mod(x + 0.5, 1) - 0.5


SIGMA_BOUNDARY = 0.05


def loglike_boundary(v):
    # Gaussian centered on the periodic boundary of the first parameter
    return -0.5 * (wrapdist(v[0])**2 + (v[1] - 0.5)**2) / SIGMA_BOUNDARY**2


def prior_transform_unit(u):
    return u


@pytest.mark.parametrize("sampler,bound",
                         [('unif', 'single'), ('unif', 'multi'),
                          ('unif', 'balls'), ('unif', 'cubes'),
                          ('rslice', 'multi'), ('rwalk', 'multi')])
def test_periodic_proposal(sampler, bound):
    # A new live point drawn by the internal sampler must be uniformly
    # distributed within the constrained prior. Here the constrained region
    # is a disk straddling the periodic boundary of x, so the marginal
    # distributions of the wrapped x and of y are semicircular.
    rstate = get_rstate()
    ndim_, nlive_, ndraws = 2, 200, 4000
    rx = 0.1  # radius of the constrained disk
    loglstar = -0.5 * (rx / SIGMA_BOUNDARY)**2

    # live points uniform within the disk, stored wrapped into [0,1)
    r = np.sqrt(rstate.uniform(0, 1, nlive_))
    phi = rstate.uniform(0, 2 * np.pi, nlive_)
    live_u = np.column_stack(
        [np.mod(rx * r * np.cos(phi), 1), 0.5 + rx * r * np.sin(phi)])

    ns = dynesty.NestedSampler(loglike_boundary,
                               prior_transform_unit,
                               ndim_,
                               nlive=nlive_,
                               sample=sampler,
                               bound=bound,
                               periodic=[0],
                               enlarge=2.,
                               rstate=rstate)
    ns.live_u[:] = live_u
    ns.update_bound_if_needed(-np.inf, force=True)
    # the boundary of the bound frame must sit in the gap opposite to
    # the cluster, making the cluster contiguous
    assert abs(ns.bound.shift[0] - 0.5) < rx
    assert ns.bound.shift[1] == 0
    if bound == 'multi':
        assert ns.bound.nells == 1

    # starting points and axes as the nested sampler would provide them
    points, axes = zip(*[ns.propose_live() for i in range(ndraws)])
    args = ns.internal_sampler.prepare_sampler(
        loglstar=loglstar,
        points=points,
        axes=axes,
        seeds=rstate.integers(0, 2**31, ndraws),
        prior_transform=prior_transform_unit,
        loglikelihood=loglike_boundary,
        nested_sampler=ns)
    us = np.array([ns.internal_sampler.sample(a).u for a in args])
    # returned unit-cube coordinates must be inside the unit cube
    assert np.all((us >= 0) & (us < 1))
    # and uniform within the constrained disk
    pval = stats.kstest(wrapdist(us[:, 0]) / rx, stats.semicircular.cdf).pvalue
    assert pval > 1e-3
    pval = stats.kstest((us[:, 1] - 0.5) / rx, stats.semicircular.cdf).pvalue
    assert pval > 1e-3


@pytest.mark.parametrize("sampler,bound",
                         [('unif', 'single'), ('unif', 'multi'),
                          ('unif', 'balls'), ('rwalk', 'multi'),
                          ('rslice', 'multi')])
def test_periodic_boundary_mode(sampler, bound):
    # full run with the posterior mode on the periodic boundary
    rstate = get_rstate()
    logz_true = 2 * np.log(SIGMA_BOUNDARY * np.sqrt(2 * np.pi) *
                           erf(0.5 / np.sqrt(2) / SIGMA_BOUNDARY))
    thresh = 5
    ns = dynesty.NestedSampler(loglike_boundary,
                               prior_transform_unit,
                               2,
                               nlive=nlive,
                               periodic=[0],
                               rstate=rstate,
                               sample=sampler,
                               bound=bound)
    ns.run_nested(dlogz=0.1, print_progress=printing)
    res = ns.results
    assert np.all((res['samples_u'] >= 0) & (res['samples_u'] < 1))
    assert (np.abs(res['logz'][-1] - logz_true) < thresh * res['logzerr'][-1])
