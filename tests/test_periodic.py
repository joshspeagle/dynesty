import numpy as np
import dynesty
import pytest
from scipy.special import erf
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


@pytest.mark.parametrize("sample", ['slice', 'rslice'])
def test_periodic_seam(sample):
    # A periodic posterior whose mode sits *on* the wrap seam (x0 = 0 == 1).
    # The slice samplers must wrap across the boundary to sample it correctly;
    # we check the recovered ln(evidence) against the analytic value.
    from scipy.special import i0
    kappa, sigma = 1.0, 0.1

    def seam_loglike(x):
        return kappa * np.cos(2 * np.pi * x[0]) - 0.5 * (
            (x[1] - 0.5) / sigma)**2

    def unit_ptform(u):
        return u

    logz_true = np.log(i0(kappa)) + np.log(sigma) + 0.5 * np.log(2 * np.pi)
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(seam_loglike,
                                    unit_ptform,
                                    2,
                                    nlive=200,
                                    periodic=[0],
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.05, print_progress=printing)
    res = sampler.results
    assert np.abs(res.logz[-1] - logz_true) < 5 * res.logzerr[-1]


def test_error():
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[22],
                                     rstate=rstate)


def test_error2():
    # check you cant combine periodic/reflective for one var
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[1],
                                     reflective=[1],
                                     rstate=rstate)
