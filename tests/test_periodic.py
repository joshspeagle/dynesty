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
    assert (np.abs(dns.results['logz'][-1] - logz_true) <
            thresh * dns.results['logzerr'][-1])


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
