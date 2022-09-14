import pytest
import numpy as np
import dynesty
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
                                             ('unif', False)])
def test_periodic(sampler, dynamic):
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    logz_true = np.log(np.sqrt(2 * np.pi) * erf(win / np.sqrt(2)) / (2 * win))
    thresh = 5
    rstate = get_rstate()
    if dynamic:
        dns = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           sample=sampler,
                                           reflective=[0],
                                           rstate=rstate)
    else:
        dns = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    sample=sampler,
                                    reflective=[0],
                                    rstate=rstate)
    dns.run_nested(print_progress=printing)
    assert (np.abs(dns.results['logz'][-1] - logz_true) <
            thresh * dns.results['logzerr'][-1])


def test_error():
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     reflective=[22],
                                     rstate=rstate)
