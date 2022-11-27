import numpy as np
import dynesty
import scipy.special
from utils import get_rstate, get_printing
import pytest

S = 3
R = 1

ndim = 2
A0 = 1
A1 = 10

printing = get_printing()


# likelihood that has value A1 inside a sphere with the radius R
# and outside it has velue A0
def loglike_inf(x):
    r = np.sqrt(np.sum(x**2))
    if r < R:
        ret = np.log(A1)  # - 1e-6 * r
    else:
        ret = np.log(A0)  # - 1e-6 * r
    # print(ret, r)
    return ret


# true value of the integral
LOGZ_TRUE = np.log(A0 + np.pi**(ndim / 2.) /
                   scipy.special.gamma(ndim / 2. + 1) * R**ndim * (A1 - A0) /
                   ((2 * S)**ndim))


def prior_transform(x):
    return (2 * x - 1) * S


nlive = 1000


# here are are trying to test different stages of plateau
# probing with different dlogz's
@pytest.mark.parametrize('sample,dlogz', [('unif', 1), ('rwalk', 1),
                                          ('rslice', 1), ('unif', .01),
                                          ('rwalk', .01), ('rslice', .01)])
def test_static(sample, dlogz):
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_inf,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    bound='none',
                                    sample=sample)
    sampler.run_nested(print_progress=printing, dlogz=dlogz)
    res = sampler.results
    THRESH = 3
    assert np.abs(res.logz[-1] - LOGZ_TRUE) < THRESH * res.logzerr[-1]
