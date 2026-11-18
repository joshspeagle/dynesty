"""Zero posterior mass must not contaminate the information integral."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import xlogy
from dynesty import NestedSampler
from dynesty import utils as dyutil
from utils import get_rstate


def test_reweight_zero_information():
    logl = np.array([-3., -2., -1., 0.])
    volume = np.array([.8, .5, .2, .05])
    samples = np.arange(4.)[:, None]
    res = dyutil.Results(dict(nlive=2, niter=2,
                              samples=samples, samples_u=samples / 4,
                              samples_id=np.array([0, 1, 0, 1]), logl=logl,
                              logvol=np.log(volume),
                              logwt=np.zeros(4), logz=np.zeros(4),
                              logzerr=np.zeros(4), information=np.zeros(4)))
    new_logl = logl.copy()
    new_logl[[0, 2]] = -np.inf
    updated = dyutil.reweight_run(res, new_logl)
    # Direct quadrature in linear space, including 0 * log(0) = 0.
    factor = np.exp(new_logl - logl)
    right = np.exp(logl) * factor
    left = np.r_[0., np.exp(logl[:-1])] * factor
    width = -np.diff(np.r_[1., volume])
    mass = .5 * (right + left) * width
    logmass = .5 * (xlogy(right, right) + xlogy(left, left)) * width
    expected = np.cumsum(logmass) / mass.sum()
    expected -= np.cumsum(mass) / mass.sum() * np.log(mass.sum())
    assert np.all(np.isfinite(updated.information))
    assert_allclose(updated.information, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(updated.logz[-1], np.log(mass.sum()), atol=1e-12)
    assert_allclose(res.information, 0.)


def test_reweight_truncated_gaussian():
    sampler = NestedSampler(lambda x: -.5 * np.dot(x, x),
                             lambda u: 20 * u - 10, 1, nlive=30,
                             rstate=get_rstate())
    sampler.run_nested(dlogz=.1, print_progress=False)
    res = sampler.results
    new_logl = res.logl.copy()
    new_logl[res.samples[:, 0] < 0] = -np.inf
    updated = dyutil.reweight_run(res, new_logl)
    assert np.all(np.isfinite(updated.information))
    assert np.isfinite(updated.logz[-1])
    control = dyutil.reweight_run(res, res.logl)
    assert_allclose(control.information, res.information)
