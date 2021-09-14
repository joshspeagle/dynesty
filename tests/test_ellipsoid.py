import itertools
import numpy as np
import scipy.stats
import pytest
import dynesty.bounding as db
from utils import get_rstate

FAILURE_THRESHOLD = 1 / 1000.
# I want this test to fail once in 1000 iterations
# because there are 16 of these tests and the probability is twosided
PVAL = FAILURE_THRESHOLD / 16 / 2.


@pytest.mark.parametrize("withq,ndim",
                         list(itertools.product([False, True], [2, 10])))
def test_sample(withq, ndim):
    # test sampling of two overlapping ellipsoids that samples are uniform
    # within
    rad = 1
    shift = 0.75
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    cen2[0] = shift
    sig = np.eye(ndim) * rad**2
    ells = [db.Ellipsoid(cen1, sig), db.Ellipsoid(cen2, sig)]
    mu = db.MultiEllipsoid(ells)
    R = []
    nsim = 100000
    rstate = get_rstate()
    if withq:
        for i in range(nsim):
            while True:
                x, _, q = mu.sample(return_q=True, rstate=rstate)
                if rstate.uniform() < 1. / q:
                    R.append(x)
                    break
    else:
        for i in range(nsim):
            R.append(mu.sample(rstate=rstate)[0])
    R = np.array(R)
    assert (all([mu.contains(_) for _ in R]))
    assert (all([ells[0].contains(_) or ells[1].contains(_) for _ in R]))

    # here I'm checking that all the points are uniformly distributed
    # within each ellipsoid
    for curc in [cen1, cen2]:
        dist1 = (np.sqrt(np.sum((R - curc)**2, axis=1)) / rad)
        # radius from 0 to 1
        xdist1 = dist1**ndim
        # should be uniformly distributed from 0 to 1
        xdist1 = xdist1[xdist1 < 1]
        pval = scipy.stats.kstest(xdist1,
                                  scipy.stats.uniform(loc=0.0, scale=1).cdf)[1]
        assert ((pval > PVAL) & (pval < (1 - PVAL)))
    nhalf = (R[:, 0] > shift / 2.).sum()
    assert (np.abs(nhalf - 0.5 * nsim) < 5 * np.sqrt(0.5 * nsim))
