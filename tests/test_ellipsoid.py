import dynesty.bounding as db
import numpy as np
import scipy.stats
import pytest


@pytest.fixture
def set_seed():
    np.random.seed(143463473)


def test_sample():
    # test sampling of two overlapping ellipsoids that samples are uniform
    # within
    rad = 1
    shift = 0.75
    ndim = 10
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    cen2[0] = shift
    sig = np.eye(10) * rad**2
    ells = [db.Ellipsoid(cen1, sig), db.Ellipsoid(cen2, sig)]
    mu = db.MultiEllipsoid(ells)
    R = []
    nsim = 100000
    for i in range(nsim):
        R.append(mu.sample()[0])
    R = np.array(R)
    assert (all([mu.contains(_) for _ in R]))

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
        assert ((pval > 0.003) & (pval < 0.997))
    nhalf = (R[:, 0] > shift / 2.).sum()
    print(nhalf, nsim)
    assert (np.abs(nhalf - 0.5 * nsim) < 5 * np.sqrt(0.5 * nsim))


def test_sample_q():
    # test sampling of two overlapping ellipsoids that samples are uniform
    # within
    rad = 1
    shift = 0.75
    ndim = 10
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    cen2[0] = shift
    sig = np.eye(10) * rad**2
    ells = [db.Ellipsoid(cen1, sig), db.Ellipsoid(cen2, sig)]
    mu = db.MultiEllipsoid(ells)
    R = []
    nsim = 100000
    for i in range(nsim):
        while True:
            x, _, q = mu.sample(return_q=True)
            if np.random.rand() < 1. / q:
                R.append(x)
                break
    R = np.array(R)
    assert (all([mu.contains(_) for _ in R]))
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
        assert ((pval > 0.003) & (pval < 0.997))
    nhalf = (R[:, 0] > shift / 2.).sum()
    print(nhalf, nsim)
    assert (np.abs(nhalf - 0.5 * nsim) < 5 * np.sqrt(0.5 * nsim))
