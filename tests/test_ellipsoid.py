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


def test_samples_single():
    rstate = get_rstate()
    ndim = 10
    cen = np.zeros(ndim) + .5
    sig = np.eye(ndim)
    ell = db.Ellipsoid(cen, sig)
    nsamp = 10000
    X = ell.samples(nsamp, rstate=rstate)
    R = np.sqrt(((X - cen[None, :])**2).sum(axis=1))
    xdist1 = R**ndim
    pval = scipy.stats.kstest(xdist1,
                              scipy.stats.uniform(loc=0.0, scale=1).cdf)[1]
    assert ((pval > 1e-3) and (pval < 1 - 1e-3))


def test_samples_multi():
    rstate = get_rstate()
    ndim = 10
    cen = np.zeros(ndim) + .5
    sig = np.eye(ndim)
    ell = db.MultiEllipsoid([db.Ellipsoid(cen, sig)])
    nsamp = 10000
    X = ell.samples(nsamp, rstate=rstate)
    R = np.sqrt(((X - cen[None, :])**2).sum(axis=1))
    xdist1 = R**ndim
    pval = scipy.stats.kstest(xdist1,
                              scipy.stats.uniform(loc=0.0, scale=1).cdf)[1]
    assert ((pval > 1e-3) and (pval < 1 - 1e-3))


def test_cube_overlap():
    rstate = get_rstate()
    ndim = 10
    cen = np.zeros(ndim) + .5
    cen[0] = 0
    sig = np.eye(ndim) * .5**2
    ell = db.Ellipsoid(cen, sig)
    nsamp = 10000
    frac = ell.unitcube_overlap(nsamp, rstate=rstate)
    true_answer = 0.5
    uncertainty = np.sqrt(true_answer * (1 - true_answer) / nsamp)
    assert ((frac - true_answer) < 5 * uncertainty)


def test_overlap():
    rstate = get_rstate()
    ndim = 2
    cen1 = np.array([0, 0])
    cen2 = np.array([1, 0])
    rad = 0.7
    sig = np.eye(ndim) * rad**2

    ell1 = db.Ellipsoid(cen1, sig)
    ell2 = db.Ellipsoid(cen2, sig)
    ell = db.MultiEllipsoid([ell1, ell2])
    nsamp = 10000
    xs = rstate.uniform(size=(nsamp, ndim))
    ind1 = np.sum((xs - cen1[None, :])**2, axis=1) < rad**2
    ind2 = np.sum((xs - cen2[None, :])**2, axis=1) < rad**2
    for i in range(nsamp):
        n1 = int(ind1[i])
        n2 = int(ind2[i])
        assert ell.overlap(xs[i]) == n1 + n2
        assert ell.overlap(xs[i], j=0) == n2
        within = ell.within(xs[i])
        within2 = []
        if n1 == 1:
            within2.append(0)
        if n2 == 1:
            within2.append(1)
        within2 = np.array(within2)
        assert np.all(within == within2)


def capvol(n, r, h):
    # this is a volume of the n-dimensional spherical cap
    # see https://en.wikipedia.org/wiki/Spherical_cap
    Cn = np.pi**(n / 2.) / scipy.special.gamma(1 + n / 2.)
    return (Cn * r**n *
            (1 / 2 - (r - h) / r * scipy.special.gamma(1 + n / 2) /
             np.sqrt(np.pi) / scipy.special.gamma(
                 (n + 1.) / 2) * scipy.special.hyp2f1(1. / 2,
                                                      (1 - n) / 2, 3. / 2,
                                                      ((r - h) / r)**2)))


def sphere_vol(n, r):
    # n-d sphere volume
    Cn = np.pi**(n / 2.) / scipy.special.gamma(1 + n / 2.) * r**n
    return Cn


def two_sphere_vol(c1, c2, r1, r2):
    # Return the volume of the unions of two n-d spheres
    D = np.sqrt(np.sum((c1 - c2)**2))
    n = len(c1)
    if D >= r1 + r2:
        return sphere_vol(n, r1) + sphere_vol(n, r2)
    # now either one is fully inside or the is overlap
    if D + r1 <= r2 or D + r2 <= r1:
        # fully inside
        return max(sphere_vol(n, r1), sphere_vol(n, r2))
    else:
        x = 1. / 2 / D * np.sqrt(2 * r1**2 * r2**2 + 2 * D**2 * r1**2 +
                                 2 * D**2 * r2**2 - r1**4 - r2**4 - D**4)
        capsize1 = r1 - np.sqrt(r1**2 - x**2)
        capsize2 = r2 - np.sqrt(r2**2 - x**2)
        V = (sphere_vol(n, r1) + sphere_vol(n, r2) - capvol(n, r1, capsize1) -
             capvol(n, r2, capsize2))
        return V


def test_mc_logvol():

    rstate = get_rstate()
    ndim = 10
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    r1 = 1
    r2 = 0.5
    sig1 = np.eye(ndim) * r1**2
    sig2 = np.eye(ndim) * r2**2
    Ds = np.linspace(0, 2, 30)
    nsamp = 10000
    for D in Ds:
        cen2[0] = D
        ell = db.MultiEllipsoid(
            [db.Ellipsoid(cen1, sig1),
             db.Ellipsoid(cen2, sig2)])
        lv = ell.monte_carlo_logvol(nsamp, rstate=rstate)[0]
        vtrue = two_sphere_vol(cen1, cen2, r1, r2)
        assert (np.abs(np.log(vtrue) - lv) < 1e-2)


def test_mc_logvolRad():

    rstate = get_rstate()
    ndim = 10
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    r1 = 0.7
    sig1 = np.eye(ndim) * r1**2
    Ds = np.linspace(0, 2, 30)
    nsamp = 10000
    for D in Ds:
        cen2[0] = D
        rf = db.RadFriends(ndim, sig1)
        lv = rf.monte_carlo_logvol(np.array([cen1, cen2]),
                                   nsamp,
                                   rstate=rstate)[0]
        vtrue = two_sphere_vol(cen1, cen2, r1, r1)
        assert (np.abs(np.log(vtrue) - lv) < 1e-2)


def test_mc_logvolCube():

    rstate = get_rstate()
    ndim = 10
    cen1 = np.zeros(ndim)
    cen2 = np.zeros(ndim)
    r1 = 0.7
    sig1 = np.eye(ndim) * r1**2
    Ds = np.linspace(0, 2, 30)
    nsamp = 10000
    for D in Ds:
        cen2[0] = D
        rf = db.SupFriends(ndim, sig1)
        lv = rf.monte_carlo_logvol(np.array([cen1, cen2]),
                                   nsamp,
                                   rstate=rstate)[0]
        if D > 2 * r1:
            lvtrue = np.log(2 * r1) * ndim + np.log(2)
        else:
            lvtrue = np.log(2 * r1) * (ndim - 1) + np.log(D + 2 * r1)
        exp_frac = np.exp(lvtrue - (np.log(2 * r1) * ndim + np.log(2)))
        nexp = exp_frac * nsamp
        rel_error = 1. / np.sqrt(nexp)
        threshold = 4
        assert (np.abs(lvtrue - lv) < threshold * rel_error)


def test_bounds():
    rstate = get_rstate()
    Ndim = 20
    Ngood = 10
    eigv = np.abs(rstate.normal(size=Ngood))
    full_eig = np.concatenate((eigv, np.zeros(Ndim - Ngood)))
    randM = rstate.normal(size=((Ndim, Ndim)))
    M = randM.T @ np.diag(full_eig) @ randM
    R = db.improve_covar_mat(M)[1]
    assert ((scipy.linalg.eigh(R)[0] > 0).all())

    M = np.zeros((Ndim, Ndim))
    R = db.improve_covar_mat(M)[1]
    assert ((scipy.linalg.eigh(R)[0] > 0).all())


@pytest.mark.parametrize("ndim", [1, 10, 100])
def test_bounding_crazy(ndim):
    rstate = get_rstate()
    xs = rstate.normal(size=ndim * 10)
    ys = rstate.normal(size=ndim)
    zs = xs[:, None] * ys[None, :]
    db.bounding_ellipsoids(zs)


def test_number_clusters():
    # check we can recover close to a correct and large number of clusters
    ndim = 4
    nxcens = 6
    ncens = nxcens**ndim
    sig = 0.01
    THRESHOLD = 0.1  # check we get the right number of clusters
    # within 10%

    npt = ncens * 10 * ndim
    rstate = get_rstate()
    # create stuff on the grid
    cens = np.array(list(itertools.product(*([np.arange(nxcens)] * ndim))))

    xs = sig * rstate.normal(size=(npt, ndim)) + cens[np.arange(npt) %
                                                      len(cens), :]

    E = db.bounding_ellipsoids(xs)

    assert np.abs(len(E.ells) * 1. / ncens - 1) < THRESHOLD
