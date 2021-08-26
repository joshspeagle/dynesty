import math
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import dynesty.bounding as db
"""
This is the code that allows to calibrate how much volume we are missing
with ellipsoidal representation

This is not part of the test suite
"""


def genball(npt, ndim, rstate=None):
    """ Simulate points in ndim ball """
    # use Barthe2005
    x = rstate.standard_normal(size=(npt, ndim))
    y = rstate.exponential(0.5, size=npt)
    x1 = x / np.sqrt((y + (x**2).sum(axis=1)))[:, None]
    return x1


def genshell(r1, r2, npt, ndim, rstate=None):
    """ Simulate points in a ndim shell """
    x = rstate.standard_normal(size=(npt, ndim))
    xnorm = x / ((x**2).sum(axis=1)**.5)[:, None]
    # normed vector
    # radii are distributed like R^(ndim-1)
    # cumul (R^ndim-r1^ndim)/(r2^ndim-r1^ndim)=y
    rs = ((r2**ndim - r1**ndim) * rstate.uniform(size=npt) + r1**ndim)**(1. /
                                                                         ndim)
    return rs[:, None] * xnorm


def gen_data(npt, typ, ndim, rstate=None):
    """ Simulate points with different topologies
    typ can be ball/shell/pin/torus
    """
    mid = .5  # i'm placing in unit cube
    if typ == 'ball':
        r0 = 0.5
        pts = genball(npt, ndim, rstate=rstate) * r0 + mid
        volume = (np.pi**(ndim / 2) / scipy.special.gamma(ndim / 2 + 1) *
                  r0**ndim)
    elif typ == 'pin':
        w = 0.01
        a = 1
        pts = np.zeros((npt, ndim))
        pts[:, 1:] = genball(npt, ndim - 1, rstate=rstate) * w + mid
        pts[:, 0] = (rstate.uniform(size=npt) - 0.5) * a + mid
        volume = (np.pi**((ndim - 1) / 2) /
                  scipy.special.gamma((ndim - 1) / 2 + 1) * w**(ndim - 1) * a)
    elif typ == 'torus':
        w = 0.01
        r0 = 0.45
        pts = np.zeros((npt, ndim))
        pts[:, :2] = genshell(r0 - w / 2, r0 + w / 2, npt, 2,
                              rstate=rstate) + mid
        pts[:,
            2:] = (rstate.uniform(size=(npt, ndim - 2)) * 2 - 1) * w / 2 + mid
        volume = w**(ndim - 2) * np.pi * ((r0 + w / 2)**2 - (r0 - w / 2)**2)
    elif typ == 'cylinder':
        w = 0.01
        r0 = 0.45
        a = 1
        pts = np.zeros((npt, ndim))
        pts[:, :2] = genshell(r0 - w / 2, r0 + w / 2, npt, 2,
                              rstate=rstate) + mid
        pts[:, 2:] = rstate.uniform(size=(npt, ndim - 2)) * a
        volume = np.pi * ((r0 + w / 2)**2 - (r0 - w / 2)**2)
    elif typ == 'shell':
        r1 = 0.45
        r2 = 0.46
        pts = genshell(r1, r2, npt, ndim, rstate=rstate) + mid
        volume = (np.pi**(ndim / 2) / scipy.special.gamma(ndim / 2 + 1) *
                  (r2**ndim - r1**ndim))
    else:
        raise RuntimeError('unknown', typ)
    return pts, volume


def plotter(ndim, bound, bootstrap=5, seed=None, ngrid=40):
    """ Plot missing volume for a given ndim
    As a function of bound topology and nlive
    """
    if seed is None:
        seed = 1
        rstate_data = np.random.default_rng(seed)
        rstate_dyn = np.random.default_rng(seed * 100 + 1)
    minnlive = 2 * ndim + 1
    maxnlive = 1000
    nlives = np.unique((10**np.linspace(np.log10(minnlive), np.log10(maxnlive),
                                        ngrid)).astype(int))
    plt.clf()
    objs = ['ball', 'pin', 'shell', 'torus', 'cylinder']
    nobs = len(objs)
    ny = int(math.floor(np.sqrt(nobs)))
    nx = int(math.ceil(nobs * 1. / ny))
    for i in range(nobs):
        plt.subplot(ny, nx, 1 + i)
        curo = objs[i]
        fracs = np.array([
            1 - doit(_,
                     curo,
                     ndim,
                     bound,
                     bootstrap=bootstrap,
                     rstate_data=rstate_data,
                     rstate_dyn=rstate_dyn)[1] for _ in nlives
        ])
        plt.semilogx(nlives, fracs, '.')
        plt.xlabel('Nlive')
        if i == 0:
            postf = '%d-D bound %s' % (ndim, bound)
        else:
            postf = ''
        plt.title(curo + ' ' + postf)
        plt.ylabel('missing fraction')
        plt.ylim(0, 1)
        plt.xlim(10, 1000)
    plt.tight_layout()


def doit(nlive,
         typ,
         ndim,
         bound='ell',
         bootstrap=None,
         rstate_data=None,
         rstate_dyn=None):
    """ Simulate  10 times nlive points with a given topology
    Then use nlive points to get the ellipsoid or multiellipsoid representation
    and then return the tuple with
    1) volume fraction. ie dynesty volume estimate over true volume
    2) point fraction. I.e fraction of unused points falling outside the
    dynesty bound (it should be zero in theory
    """
    oversample = 10  # simulate that many more points
    totpt = oversample * nlive  # simulate more points
    pts, volume = gen_data(totpt, typ, ndim, rstate=rstate_data)
    assert ((pts.min() > 0) and (pts.max() < 1))  # inside cube
    fitpts = pts[:nlive]
    testpts = pts[nlive:]
    logvol_ell, fracin, nell = computer(fitpts,
                                        testpts,
                                        bound=bound,
                                        rstate=rstate_dyn,
                                        bootstrap=bootstrap)
    return np.exp(logvol_ell) / volume, fracin, fitpts, testpts, nell


def computer(fitpts, testpts, bound='multi', bootstrap=None, rstate=None):
    """ Compute logvolume and fraction of points covered
    given actual live points (fitpts) and test points (testpts)"""
    # ndim = fitpts.shape[-1]
    cent = fitpts.mean(axis=0)
    cov = np.cov(fitpts.T)  # ndim)
    curb = db.Ellipsoid(cent, cov)  # pts)
    curb.update(fitpts, rstate=rstate, bootstrap=bootstrap)
    if bound == 'multi':
        curb = db.MultiEllipsoid([curb])
        curb.update(fitpts, rstate=rstate, bootstrap=bootstrap)
    if bound not in ['single', 'multi']:
        raise RuntimeError('unknown bound', bound)
    frac = np.array([curb.contains(_) for _ in testpts]).sum() / len(testpts)
    if bound == 'single':
        nell = 1
        logvol = curb.logvol
    else:
        nell = len(curb.ells)
        logvol = curb.logvol_tot
    return logvol, frac, nell
