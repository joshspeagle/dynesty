import numpy as np
import dynesty.sampling as ds
from utils import get_rstate


def diamond_logl(X):
    x, y = X
    x1 = np.abs(x - 0.5)
    y1 = np.abs(y - 0.5)
    if X.min() < 0 or X.max() > 1:
        return -np.inf
    D2 = (x1 - 0.5)**2 + (y1 - 0.5)**2
    if D2 > 0.5**2:
        return (D2 - 0.5**2) / (0.5 - 0.5**2)
    else:
        return -np.inf


def checker_logl(X):
    mult = 16 * 2 * np.pi
    x, y = X
    logl = np.sin(x * mult) * np.sin(y * mult)
    if X.min() < 0 or X.max() > 1:
        return -np.inf
    return logl


def pdf_test(func, curx, nbins=100, thresh=6):
    hh, loc = np.histogram(curx, range=[0, 1], bins=nbins)
    xloc = loc[:-1] + .5 * np.diff(loc)
    pdf = hh / (loc[1] - loc[0]) / len(curx)
    epdf = np.maximum(hh, 1)**.5 / (loc[1] - loc[0]) / len(curx)

    rat = np.abs(func(xloc) - pdf) / epdf
    assert rat.max() < thresh


def diamond_test(X):

    def func(x):
        return (1 - 2 * np.sqrt(np.abs(x - 0.5) -
                                (x - 0.5)**2)) / (1 - np.pi / 4)

    for i in range(2):
        curx = X[:, i]
        pdf_test(func, curx)


def checker_test(X, thresh=6):

    def func(x):
        return 1

    for i in range(2):
        curx = X[:, i]
        pdf_test(func, curx, thresh=thresh)


def doit(model='diamond',
         sample='rslice',
         scale=1,
         rstate=None,
         niter=100_000,
         doubling=False,
         slices=1,
         walks=1):
    loglstar = 0.
    u = np.r_[.5, .5]
    kwargs = {'slices': slices, 'walks': walks, 'slice_doubling': doubling}
    if rstate is not None:
        rng = rstate
    else:
        rng = np.random.default_rng(1)
    us = np.zeros((niter, 2))
    if model == 'diamond':
        curlogl = diamond_logl
    elif model == 'checkerboard':
        curlogl = checker_logl
    else:
        raise Exception('unknown')
    func = {
        'rslice': ds.sample_rslice,
        'slice': ds.sample_slice,
        'rwalk': ds.sample_rwalk
    }[sample]

    for i in range(niter):
        seed = rng.integers(1e9)
        args = (u, loglstar, np.eye(2), scale, lambda x: x, curlogl, seed,
                kwargs)
        u = func(args)[0]
        us[i] = u
    return us


def test_all():
    rs = get_rstate()
    us = doit(model='diamond',
              sample='rwalk',
              scale=.3,
              rstate=rs,
              niter=100_000,
              walks=10)
    diamond_test(us)

    us = doit(model='diamond',
              sample='rslice',
              scale=.3,
              rstate=rs,
              slices=10,
              niter=100_000)
    diamond_test(us)

    us = doit(model='diamond',
              sample='rslice',
              scale=.3,
              rstate=rs,
              niter=100_000,
              doubling=True)
    diamond_test(us)

    us = doit(model='checkerboard',
              sample='rslice',
              scale=.001,
              rstate=rs,
              niter=100_000,
              doubling=True)
    checker_test(us)

    us = doit(model='diamond',
              sample='slice',
              scale=.3,
              rstate=rs,
              slices=1,
              niter=100_000)
    diamond_test(us)
