import numpy as np
import scipy.integrate
import dynesty.sampling as ds
from utils import get_rstate


def diamond_logl(X):
    x, y = X
    x1 = np.abs(x - 0.5)
    y1 = np.abs(y - 0.5)
    if X.min() < 0 or X.max() > 1:
        return -np.inf
    D2 = (x1 - 0.5)**2 + (y1 - 0.5)**2
    return np.where(D2 > 0.5**2, D2 - 0.5**2, -np.inf)


def checker_logl(X):
    mult = 16 * 2 * np.pi
    x, y = X
    logl = np.sin(x * mult) * np.sin(y * mult)
    return np.where((x >= 0) & (x <= 1) & (y >= 0) & (y < 1), logl, -np.inf)


def pdf_test(func, curx, nbins=100, thresh=6):
    hh, loc = np.histogram(curx, range=[0, 1], bins=nbins)
    norm = (loc[1] - loc[0]) * len(curx)
    pdf = hh / norm
    model_pdf = np.array([
        scipy.integrate.quad(func, l, l + loc[1] - loc[0])[0] /
        (loc[1] - loc[0]) for l in loc[:-1]
    ])  # (func(loc[:-1]) + func(loc[1:])) / 2.
    frac = 0.01 * model_pdf.max()
    epdf = (model_pdf * norm)**.5 / norm
    epdf1 = hh**.5 / norm
    margin = np.maximum(thresh * np.maximum(epdf, epdf1), frac)
    rat = np.abs(model_pdf - pdf)
    assert (rat / margin).max() < 1


def diamond_test(X):

    def func(x):
        return (1 - 2 * np.sqrt(np.abs(x - 0.5) -
                                (x - 0.5)**2)) / (1 - np.pi / 4)

    for i in range(2):
        curx = X[:, i]
        pdf_test(func, curx)


def checker_test(X, thresh=6):

    def func(x):
        return x * 0 + 1

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
        'hslice': ds.sample_hslice,
        'slice': ds.sample_slice,
        'rwalk': ds.sample_rwalk
    }[sample]

    eye2 = np.eye(2)

    def TRANS(x):
        return x

    for i in range(niter):
        seed = rng.integers(1e9)
        args = ds.SamplerArgument(u, loglstar, eye2, scale, TRANS, curlogl,
                                  seed, kwargs)
        u = func(args)[0]
        us[i] = u
    return us


def test_diamond_rwalk():
    rs = get_rstate()
    us = doit(model='diamond',
              sample='rwalk',
              scale=.3,
              rstate=rs,
              niter=100_000,
              walks=10)
    diamond_test(us)


def test_diamond_rslice():
    rs = get_rstate()
    us = doit(model='diamond',
              sample='rslice',
              scale=.1,
              rstate=rs,
              slices=10,
              niter=100_000)
    diamond_test(us)


def test_diamond_rslice_double():
    rs = get_rstate()
    us = doit(model='diamond',
              sample='rslice',
              scale=.001,
              rstate=rs,
              niter=100_000,
              doubling=True)
    us = us[::10, :]  # thinning
    diamond_test(us)


def test_checkerboard_rslice():
    rs = get_rstate()
    us = doit(model='diamond',
              sample='slice',
              scale=.3,
              rstate=rs,
              slices=1,
              niter=100_000)
    diamond_test(us)


def test_checkerboard_rslice_double():
    rs = get_rstate()
    us = doit(model='checkerboard',
              sample='rslice',
              scale=.001,
              rstate=rs,
              niter=100_000,
              doubling=True)
    us = us[::10, :]  # thinning
    checker_test(us)
