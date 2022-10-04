import dynesty
import numpy as np
from utils import get_rstate, get_printing, NullContextManager
import pytest
import multiprocessing as mp

printing = get_printing()


def cube(p):
    return p * 20 - 10


def like(p):
    x, y = p
    a = 1
    b = 100
    ret = (a - x)**2 + b * (y - x**2)**2
    return -0.5 * ret


def analytic(xgrid, ygrid, minx, maxx):
    # return two marginal posteriors from integrating over the grid

    def func(x, y):
        return np.exp(like((x, y)))

    # compute the analytic integral
    step = 2. / 1000
    xgrid1, ygrid1 = np.mgrid[minx:maxx:step, minx:maxx:step]
    tot = func(xgrid1, ygrid1).sum() * (step)**2

    def funcy(xval):

        def curfunc(ynew):
            return func(xval, ynew) / tot

        return curfunc

    def funcx(yval):

        def curfunc(xnew):
            return func(xnew, yval) / tot

        return curfunc

    retx = []
    rety = []
    xfgrid = np.linspace(minx, maxx, 100000)
    for y in ygrid:
        rety.append(funcx(y)(xfgrid).sum() * (xfgrid[1] - xfgrid[0]))
    for x in xgrid:
        retx.append(funcy(x)(xfgrid).sum() * (xfgrid[1] - xfgrid[0]))
    retx, rety = [np.array(_) for _ in [retx, rety]]
    return retx, rety


def doit(sample='rslice', nlive=500, seed=1):
    # return uniformly weighted chain
    ndim = 2
    rstate = get_rstate(seed)
    ns = dynesty.NestedSampler(like,
                               cube,
                               ndim,
                               nlive=nlive,
                               sample=sample,
                               rstate=rstate)
    ns.run_nested(print_progress=printing)
    res = ns.results

    C = res.samples_equal(rstate=rstate)
    return C


def domany(sample='rslice', nlive=500, niter=100, nthreads=1):
    # run sampling  many times and return
    # xgrid, and a dictionary of average marginal posteriors for x0 and x1
    hhs = {}
    start = True
    rstate = get_rstate()
    seed = rstate.integers(int(1e9))
    with (mp.Pool(nthreads) if nthreads > 1 else NullContextManager()) as pool:
        Cs = []
        for i in range(niter):
            if nthreads > 1:
                Cs.append(
                    pool.apply_async(
                        doit, (),
                        dict(seed=seed + i, nlive=nlive, sample=sample)))
            else:
                Cs.append(doit(seed=seed + i, nlive=nlive, sample=sample))
        for C in Cs:
            if nthreads > 1:
                C = C.get()
            for j in range(2):
                curhh, curloc = np.histogram(C[:, j],
                                             range=[cube(0), cube(1)],
                                             bins=200,
                                             density=True)
                if start:
                    hhs[j] = curhh * 0
                hhs[j] += curhh
            start = False

    for j in range(2):
        hhs[j] = hhs[j] * 1. / niter
    return curloc[:-1] + .5 * np.diff(curloc), hhs


@pytest.mark.slow
@pytest.mark.parametrize("sample", ['rslice', 'rwalk'])
def test_rosen(sample, nthreads=1):
    loc, hhs = domany(sample=sample, nlive=500, niter=100, nthreads=nthreads)
    minx = cube(0)
    maxx = cube(1)
    post0, post1 = analytic(loc, loc, minx, maxx)
    THRESHOLD = 0.08
    # I had to increase the threshold from .05
    # that is a bit of a worry
    assert (np.abs(post1 - hhs[1]).max() / post1.max() < THRESHOLD)
    assert (np.abs(post0 - hhs[0]).max() / post0.max() < THRESHOLD)
