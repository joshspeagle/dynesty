import os
import time
import sys
import multiprocessing as mp
import dynesty
import numpy as np
import pytest
from utils import get_rstate, NullContextManager
import itertools
import dynesty.pool


def like(x):
    return -.5 * np.sum(x**2)


NLIVE = 300
NEFF = 5000


def get_fname():
    pid = os.getpid()
    t = time.time()
    fname = 'test_%d_%f.pkl' % (pid, t)
    return fname


def ptform(x):
    return 20 * x - 10


def fit_main(fname,
             dynamic,
             checkpoint_every=0.01,
             npool=None,
             dyn_pool=False):
    """
    Fit while checkpointing
    """
    ndim = 2
    with (NullContextManager() if npool is None else (dynesty.pool.Pool(
            npool, like, ptform) if dyn_pool else mp.Pool(npool))) as pool:
        queue_size = npool
        if dyn_pool:
            curlike, curpt = pool.loglike, pool.prior_transform
        else:
            curlike, curpt = like, ptform
        if dynamic:
            dns = dynesty.DynamicNestedSampler(curlike,
                                               curpt,
                                               ndim,
                                               nlive=NLIVE,
                                               rstate=get_rstate(),
                                               pool=pool,
                                               queue_size=queue_size)
            neff = NEFF
        else:
            dns = dynesty.NestedSampler(curlike,
                                        curpt,
                                        ndim,
                                        nlive=NLIVE,
                                        rstate=get_rstate(),
                                        pool=pool,
                                        queue_size=queue_size)
            neff = None
        dns.run_nested(checkpoint_file=fname,
                       checkpoint_every=checkpoint_every,
                       n_effective=neff)
    return dns


def fit_resume(fname, dynamic, prev_logz, pool=None):
    """
    Resume and finish the fit as well as compare the logz to 
    previously computed logz
    """
    if dynamic:
        dns = dynesty.DynamicNestedSampler.restore(fname, pool=pool)
        neff = NEFF
    else:
        dns = dynesty.NestedSampler.restore(fname, pool=pool)
        neff = None
    print('resuming', file=sys.stderr)
    dns.run_nested(resume=True, n_effective=neff)
    # verify that the logz value is *identical*
    if prev_logz is not None:
        assert dns.results['logz'][-1] == prev_logz


class cache:
    dt0 = None
    dt1 = None
    res0 = None
    res1 = None


def getlogz(fname, save_every):
    """ Compute the execution time of static/dynamic runs as well
    logz value """

    if cache.dt0 is None:
        t0 = time.time()
        print('caching', file=sys.stderr)
        result0 = fit_main(fname, False, save_every).results['logz'][-1]
        try:
            os.unlink(fname)
        except:  # noqa
            pass
        t1 = time.time()
        print('static done', file=sys.stderr)
        result1 = fit_main(fname, True, save_every).results['logz'][-1]
        try:
            os.unlink(fname)
        except:  # noqa
            pass
        print('done caching', file=sys.stderr)
        t2 = time.time()
        (cache.dt0, cache.dt1, cache.res0, cache.res1) = (t1 - t0, t2 - t1,
                                                          result0, result1)
    return cache.dt0, cache.dt1, cache.res0, cache.res1


@pytest.mark.parametrize("dynamic,delay_frac,with_pool,dyn_pool",
                         itertools.chain(
                             itertools.product([False, True],
                                               [.2, .5, .75, .9], [False],
                                               [False]),
                             itertools.product([False, True], [.5], [True],
                                               [False]),
                             [[True, .5, True, True]]))
@pytest.mark.xdist_group(name="resume_group")
def test_resume(dynamic, delay_frac, with_pool, dyn_pool):
    """
    Test we can interrupt and resume nested runs
    Note that I used xdist_group here in order to guarantee that if all the
    tests are run in parallel, this one is executed in one thread because
    I want to only use one getlogz() call.
    """
    fname = get_fname()
    save_every = 1
    dt_static, dt_dynamic, res_static, res_dynamic = getlogz(fname, save_every)
    if with_pool:
        npool = 2
    else:
        npool = None
    if dynamic:
        curdt = dt_dynamic
        curres = res_dynamic
    else:
        curdt = dt_static
        curres = res_static

    save_every = min(save_every, curdt / 10)
    curdt *= delay_frac
    try:
        fit_proc = mp.Process(target=fit_main,
                              args=(fname, dynamic, save_every, npool,
                                    dyn_pool))
        fit_proc.start()
        res = fit_proc.join(curdt)
        if res is None:
            print('terminating', file=sys.stderr)
            fit_proc.terminate()
            if npool is not None:
                # in the case of pooled run do not compare
                # as I am comparing with single threaded version
                curres = None
            with (NullContextManager() if npool is None else
                  (dynesty.pool.Pool(npool, like, ptform)
                   if dyn_pool else mp.Pool(npool))) as pool:
                fit_resume(fname, dynamic, curres, pool=pool)
        else:
            assert res == 0
    finally:
        try:
            os.unlink(fname)
        except:  # noqa
            pass
        try:
            os.unlink(fname + '.tmp')
        except:  # noqa
            pass


@pytest.mark.parametrize("dynamic", [False, True])
def test_save(dynamic):
    """
    Here I test that I can actually save the 
    files (in  the previous test the saving is done in a separate thread)
    """
    try:
        fname = get_fname()
        fit_main(fname, dynamic, 1)
    finally:
        try:
            os.unlink(fname)
        except:  # noqa
            pass
