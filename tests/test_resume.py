import inspect
import itertools
import os
import sys
import time
import multiprocessing as mp
import dynesty
import numpy as np
import pytest
from utils import get_rstate, NullContextManager, get_printing
import dynesty.pool

printing = get_printing()


def like(x):
    blob = np.zeros(1, dtype=int)
    # I'm returning the blob to be able to
    # check that the function was executed in different threads
    blob[0] = os.getpid()
    return -.5 * np.sum(x**2), blob


NLIVE = 100
NEFF0 = 1000


def get_fname(pref='test_'):
    pid = os.getpid()
    t = time.time()
    fname = 'test_%s_%d_%f.pkl' % (pref, pid, t)
    return fname


def ptform(x):
    return 20 * x - 10


def fit_main(fname,
             dynamic,
             checkpoint_every=0.01,
             npool=None,
             dyn_pool=False,
             neff=NEFF0,
             ready_file=None):
    """
    Fit while checkpointing
    """
    # Signal that we're ready (for spawn context timing)
    if ready_file is not None:
        with open(ready_file, 'w') as f:
            f.write(str(time.time()))

    ndim = 2
    with (NullContextManager() if npool is None else
          (dynesty.pool.Pool(npool, like, ptform)
           if dyn_pool else mp.get_context('spawn').Pool(npool))) as pool:
        queue_size = 100 if npool is not None else None
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
                                               queue_size=queue_size,
                                               blob=True)
            kw = dict(n_effective=NEFF0)
        else:
            dns = dynesty.NestedSampler(curlike,
                                        curpt,
                                        ndim,
                                        nlive=NLIVE,
                                        rstate=get_rstate(),
                                        pool=pool,
                                        queue_size=queue_size,
                                        blob=True)
            kw = dict()
        dns.run_nested(checkpoint_file=fname,
                       print_progress=printing,
                       checkpoint_every=checkpoint_every,
                       **kw)
    return dns


def fit_resume(fname, dynamic, prev_logz, pool=None, neff=NEFF0):
    """
    Resume and finish the fit as well as compare the logz to
    previously computed logz
    """
    if dynamic:
        dns = dynesty.DynamicNestedSampler.restore(fname, pool=pool)
    else:
        dns = dynesty.NestedSampler.restore(fname, pool=pool)
        neff = None
    print('resuming', file=sys.stderr)
    if dynamic:
        kw = dict(n_effective=neff)
    else:
        kw = dict()
    ncall0 = dns.ncall
    dns.run_nested(resume=True, print_progress=printing, **kw)
    # verify that the logz value is *identical*
    if prev_logz is not None:
        assert dns.results['logz'][-1] == prev_logz
    # the second value tells whether any likelihood calls were made
    # after resuming (the checkpoint could have been the final one)
    return dns.results['blob'], dns.ncall > ncall0


def run_interrupted(fname, dynamic, save_every, npool, dyn_pool, delay):
    """
    Run the fit in a separate process and kill it after delay seconds
    since the start of the fit, but not before the first checkpoint
    is written.
    Returns True if the fit was interrupted and False if it finished
    before we killed it.
    """
    ready_file = fname + '.ready'
    try:
        # Always use spawn context to match actual usage
        fit_proc = mp.get_context('spawn').Process(
            target=fit_main,
            args=(fname, dynamic, save_every, npool, dyn_pool, NEFF0,
                  ready_file))
        start_time = time.time()
        fit_proc.start()

        # Wait for spawn process to be ready before starting timer
        # as the process startup is not part of the cached timing
        while not os.path.exists(ready_file):
            time.sleep(0.01)
            if time.time() - start_time > 5:  # Safety timeout
                raise RuntimeError("Process failed to start")
        fit_proc.join(delay)
        # Make sure that we have something to resume from
        while fit_proc.is_alive() and not os.path.exists(fname):
            time.sleep(0.01)
        if not fit_proc.is_alive():
            assert fit_proc.exitcode == 0
            return False
        print('terminating', file=sys.stderr)
        fit_proc.terminate()
        fit_proc.join()
        return True
    finally:
        try:
            os.unlink(ready_file)
        except:  # noqa
            pass


class cache:
    dt = None
    logz = None


def getlogz(fname, save_every):
    """ Compute the execution time of static/dynamic runs as well
    logz value """

    if cache.dt is None:
        cache.dt = {}
        cache.logz = {}
        print('caching', file=sys.stderr)
        for dynamic, with_pool in itertools.product([False, True],
                                                    [False, True]):
            t0 = time.time()
            if with_pool:
                npool = 2
            else:
                npool = None
            curlogz = fit_main(fname, dynamic, save_every,
                               npool=npool).results['logz'][-1]
            try:
                os.unlink(fname)
            except:  # noqa
                pass
            t1 = time.time()
            cache.logz[dynamic, with_pool] = curlogz
            cache.dt[dynamic, with_pool] = t1 - t0
            print(f'done {dynamic} {with_pool}', file=sys.stderr)
        print('done caching', file=sys.stderr)
    return cache.dt, cache.logz


@pytest.mark.parametrize(
    "dynamic,delay_frac,with_pool,dyn_pool",
    itertools.chain(
        itertools.product([False, True], [.2, .5, .75, .9], [False], [False]),
        itertools.product([False, True], [.5], [True], [False]),
        [[True, .5, True, True]]))
@pytest.mark.xdist_group(name="resume_group")
def test_resume(dynamic, delay_frac, with_pool, dyn_pool):
    """
    Test we can interrupt and resume nested runs
    Note that I used xdist_group here in order to guarantee that if all the
    tests are run in parallel, this one is executed in one thread because
    I want to only use one getlogz() call.
    """
    fname = get_fname(inspect.currentframe().f_code.co_name)

    save_every = 1
    cache_dt, cache_logz = getlogz(fname, save_every)
    if with_pool:
        npool = 2
    else:
        npool = None
    curdt, curlogz = [_[dynamic, with_pool] for _ in [cache_dt, cache_logz]]
    save_every = min(save_every, curdt / 10)
    curdt *= delay_frac

    try:
        if not run_interrupted(fname, dynamic, save_every, npool, dyn_pool,
                               curdt):
            # the fit finished before we could interrupt it
            return
        with (NullContextManager() if npool is None else
              (dynesty.pool.Pool(npool, like, ptform) if dyn_pool else
               mp.get_context('spawn').Pool(npool))) as pool:
            blob, resumed = fit_resume(fname, dynamic, curlogz, pool=pool)
        nproc = 1 if npool is None else npool
        npids = len(np.unique(blob))
        if npool is not None:
            # with the pool nothing should be evaluated in this process,
            # so any extra pids after resuming must be the new pool workers
            assert os.getpid() not in blob
        if resumed:
            # we ran in nproc pids before and in new pids after
            # (not necessarily all workers of the pool get used)
            assert nproc < npids <= 2 * nproc
        else:
            # the interruption happened after the final checkpoint
            # so there was nothing left to sample
            assert npids <= nproc
    finally:
        try:
            os.unlink(fname)
        except:  # noqa
            pass
        try:
            os.unlink(fname + '.tmp')
        except:  # noqa
            pass


@pytest.mark.parametrize(
    "delay_frac",
    [.2, .5, .75, .9],
)
@pytest.mark.xdist_group(name="resume_group")
def test_resume_queue_size(delay_frac):
    """
    Test we can interrupt and resume nested runs
    Note that I used xdist_group here in order to guarantee that if all the
    tests are run in parallel, this one is executed in one thread because
    I want to only use one getlogz() call.
    """
    fname = get_fname(inspect.currentframe().f_code.co_name)

    save_every = 1
    cache_dt, cache_logz = getlogz(fname, save_every)
    npool = 2
    curdt, curlogz = [_[False, True] for _ in [cache_dt, cache_logz]]
    save_every = min(save_every, curdt / 10)
    curdt *= delay_frac

    try:
        if not run_interrupted(fname, False, save_every, npool, False,
                               curdt):
            # the fit finished before we could interrupt it
            return
        with mp.get_context('spawn').Pool(npool + 1) as pool:
            fit_resume(fname, False, curlogz, pool=pool)
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
        fname = get_fname(inspect.currentframe().f_code.co_name)
        fit_main(fname, dynamic, 1)
    finally:
        try:
            os.unlink(fname)
        except:  # noqa
            pass


def test_resume_finished():
    """
    Here i exercise the warning when I tried to resume a fully finished
    dynamic run
    """
    fname = get_fname(inspect.currentframe().f_code.co_name)
    try:
        fit_main(fname, True, .1, neff=1000)
        fit_resume(fname, True, None, neff=1000)
    finally:
        try:
            os.unlink(fname)
        except:  # noqa
            pass
