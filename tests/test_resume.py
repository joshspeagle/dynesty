import dynesty
import numpy as np
import multiprocessing as mp
import os
import time
import pytest
from utils import get_rstate
import itertools
import multiprocessing as mp


def like(x):
    return -.5 * np.sum(x**2)


NLIVE = 300


def get_fname():
    pid = os.getpid()
    t = time.time()
    fname = 'test_%d_%f.pkl' % (pid, t)
    return fname


def ptform(x):
    return 20 * x - 10


def interrupter(pid, dt):
    """ This is to kill a process after some time dt """
    time.sleep(dt)
    os.kill(pid, 2)  # SIGINT


def start_interrupter(pid, dt):
    """ Start a killer process """
    pp = mp.Process(target=interrupter, args=(pid, dt))
    pp.start()
    return pp


class NullContextManager(object):
    # https://stackoverflow.com/questions/45187286/how-do-i-write-a-null-no-op-contextmanager-in-python
    # this is to make it work for 3.6
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def fit_main(fname, dynamic, checkpoint_every=0.01, npool=None):
    """
    Fit while checkpointing
    """
    ndim = 2
    with (NullContextManager() if npool is None else mp.Pool(npool)) as pool:
        queue_size = npool
        if dynamic:
            dns = dynesty.DynamicNestedSampler(like,
                                               ptform,
                                               ndim,
                                               nlive=NLIVE,
                                               rstate=get_rstate(),
                                               pool=pool,
                                               queue_size=queue_size)
        else:
            dns = dynesty.NestedSampler(like,
                                        ptform,
                                        ndim,
                                        nlive=NLIVE,
                                        rstate=get_rstate(),
                                        pool=pool,
                                        queue_size=queue_size)

        dns.run_nested(checkpoint_file=fname,
                       checkpoint_every=checkpoint_every,
                       n_effective=1000)  # .2
    return dns


def fit_resume(fname, dynamic, prev_logz, pool=None):
    """
    Resume and finish the fit as well as compare the logz to 
    previously computed logz
    """
    if dynamic:
        dns = dynesty.DynamicNestedSampler.restore(fname, pool=pool)
    else:
        dns = dynesty.NestedSampler.restore(fname, pool=pool)
    print('resuming')
    dns.run_nested(resume=True)
    # verify that the logz value is *identical*
    if prev_logz is not None:
        assert dns.results.logz[-1] == prev_logz


class cache:
    dt0 = None
    dt1 = None
    res0 = None
    res1 = None


def getlogz(save_every):
    """ Compute the execution time of static/dynamic runs as well
    logz value """

    if cache.dt0 is None:
        t0 = time.time()
        result0 = fit_main(None, False, save_every).results.logz[-1]
        t1 = time.time()
        result1 = fit_main(None, True, save_every).results.logz[-1]
        t2 = time.time()
        (cache.dt0, cache.dt1, cache.res0, cache.res1) = (t1 - t0, t2 - t1,
                                                          result0, result1)
    return cache.dt0, cache.dt1, cache.res0, cache.res1


@pytest.mark.parametrize("dynamic,delay_frac,with_pool",
                         itertools.chain(
                             itertools.product([False, True],
                                               [.1, .5, .75, .9], [False]),
                             itertools.product([False, True], [.5], [True])))
@pytest.mark.xdist_group(name="resume_group")
def test_resume(dynamic, delay_frac, with_pool):
    """
    Test we can interrupt and resume nested runs
    Note that I used xdist_group here in order to guarantee that if all the
    tests are run in parallel, this one is executed in one thread because
    I want to only use one getlogz() call.
    """
    fname = get_fname()
    save_every = 0.1
    dt_static, dt_dynamic, res_static, res_dynamic = getlogz(save_every)
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
    curdt *= delay_frac
    try:
        fit_proc = mp.Process(target=fit_main,
                              args=(fname, dynamic, save_every, npool))
        fit_proc.start()
        fit_pid = fit_proc.pid
        interrupt_proc = start_interrupter(fit_pid, curdt)
        time.sleep(curdt + 1)
        interrupt_proc.join()
        fit_proc.join()
        if npool is not None:
            # in the case of pooled run do not compare
            # as I am comparing with single threaded version
            curres = None
        with (NullContextManager()
              if npool is None else mp.Pool(npool)) as pool:
            fit_resume(fname, dynamic, curres, pool=pool)
    finally:
        try:
            os.unlink(fname)
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
