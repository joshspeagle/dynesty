import dynesty
import numpy as np
import multiprocessing as mp
import os
import time
import pytest
from utils import get_rstate
import itertools


def like(x):
    return -.5 * np.sum(x**2)


def ptform(x):
    return 20 * x - 10


def interrupter(pid, dt):
    time.sleep(dt)
    os.kill(pid, 9)


def start_interrupter(pid, dt):
    pp = mp.Process(target=interrupter, args=(pid, dt))
    pp.start()
    return pp


def fit(fname, dynamic):
    ndim = 2
    if dynamic:
        dns = dynesty.DynamicNestedSampler(like,
                                           ptform,
                                           ndim,
                                           nlive=100,
                                           rstate=get_rstate())
    else:
        dns = dynesty.NestedSampler(like,
                                    ptform,
                                    ndim,
                                    nlive=100,
                                    rstate=get_rstate())
    dns.run_nested(checkpoint_file=fname)


def fit_resume(fname, dynamic):
    if dynamic:
        dns = dynesty.DynamicNestedSampler.restore(fname)
    else:
        dns = dynesty.NestedSampler.restore(fname)
    print('resuming')
    dns.run_nested(resume=True)


@pytest.mark.parametrize("dynamic,delay",
                         itertools.product([False, True], [1, 5, 10, 15]))
def test_resume(dynamic, delay):
    pid = os.getpid()
    fname = 'xx%d.pkl' % (pid)
    pp = mp.Process(target=fit, args=(fname, dynamic))
    pp.start()
    fit_pid = pp.pid
    pp = start_interrupter(fit_pid, delay)
    time.sleep(delay + 1)
    fit_resume(fname, dynamic)
    os.unlink(fname)
