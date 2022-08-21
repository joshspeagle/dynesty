import dynesty
import dynesty.dynamicsampler as dds
import numpy as np
import multiprocessing as mp
import os
import time
import pytest
from utils import get_rstate


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


def fit():
    ndim = 2
    dns = dynesty.DynamicNestedSampler(like,
                                       ptform,
                                       ndim,
                                       nlive=100,
                                       rstate=get_rstate())
    fname = 'xx%d.pkl' % (os.getpid())
    dns.run_nested(checkpoint_file=fname)


def fit_resume(pid):
    fname = 'xx%d.pkl' % pid
    dns = dds.DynamicSampler.restore(fname)
    print('resuming')
    dns.run_nested(resume=True)
    os


@pytest.mark.parametrize("delay", [5, 15])
def test_resume(delay):
    pp = mp.Process(target=fit)
    pp.start()
    fit_pid = pp.pid
    pp = start_interrupter(fit_pid, delay)
    time.sleep(delay + 1)
    fit_resume(fit_pid)
    fname = 'xx%d.pkl' % fit_pid
    os.unlink(fname)
