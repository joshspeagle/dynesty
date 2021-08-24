import numpy as np
import dynesty
import os
import multiprocessing as mp
import pytest
from utils import get_rstate
"""
Run a series of basic tests to check whether saving likelikelihood evals
are broken

"""

nlive = 100
printing = False

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


@pytest.mark.parametrize('dopool', [False, True])
def test_saving(dopool):
    # test saving
    ndim = 2
    fname = 'dynesty_test_%d.h5' % (os.getpid())
    rstate = get_rstate()
    kw = {}
    if dopool:
        pool = mp.Pool(2)
        kw['pool'] = pool
        kw['queue_size'] = 2

    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    save_history=True,
                                    history_filename=fname,
                                    rstate=rstate,
                                    **kw)
    sampler.run_nested(dlogz=1, print_progress=printing, maxiter=300)
    assert (os.path.exists(fname))
    try:
        os.unlink(fname)
    except FileNotFoundError:
        pass
    if dopool:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
