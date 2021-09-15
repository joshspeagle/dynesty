import numpy as np
import dynesty
import os
import multiprocessing as mp
import pytest
from utils import get_rstate, get_printing
"""
Run a series of basic tests to check whether saving likelikelihood evals
are broken

"""

nlive = 100
printing = get_printing()

# EGGBOX


# see 1306.2144
def loglike(x):
    return -.5 * np.sum(x**2)


def prior_transform(x):
    return 20 * x - 10


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

    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    save_history=True,
                                    history_filename=fname,
                                    rstate=rstate,
                                    **kw)
    sampler.run_nested(dlogz=1, print_progress=printing, maxiter=3000)
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
