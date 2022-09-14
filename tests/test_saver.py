import numpy as np
import dynesty
import os
import multiprocessing as mp
import pytest
from utils import get_rstate, get_printing, NullContextManager
"""
Run a series of basic tests to check whether saving likelikelihood evals
are broken

"""

nlive = 500
printing = get_printing()

# EGGBOX


# see 1306.2144
def loglike(x):
    return -.5 * np.sum(x**2)


def prior_transform(x):
    return 20 * x - 10


@pytest.mark.parametrize('dopool,maxiter', [(False, 1000), (True, 1000),
                                            (False, 11000)])
def test_saving(dopool, maxiter):
    # test saving
    ndim = 2
    fname = 'dynesty_test_%d.h5' % (os.getpid())
    rstate = get_rstate()
    kw = {}

    with (NullContextManager() if not dopool else mp.Pool(2)) as pool:
        if dopool:
            pool = pool
            kw['pool'] = pool
            kw['queue_size'] = 100

        sampler = dynesty.NestedSampler(loglike,
                                        prior_transform,
                                        ndim,
                                        nlive=nlive,
                                        save_history=True,
                                        history_filename=fname,
                                        rstate=rstate,
                                        **kw)
        sampler.run_nested(dlogz=0.01,
                           print_progress=printing,
                           maxiter=maxiter)
        assert (os.path.exists(fname))
        try:
            os.unlink(fname)
        except FileNotFoundError:
            pass
