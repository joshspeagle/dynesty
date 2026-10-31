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

    with (NullContextManager()
          if not dopool else mp.get_context('spawn').Pool(2)) as pool:
        if dopool:
            pool = pool
            kw['pool'] = pool
            kw['queue_size'] = 100

        sampler = dynesty.NestedSampler(loglike,
                                        prior_transform,
                                        ndim,
                                        nlive=nlive,
                                        save_evaluation_history=True,
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


class Interrupt(Exception):
    pass


# number of likelihood calls left before the run is interrupted
# (this is a global, so that it is not stored in the checkpoint)
NCALL_LEFT = [None]


def loglike_interrupted(x):
    if NCALL_LEFT[0] is not None:
        NCALL_LEFT[0] -= 1
        if NCALL_LEFT[0] < 0:
            raise Interrupt()
    return loglike(x)


@pytest.mark.parametrize('dynamic,remove_history', [(False, False),
                                                    (True, False),
                                                    (False, True)])
def test_saving_resume(dynamic, remove_history, tmp_path):
    # the history must continue to be saved after restoring
    # an interrupted run from the checkpoint
    h5py = pytest.importorskip('h5py')
    ndim = 2
    fname = str(tmp_path / 'history.h5')
    checkpoint = str(tmp_path / 'checkpoint.pkl')
    if dynamic:
        cls = dynesty.DynamicNestedSampler
        kw = dict(n_effective=2000)
    else:
        cls = dynesty.NestedSampler
        kw = {}
    sampler = cls(loglike_interrupted,
                  prior_transform,
                  ndim,
                  nlive=50,
                  save_evaluation_history=True,
                  history_filename=fname,
                  rstate=get_rstate())
    NCALL_LEFT[0] = 1000
    try:
        with pytest.raises(Interrupt):
            sampler.run_nested(print_progress=printing,
                               checkpoint_file=checkpoint,
                               checkpoint_every=0,
                               **kw)
    finally:
        NCALL_LEFT[0] = None
    sampler = cls.restore(checkpoint)
    ncall0 = sampler.ncall
    if remove_history:
        # if the history file is gone, we just warn
        os.unlink(fname)
        with pytest.warns(UserWarning, match='Failed to save history'):
            sampler.run_nested(resume=True, print_progress=printing, **kw)
        return
    sampler.run_nested(resume=True, print_progress=printing, **kw)
    assert sampler.ncall > ncall0
    with h5py.File(fname, 'r') as fp:
        assert len(fp['evaluation_logl']) == sampler.ncall
