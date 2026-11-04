import numpy as np
import dynesty
import os
import h5py
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


def _history(fname):
    with h5py.File(fname, 'r') as fp:
        return fp['evaluation_logl'][:]


def _sampler(fname, sampler_type=dynesty.NestedSampler):
    return sampler_type(loglike,
                        prior_transform,
                        2,
                        nlive=50,
                        save_evaluation_history=True,
                        history_filename=fname,
                        rstate=get_rstate())


@pytest.mark.parametrize('sampler_type',
                         [dynesty.NestedSampler, dynesty.DynamicNestedSampler])
def test_saving_resume(sampler_type, tmp_path):
    # a run resumed from a checkpoint must save the same history
    # as an uninterrupted run
    dynamic = sampler_type is dynesty.DynamicNestedSampler
    ckpt = str(tmp_path / 'ckpt.pkl')
    hist = []
    for restore in [False, True]:
        fname = str(tmp_path / f'hist{restore}.h5')
        sampler = _sampler(fname, sampler_type)
        if dynamic:
            sampler.run_nested(maxbatch=1,
                               checkpoint_file=ckpt,
                               print_progress=printing)
        else:
            sampler.run_nested(maxiter=300,
                               add_live=False,
                               checkpoint_file=ckpt,
                               print_progress=printing)
        if restore:
            sampler = sampler_type.restore(ckpt)
        if dynamic:
            sampler.add_batch(print_progress=printing)
        else:
            sampler.run_nested(resume=True, print_progress=printing)
        hist.append(_history(fname))
    assert len(hist[0]) == sampler.ncall
    assert np.array_equal(hist[0], hist[1])


@pytest.mark.parametrize('reuse', [False, True])
def test_saving_resume_lost_file(reuse, tmp_path):
    # if the history file was removed or overwritten by another run,
    # the restored run must warn and not touch the file
    fname, ckpt = str(tmp_path / 'hist.h5'), str(tmp_path / 'ckpt.pkl')
    sampler = _sampler(fname)
    sampler.run_nested(maxiter=300, add_live=False, print_progress=printing)
    sampler.save(ckpt)
    if reuse:
        _sampler(fname).run_nested(maxiter=1000, print_progress=printing)
        ref = _history(fname)
    else:
        os.unlink(fname)
    with pytest.warns(UserWarning, match='history file'):
        sampler = dynesty.NestedSampler.restore(ckpt)
    sampler.run_nested(print_progress=printing)
    if reuse:
        assert np.array_equal(_history(fname), ref)
    else:
        assert not os.path.exists(fname)
