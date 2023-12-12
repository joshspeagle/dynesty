import numpy as np
import pytest
import dynesty
import multiprocessing as mp
import dynesty.pool as dypool
from utils import get_rstate, get_printing
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 1000
printing = get_printing()

ndim = 2
gau_s = 0.01


def loglike_gau(x):
    return (-0.5 * np.log(2 * np.pi) * ndim - np.log(gau_s) * ndim -
            0.5 * np.sum((x - 0.5)**2) / gau_s**2)


def prior_transform_gau(x):
    return x


# EGGBOX
# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


LOGZ_TRUTH_GAU = 0
LOGZ_TRUTH_EGG = 235.856


def terminator(pool):
    # Because of https://github.com/nedbat/coveragepy/issues/1310
    # I have to close join and can't fully rely on contexts that
    # do send SIGTERMS
    pool.close()
    pool.join()


def test_pool():
    # test pool on egg problem
    rstate = get_rstate()

    # i specify large queue_size here, otherwise it is too slow
    with dypool.Pool(2, loglike_egg, prior_transform_egg) as pool:
        sampler = dynesty.NestedSampler(pool.loglike,
                                        pool.prior_transform,
                                        ndim,
                                        nlive=nlive,
                                        pool=pool,
                                        queue_size=100,
                                        rstate=rstate)
        sampler.run_nested(dlogz=0.1, print_progress=printing)

        assert (abs(LOGZ_TRUTH_EGG - sampler.results['logz'][-1])
                < 5. * sampler.results['logzerr'][-1])
        terminator(pool)


def test_pool_x():
    # check without specifying queue_size
    rstate = get_rstate()

    with dypool.Pool(2, loglike_egg, prior_transform_egg) as pool:
        sampler = dynesty.NestedSampler(pool.loglike,
                                        pool.prior_transform,
                                        ndim,
                                        nlive=50,
                                        pool=pool,
                                        rstate=rstate)
        sampler.run_nested(print_progress=printing, maxiter=100)
        # not checking the results since I'm interrupting
        terminator(pool)


def test_pool_dynamic():
    # test pool on gau problem
    # i specify large queue_size here, otherwise it is too slow
    rstate = get_rstate()

    with dypool.Pool(2, loglike_gau, prior_transform_gau) as pool:
        sampler = dynesty.DynamicNestedSampler(pool.loglike,
                                               pool.prior_transform,
                                               ndim,
                                               nlive=nlive,
                                               pool=pool,
                                               queue_size=100,
                                               rstate=rstate)
        sampler.run_nested(dlogz_init=1, print_progress=printing)

        assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1])
                < 5. * sampler.results['logzerr'][-1])
        terminator(pool)


def loglike_gau_args(x, y, z=None, a=0, b=0):
    return (-0.5 * np.log(2 * np.pi) * ndim - np.log(gau_s) * ndim -
            0.5 * np.sum((x - 0.5)**2) / gau_s**2) + y + z + a + b


def prior_transform_gau_args(x, y, z=None, a=0, b=0):
    return x + y + z + a + b


def test_pool_args():
    # test pool on gau problem
    # i specify large queue_size here, otherwise it is too slow
    rstate = get_rstate()

    with dypool.Pool(2,
                     loglike_gau_args,
                     prior_transform_gau_args,
                     logl_args=(1, ),
                     ptform_args=(1, ),
                     logl_kwargs=dict(z=-1),
                     ptform_kwargs=dict(z=-1)) as pool:
        sampler = dynesty.DynamicNestedSampler(pool.loglike,
                                               pool.prior_transform,
                                               ndim,
                                               nlive=nlive,
                                               pool=pool,
                                               queue_size=100,
                                               rstate=rstate)
        sampler.run_nested(maxiter=300, print_progress=printing)

        assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1])
                < 5. * sampler.results['logzerr'][-1])

        # to ensure we get coverage
        terminator(pool)


def test_pool_args2():
    # test pool on gau problem
    # i specify large queue_size here, otherwise it is too slow

    # Here I am testing that args from pool and Nested sampler are
    # properly concatenated
    rstate = get_rstate()

    with dypool.Pool(
            2,
            loglike_gau_args,
            prior_transform_gau_args,
            logl_args=(1, ),
            ptform_args=(1, ),
            logl_kwargs={'a': 2},
            ptform_kwargs={'a': 2},
    ) as pool:
        sampler = dynesty.DynamicNestedSampler(pool.loglike,
                                               pool.prior_transform,
                                               ndim,
                                               nlive=nlive,
                                               pool=pool,
                                               logl_args=(-1, ),
                                               ptform_args=(-1, ),
                                               logl_kwargs={'b': -2},
                                               ptform_kwargs={'b': -2},
                                               queue_size=100,
                                               rstate=rstate)
        sampler.run_nested(maxiter=300, print_progress=printing)

        assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1])
                < 5. * sampler.results['logzerr'][-1])

        # to ensure we get coverage
        terminator(pool)


@pytest.mark.parametrize('sample', ['slice', 'rwalk', 'rslice'])
def test_pool_samplers(sample):
    # this is to test how the samplers are dealing with queue_size>1
    rstate = get_rstate()

    with mp.Pool(2) as pool:
        sampler = dynesty.NestedSampler(loglike_gau,
                                        prior_transform_gau,
                                        ndim,
                                        nlive=nlive,
                                        sample=sample,
                                        pool=pool,
                                        queue_size=100,
                                        rstate=rstate)
        sampler.run_nested(print_progress=printing)
        assert (abs(LOGZ_TRUTH_GAU - sampler.results['logz'][-1])
                < 5. * sampler.results['logzerr'][-1])
        terminator(pool)


POOL_KW = ['prior_transform', 'loglikelihood', 'propose_point', 'update_bound']


@pytest.mark.parametrize('func', POOL_KW)
def test_usepool(func):
    # test all the use_pool options, toggle them one by one
    rstate = get_rstate()
    use_pool = {}
    for k in POOL_KW:
        use_pool[k] = False
    use_pool[func] = True

    with mp.Pool(2) as pool:
        sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                               prior_transform_gau,
                                               ndim,
                                               nlive=nlive,
                                               rstate=rstate,
                                               use_pool=use_pool,
                                               pool=pool,
                                               queue_size=100)
        sampler.run_nested(maxiter=10000, print_progress=printing)
        terminator(pool)
