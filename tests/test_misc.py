import numpy as np
import pytest
import dynesty
import pickle
import dynesty.utils as dyutil

from utils import get_rstate, get_printing
"""
Run a series of basic tests changing various things like
maxcall options and potentially other things
"""

printing = get_printing()

nlive = 100

size = 10  # box size


def loglike(x):
    return -0.5 * np.sum(x**2)


def loglike_inf(x):
    r2 = np.sum(x**2)
    if r2 > size * size or np.abs(x[0]) < 0.1:
        return -np.inf
    return -0.5 * r2


class MyException(Exception):
    pass


def loglike_exc(x):
    r2 = np.sum(x**2)
    if r2 < 0.1:
        raise MyException('ooops')
    return -0.5 * r2


def prior_transform(x):
    return (2 * x - 1) * size


def test_maxcall():
    # test of maxcall functionality
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(dlogz=1, maxcall=1000, print_progress=printing)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, maxcall=1000, print_progress=printing)


def test_pickle():
    # test of maxcall functionality
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    pickle.dumps(sampler)
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(print_progress=printing)
    pickle.dumps(sampler)


def test_inf():
    # Test of logl that returns -inf
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_inf,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)

    sampler = dynesty.DynamicNestedSampler(loglike_inf,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)


def test_unravel():
    # test unravel_run
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    dyutil.unravel_run(sampler.results)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, maxcall=1000, print_progress=printing)
    dyutil.unravel_run(sampler.results)
    logps = sampler.results.logl
    dyutil.reweight_run(sampler.results, logps / 4.)


def test_livepoints():
    # Test the providing of initial live-points to the sampler
    ndim = 2
    rstate = get_rstate()
    live_u = rstate.uniform(size=(nlive, ndim))
    live_v = np.array([prior_transform(_) for _ in live_u])
    live_logl = np.array([loglike(_) for _ in live_v])
    live_points = [live_u, live_v, live_logl]
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    live_points=live_points,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    dyutil.unravel_run(sampler.results)


def test_exc():
    # Test of exceptions that the exception is reraised
    ndim = 2
    rstate = get_rstate()
    with pytest.raises(MyException):
        sampler = dynesty.NestedSampler(loglike_exc,
                                        prior_transform,
                                        ndim,
                                        nlive=nlive,
                                        rstate=rstate)
        sampler.run_nested(print_progress=printing)


def test_neff():
    # test of neff functionality
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    assert sampler.n_effective == 0
    sampler.run_nested(print_progress=printing)
    assert sampler.n_effective > 10

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    assert sampler.n_effective == 0
    sampler.run_nested(dlogz_init=1, n_effective=1000, print_progress=printing)
    assert sampler.n_effective > 1000
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1,
                       n_effective=10000,
                       print_progress=printing)
    assert sampler.n_effective > 10000


def test_oldstop():
    # test of old stopping function functionality
    ndim = 2
    rstate = get_rstate()
    import dynesty.utils as dyutil
    stopfn = dyutil.old_stopping_function
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1,
                       n_effective=None,
                       stop_function=stopfn,
                       print_progress=printing)


def test_stop_nmc():
    # test stopping relying in n_mc
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1,
                       n_effective=None,
                       stop_kwargs=dict(n_mc=25),
                       print_progress=printing)


def test_results():
    # test of various results interfaces functionality
    ndim = 2
    rstate = get_rstate()

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    for k in res.keys():
        pass
    for k, v in res.items():
        pass
    for k, v in res.asdict().items():
        pass


@pytest.mark.parametrize('ndim', [2, 10])
def test_deterministic(ndim):
    # test we are deterministic

    results = []
    for i in range(2):
        rstate = get_rstate()

        sampler = dynesty.DynamicNestedSampler(loglike,
                                               prior_transform,
                                               ndim,
                                               nlive=nlive,
                                               rstate=rstate)
        sampler.run_nested(print_progress=printing)
        res = sampler.results
        results.append(res)

    for k in results[0].keys():
        val0 = results[0][k]
        val1 = results[1][k]
        if (isinstance(val0, int) or isinstance(val0, float)
                or isinstance(val0, np.ndarray)):
            assert np.allclose(val0, val1)


def test_update_interval():
    # test that we cab set update_interval
    # ideally i'd need to see if it makes a difference...
    ndim = 2
    rstate = get_rstate()

    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    update_interval=10)
    sampler.run_nested(print_progress=printing)
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    update_interval=0.5)
    sampler.run_nested(print_progress=printing)
