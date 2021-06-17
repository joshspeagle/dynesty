import numpy as np
import dynesty
import dynesty.utils as dyutil
"""
Run a series of basic tests changing various things like
maxcall options and potentially other things
"""

nlive = 100

size = 10  # box size


def loglike(x):
    return -0.5 * np.sum(x**2)


def loglike_inf(x):
    r2 = np.sum(x**2)
    if r2 > size * size or np.abs(x[0]) < 0.1:
        return -np.inf
    return -0.5 * r2


def prior_transform(x):
    return (2 * x - 1) * size


def test_maxcall():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested(dlogz=1, maxcall=1000)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive)
    sampler.run_nested(dlogz_init=1, maxcall=1000)


def test_inf():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.NestedSampler(loglike_inf,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested()

    sampler = dynesty.DynamicNestedSampler(loglike_inf,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive)
    sampler.run_nested(dlogz_init=1)


def test_unravel():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested()
    dyutil.unravel_run(sampler.results)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive)
    sampler.run_nested(dlogz_init=1, maxcall=1000)
    dyutil.unravel_run(sampler.results)
    logps = sampler.results.logl
    dyutil.reweight_run(sampler.results, logps / 4.)


def test_livepoints():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    live_u = np.random.uniform(size=(ndim, 2))
    live_v = np.array([prior_transform(_) for _ in live_u])
    live_logl = np.array([loglike(_) for _ in live_v])
    live_points = [live_u, live_v, live_logl]
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    live_points=live_points)
    sampler.run_nested()
    dyutil.unravel_run(sampler.results)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           live_points=live_points)
    sampler.run_nested(dlogz_init=1, maxcall=1000)
