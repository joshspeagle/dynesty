import numpy as np
import dynesty
"""
Run a series of basic tests of the 2d eggbox
"""

nlive = 1000
printing = False

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


def test_bounds():
    # stress test various boundaries
    ndim = 2
    for bound in ['multi', 'balls', 'cubes']:
        sampler = dynesty.NestedSampler(loglike_egg,
                                        prior_transform_egg,
                                        ndim,
                                        nlive=nlive,
                                        bound=bound,
                                        sample='unif')
        sampler.run_nested(dlogz=0.01, print_progress=printing)
        logz_truth = 235.856
        assert (abs(logz_truth - sampler.results.logz[-1]) <
                5. * sampler.results.logzerr[-1])


def test_ellipsoids_bootstrap():
    # stress test ellipsoid decompositions with bootstrap
    ndim = 2
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    bootstrap=5)
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])
