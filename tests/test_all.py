#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a series of basic tests to check whether anything huge is broken.

"""

# compatibility
from __future__ import (print_function, division)
from six.moves import range

# system functions that are always useful to have
import sys

# basic numeric setup
import numpy as np
from numpy import linalg

# plotting
import matplotlib
# if os.environ.get('DISPLAY', '') == '':
#    print('No display found. Using non-interactive Agg backend.')
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# dynesty
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

# seed the random number generator
np.random.seed(5647)

# configuration
printing = False
nlive = 1000


# EGGBOX
def loglike_egg(x):
    tmax = 5.0 * np.pi
    t = 2.0 * tmax * x - tmax
    return (2.0 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0))**5.0


def prior_transform_egg(x):
    return x


def test_ellipsoids():
    # stress test ellipsoid decompositions
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    2,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    first_update={
                                        'min_ncall': 0,
                                        'min_eff': 100
                                    })
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    lnz_truth = 235.88
    assert (abs(lnz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def bootstrap_tol(results):
    n = len(results.logz)
    niter = 50
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    means = []
    covs = []

    for i in range(niter):
        xid = np.random.randint(n, size=n)
        mean, cov = dyfunc.mean_and_cov(pos[xid], wts[xid])
        means.append(mean)
        covs.append(cov)
    return np.std(means, axis=0), np.std(covs, axis=0)


# basic checks
def check_results(C, lnz_truth, results, lz_tol, m_tol, c_tol, sig=5):
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz, logzerr = results.logz[-1], results.logzerr[-1]
    mean_check = np.all(np.abs(mean) < sig * m_tol)
    cov_check = np.all(np.abs(cov - C) < sig * c_tol)
    logz_check = abs((lnz_truth - logz)) < sig * lz_tol
    assert (mean_check)
    assert (cov_check)
    assert (logz_check)


# GAUSSIAN TEST

ndim_gau = 3
C_gau = np.identity(ndim_gau)  # set covariance to identity matrix
C_gau[C_gau == 0] = 0.95  # set off-diagonal terms (strongly correlated)
Cinv_gau = linalg.inv(C_gau)  # precision matrix
lnorm_gau = -0.5 * (np.log(2 * np.pi) * ndim_gau + np.log(linalg.det(C_gau))
                    )  # ln(norm)
lnz_truth_gau = ndim_gau * -np.log(2 * 10.)


# 3-D correlated multivariate normal log-likelihood
def loglikelihood_gau(x):
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot(x, np.dot(Cinv_gau, x)) + lnorm_gau


# prior transform
def prior_transform_gau(u):
    """Flat prior between -10. and 10."""
    return 10. * (2. * u - 1.)


# gradient (no jacobian)
def grad_x_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(Cinv_gau, x)


# gradient (with jacobian)
def grad_u_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(Cinv_gau, x) * 20.


def test_gaussian():

    # run sampling
    # check default behavior
    sys.stderr.write('\n\nDefault MVN\n')
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive)
    sampler.run_nested(print_progress=printing)
    # add samples
    # check continuation behavior
    sampler.run_nested(dlogz=0.1, print_progress=printing)

    # get errors
    # check resets and repeated runs
    means, covs, logzs = [], [], []
    nerr = 1
    for i in range(nerr):
        if printing:
            sys.stderr.write('\r{}/{}'.format(i + 1, nerr))
        sampler.reset()
        sampler.run_nested(print_progress=False)
        results = sampler.results
        pos = results.samples
        wts = np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(pos, wts)
        logz = results.logz[-1]
        means.append(mean)
        covs.append(cov)
        logzs.append(logz)

    # check summary
    res = sampler.results
    res.summary()

    # check plots
    dyplot.runplot(sampler.results)
    plt.close()
    dyplot.traceplot(sampler.results)
    plt.close()
    dyplot.cornerpoints(sampler.results)
    plt.close()
    dyplot.cornerplot(sampler.results)
    plt.close()
    dyplot.boundplot(sampler.results,
                     dims=(0, 1),
                     it=3000,
                     prior_transform=prior_transform_gau,
                     show_live=True,
                     span=[(-10, 10), (-10, 10)])
    plt.close()
    dyplot.cornerbound(sampler.results,
                       it=3500,
                       prior_transform=prior_transform_gau,
                       show_live=True,
                       span=[(-10, 10), (-10, 10)])
    plt.close()


def test_bounding():
    # check various bounding methods
    lz_tol = 1

    for bound in ['none', 'single', 'multi', 'balls', 'cubes']:
        sampler = dynesty.NestedSampler(loglikelihood_gau,
                                        prior_transform_gau,
                                        ndim_gau,
                                        nlive=nlive,
                                        bound=bound,
                                        sample='unif')
        sampler.run_nested(print_progress=printing)
        m_tol, c_tol = bootstrap_tol(sampler.results)
        check_results(C_gau, lnz_truth_gau, sampler.results, lz_tol, m_tol,
                      c_tol)


def test_sampling():
    # check various sampling methods
    lz_tol = 1
    for sample in ['unif', 'rwalk', 'rstagger', 'slice', 'rslice']:
        sampler = dynesty.NestedSampler(loglikelihood_gau,
                                        prior_transform_gau,
                                        ndim_gau,
                                        nlive=nlive,
                                        sample=sample)
        sampler.run_nested(print_progress=printing)
        m_tol, c_tol = bootstrap_tol(sampler.results)
        check_results(C_gau, lnz_truth_gau, sampler.results, lz_tol, m_tol,
                      c_tol)


# extra checks for gradients
def test_slice_nograd():
    lz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice')
    sampler.run_nested(print_progress=printing)
    m_tol, c_tol = bootstrap_tol(sampler.results)
    check_results(C_gau, lnz_truth_gau, sampler.results, lz_tol, m_tol, c_tol)


def test_slice_grad():
    lz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_x_gau,
                                    compute_jac=True)
    sampler.run_nested(print_progress=printing)
    m_tol, c_tol = bootstrap_tol(sampler.results)
    check_results(C_gau, lnz_truth_gau, sampler.results, lz_tol, m_tol, c_tol)


def test_slice_grad1():
    lz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_u_gau)
    sampler.run_nested(print_progress=printing)
    m_tol, c_tol = bootstrap_tol(sampler.results)
    check_results(C_gau, lnz_truth_gau, sampler.results, lz_tol, m_tol, c_tol)


def test_dynamic():
    # check dynamic nested sampling behavior
    lz_tol = 1
    dsampler = dynesty.DynamicNestedSampler(loglikelihood_gau,
                                            prior_transform_gau, ndim_gau)
    dsampler.run_nested(print_progress=printing)
    m_tol, c_tol = bootstrap_tol(dsampler.results)

    check_results(C_gau, lnz_truth_gau, dsampler.results, lz_tol, m_tol, c_tol)

    # check error analysis functions
    dres = dyfunc.jitter_run(dsampler.results)
    check_results(dres, lz_tol, m_tol, c_tol)
    dres = dyfunc.resample_run(dsampler.results)
    check_results(dres, lz_tol, m_tol, c_tol)
    dres = dyfunc.simulate_run(dsampler.results)
    check_results(dres, lz_tol, m_tol, c_tol)
    sys.stderr.write('KLD Error\n')
    dyfunc.kld_error(dsampler.results)
