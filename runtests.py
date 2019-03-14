#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a series of basic tests to check whether anything huge is broken.

"""

# compatibility
from __future__ import (print_function, division)
from six.moves import range

# system functions that are always useful to have
import time
import sys
import os

# basic numeric setup
import numpy as np
from numpy import linalg

# plotting
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
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
def loglike(x):
    tmax = 5.0 * np.pi
    t = 2.0 * tmax * x - tmax
    return (2.0 + np.cos(t[0] / 2.0) * np.cos(t[1] / 2.0)) ** 5.0


def prior_transform(x):
    return x


# stress test ellipsoid decompositions
sys.stderr.write('\nEggbox\n')
sampler = dynesty.NestedSampler(loglike, prior_transform, 2, nlive=nlive,
                                bound='multi', sample='unif',
                                first_update={'min_ncall': 0, 'min_eff': 100})
sampler.run_nested(dlogz=0.01, print_progress=printing)
lnz_truth = 235.88
sys.stderr.write('\nlogz: {}'.format(abs(lnz_truth - sampler.results.logz[-1])
                                     < 5. * sampler.results.logzerr[-1]))


# basic checks
def check_results(results, lz_tol, m_tol, c_tol, sig=5):
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz, logzerr = results.logz[-1], sampler.results.logzerr[-1]
    mean_check = np.all(np.abs(mean) < sig * m_tol)
    cov_check = np.all(np.abs(cov - C) < sig * c_tol)
    logz_check = abs((lnz_truth - logz)) < sig * lz_tol
    sys.stderr.write('\nlogz: {} | mean: {} | cov: {}\n'
                     .format(logz_check, mean_check, cov_check))


# GAUSSIAN TEST
ndim = 3
C = np.identity(ndim)  # set covariance to identity matrix
C[C == 0] = 0.95  # set off-diagonal terms (strongly correlated)
Cinv = linalg.inv(C)  # precision matrix
lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(linalg.det(C)))  # ln(norm)
lnz_truth = ndim * -np.log(2 * 10.)


# 3-D correlated multivariate normal log-likelihood
def loglikelihood(x):
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm


# prior transform
def prior_transform(u):
    """Flat prior between -10. and 10."""
    return 10. * (2. * u - 1.)


# gradient (no jacobian)
def grad_x(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(Cinv, x)


# gradient (with jacobian)
def grad_u(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(Cinv, x) * 20.


# run sampling
# check default behavior
sys.stderr.write('\n\nDefault MVN\n')
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                nlive=nlive)
sampler.run_nested(print_progress=printing)

# add samples
# check continuation behavior
sys.stderr.write('\n\nExtra samples\n')
sampler.run_nested(dlogz=0.1, print_progress=printing)


# get errors
# check resets and repeated runs
sys.stderr.write('\n\nDeriving Errors\n')
means, covs, logzs = [], [], []
nerr = 50
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
if printing:
    sys.stderr.write('\n')
lz_tol, m_tol, c_tol = (np.std(logzs), np.std(means, axis=0),
                        np.std(covs, axis=0))
sys.stderr.write('logz_tol: {}\n'.format(lz_tol))
sys.stderr.write('mean_tol: {}\n'.format(m_tol))
sys.stderr.write('cov_tol: {}\n'.format(c_tol))

# check summary
sys.stderr.write('\nResults\n')
res = sampler.results
res.summary()

# check plots
sys.stderr.write('\nPlotting\n')
sys.stderr.write('Summary/Run Plot\n')
dyplot.runplot(sampler.results)
plt.close()
sys.stderr.write('Trace Plot\n')
dyplot.traceplot(sampler.results)
plt.close()
sys.stderr.write('Sub-Corner Plot (Points)\n')
dyplot.cornerpoints(sampler.results)
plt.close()
sys.stderr.write('Corner Plot (Contours)\n')
dyplot.cornerplot(sampler.results)
plt.close()
sys.stderr.write('2-D Bound Plot\n')
dyplot.boundplot(sampler.results, dims=(0, 1), it=3000,
                 prior_transform=prior_transform, show_live=True,
                 span=[(-10, 10), (-10, 10)])
plt.close()
sys.stderr.write('Sub-Corner Plot (Bounds)\n')
dyplot.cornerbound(sampler.results, it=3500, prior_transform=prior_transform,
                   show_live=True, span=[(-10, 10), (-10, 10)])
plt.close()

# check various bounding methods
for bound in ['none', 'single', 'multi', 'balls', 'cubes']:
    sys.stderr.write('\n'+bound+'\n')
    sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                    nlive=nlive, bound=bound, sample='unif')
    sampler.run_nested(print_progress=printing)
    check_results(sampler.results, lz_tol, m_tol, c_tol)

# check various sampling methods
for sample in ['unif', 'rwalk', 'rstagger', 'slice', 'rslice']:
    sys.stderr.write('\n'+sample+'\n')
    sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                    nlive=nlive, sample=sample)
    sampler.run_nested(print_progress=printing)
    check_results(sampler.results, lz_tol, m_tol, c_tol)

# extra checks for gradients
sys.stderr.write('\nhslice (no grad)\n')
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                nlive=nlive, sample='hslice')
sampler.run_nested(print_progress=printing)
check_results(sampler.results, lz_tol, m_tol, c_tol)

sys.stderr.write('\nhslice (grad w/o jac)\n')
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                nlive=nlive, sample='hslice', gradient=grad_x,
                                compute_jac=True)
sampler.run_nested(print_progress=printing)
check_results(sampler.results, lz_tol, m_tol, c_tol)

sys.stderr.write('\nhslice (grad w/ jac)\n')
sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim,
                                nlive=nlive, sample='hslice', gradient=grad_u)
sampler.run_nested(print_progress=printing)
check_results(sampler.results, lz_tol, m_tol, c_tol)

# check dynamic nested sampling behavior
sys.stderr.write('\nDynamic Nested Sampling\n')
dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim)
dsampler.run_nested(print_progress=printing)
check_results(dsampler.results, lz_tol, m_tol, c_tol)

# check error analysis functions
sys.stderr.write('\nError Analysis\n')
sys.stderr.write('Jittering')
dres = dyfunc.jitter_run(dsampler.results)
check_results(dres, lz_tol, m_tol, c_tol)
sys.stderr.write('Resampling')
dres = dyfunc.resample_run(dsampler.results)
check_results(dres, lz_tol, m_tol, c_tol)
sys.stderr.write('Combined')
dres = dyfunc.simulate_run(dsampler.results)
check_results(dres, lz_tol, m_tol, c_tol)
sys.stderr.write('KLD Error\n')
dyfunc.kld_error(dsampler.results)

# if we got to the end, then we know things at least aren't totally broken!
