#!/usr/bin/env python
import numpy as np

import nestle


np.random.seed(0)

# Test problem with a two-mode gaussian.

# gaussians centered at (1, 1) and (-1, -1)
mu1 = np.ones(2)
mu2 = -np.ones(2)

# Width of 0.1 in each dimension
sigma = 0.1
ivar = 1.0/(sigma*sigma)
sigma1inv = np.diag([ivar, ivar])
sigma2inv = np.diag([ivar, ivar])

def logl(x):
    dx1 = x - mu1
    dx2 = x - mu2
    return np.logaddexp(-np.dot(dx1, np.dot(sigma1inv, dx1))/2.0,
                        -np.dot(dx2, np.dot(sigma2inv, dx2))/2.0)

# Use a flat prior, over [-5, 5] in both dimensions
def prior(x):
    return 10.0 * x - 5.0


def test_two_gaussian_nest():
    # (Approximate) analytic evidence for two identical Gaussian blobs,
    # over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:
    analytic_evidence = np.log(2.0 * 2.0*np.pi*sigma*sigma / 100.0)

    res = nestle.nest(logl, prior, 2, nobj=100)

    print "logz =", res.logz, " +/- ", res.logzerr
    print "analytic", analytic_evidence

test_two_gaussian_nest()
