#!/usr/bin/env py.test
import numpy as np

import nestle

# Note that this is a terrible test in that it will only pass for some 
# random seeds, so if you change the seed, it may fail.
def test_two_gaussian_nest():
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

    np.random.seed(0)
    res = nestle.nest(logl, prior, 2, nobj=100)
    print "evidence = {0:6.3f} +/- {1:6.3f}".format(res.logz, res.logzerr)

    #(Approximate) analytic evidence for two identical Gaussian blobs,
    # over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:
    analytic_logz = np.log(2.0 * 2.0*np.pi*sigma*sigma / 100.)
    print "analytic = {0:6.3f}".format(analytic_logz)

    # calculate evidence on fine grid.
    dx = 0.1
    xv = np.arange(-5.0 + dx/2., 5., dx)
    yv = np.arange(-5.0 + dx/2., 5., dx)
    grid_logz = -1.e300
    for x in xv:
        for y in yv:
            grid_logz = np.logaddexp(grid_logz, logl(np.array([x, y])))
    grid_logz += np.log(dx * dx / 100.)  # adjust for point density
    print "grid_logz =", grid_logz

    assert abs(res.logz - analytic_logz) < 2.0 * res.logzerr
    assert abs(res.logz - grid_logz) < 2.0 * res.logzerr

if __name__ == "__main__":
    test_two_gaussian_nest()
