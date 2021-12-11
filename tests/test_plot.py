import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools
from utils import get_rstate, get_printing
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa
import dynesty  # noqa
from dynesty import plotting as dyplot  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = get_printing()


class Gaussian:
    def __init__(self, corr=.95):
        self.ndim = 3
        self.mean = np.linspace(-1, 1, self.ndim)
        self.cov = np.identity(self.ndim)  # set covariance to identity matrix
        self.cov[self.cov ==
                 0] = corr  # set off-diagonal terms (strongly correlated)
        self.cov_inv = linalg.inv(self.cov)  # precision matrix
        self.lnorm = -0.5 * (np.log(2 * np.pi) * self.ndim +
                             np.log(linalg.det(self.cov)))
        self.prior_win = 10  # +/- 10 on both sides
        self.logz_truth = self.ndim * (-np.log(2 * self.prior_win))

    # 3-D correlated multivariate normal log-likelihood
    def loglikelihood(self, x):
        """Multivariate normal log-likelihood."""
        return -0.5 * np.dot(
            (x - self.mean), np.dot(self.cov_inv,
                                    (x - self.mean))) + self.lnorm

    # prior transform
    def prior_transform(self, u):
        """Flat prior between -10. and 10."""
        return self.prior_win * (2. * u - 1.)

    # gradient (no jacobian)
    def grad_x(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, (x - self.mean))

    # gradient (with jacobian)
    def grad_u(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, x - self.mean) * 2 * self.prior_win


@pytest.mark.parametrize("dynamic", [(False, ), (True, )])
def test_gaussian(dynamic):
    rstate = get_rstate()
    g = Gaussian()
    for i in range(2):
        if i == 0:
            sampler = dynesty.NestedSampler(g.loglikelihood,
                                            g.prior_transform,
                                            g.ndim,
                                            nlive=nlive,
                                            rstate=rstate)
        else:
            sampler = dynesty.DynamicNestedSampler(g.loglikelihood,
                                                   g.prior_transform,
                                                   g.ndim,
                                                   nlive=nlive,
                                                   rstate=rstate)
        sampler.run_nested(print_progress=printing)
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
                         prior_transform=g.prior_transform,
                         show_live=True,
                         span=[(-10, 10), (-10, 10)])
        plt.close()
        dyplot.cornerbound(sampler.results,
                           it=3500,
                           prior_transform=g.prior_transform,
                           show_live=True,
                           span=[(-10, 10), (-10, 10)])
        dyplot.cornerbound(sampler.results,
                           it=3500,
                           show_live=True,
                           span=[(-10, 10), (-10, 10)],
                           fig=(plt.gcf(), plt.gcf().axes))
        plt.close()
