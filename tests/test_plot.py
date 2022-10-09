import numpy as np
import pytest
from numpy import linalg
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

    def __init__(self, ndim=3, corr=.95):
        self.ndim = ndim
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


@pytest.mark.parametrize("dynamic,periodic,ndim,bound",
                         [(False, False, 3, 'multi'),
                          (True, False, 3, 'multi'), (True, True, 3, 'multi'),
                          (True, False, 1, 'multi')])
def test_gaussian(dynamic, periodic, ndim, bound):
    rstate = get_rstate()
    g = Gaussian(ndim=ndim)
    if periodic:
        periodic = [0]
    else:
        periodic = None
    if dynamic:
        sampler = dynesty.DynamicNestedSampler(g.loglikelihood,
                                               g.prior_transform,
                                               g.ndim,
                                               nlive=nlive,
                                               rstate=rstate,
                                               periodic=periodic,
                                               bound=bound)
    else:
        sampler = dynesty.NestedSampler(g.loglikelihood,
                                        g.prior_transform,
                                        g.ndim,
                                        nlive=nlive,
                                        rstate=rstate,
                                        periodic=periodic,
                                        bound=bound)
    sampler.run_nested(print_progress=printing)
    results = sampler.results
    # check plots
    dyplot.runplot(results)
    plt.close()
    dyplot.runplot(results, span=[(0., 10.), 0.001, 0.2, (None, None)])
    plt.close()
    dyplot.runplot(results, logplot=True)
    dyplot.runplot(results,
                   fig=(plt.gcf(), plt.gcf().axes),
                   max_x_ticks=0,
                   max_y_ticks=0)
    plt.close()
    dyplot.traceplot(results)
    dyplot.traceplot(results, smooth=[10] * ndim)
    plt.close()
    dyplot.traceplot(results, connect=True)
    dyplot.traceplot(results,
                     fig=(plt.gcf(), plt.gcf().axes),
                     show_titles=True,
                     truths=np.zeros(ndim),
                     verbose=True,
                     max_n_ticks=0)
    plt.close()

    truths = np.zeros(ndim)
    truths[0] = -.1
    span = [[-10, 10]] * ndim
    if ndim > 1:
        truths[1] = .1
        span[1] = .9

    dyplot.cornerplot(results, show_titles=True, truths=truths)
    plt.close()
    dyplot.cornerplot(results,
                      smooth=10,
                      verbose=True,
                      hist2d_kwargs=dict(plot_datapoints=True),
                      max_n_ticks=0)
    plt.close()
    if ndim != 1:
        # cornerbound
        dyplot.cornerbound(results,
                           it=500,
                           prior_transform=g.prior_transform,
                           show_live=True,
                           span=span)
        dyplot.cornerbound(results,
                           it=500,
                           show_live=True,
                           span=span,
                           fig=(plt.gcf(), plt.gcf().axes))
        plt.close()
        # boundplot
        dyplot.boundplot(results,
                         dims=(0, 1)[:min(ndim, 2)],
                         it=1000,
                         prior_transform=g.prior_transform,
                         show_live=True,
                         span=span)
        plt.close()
        dyplot.boundplot(results, dims=(0, 1)[:min(ndim, 2)], it=1000)
        plt.close()

        # cornerpoints
        dyplot.cornerpoints(results)
        plt.close()
        dyplot.cornerpoints(results, span=span, truths=truths, max_n_ticks=0)
        plt.close()


@pytest.mark.parametrize("bound", ['balls', 'cubes'])
def test_gaussianx(bound):
    ndim = 3
    rstate = get_rstate()
    g = Gaussian(ndim=ndim)
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    bound=bound)
    sampler.run_nested(print_progress=printing)
    results = sampler.results
    dyplot.boundplot(results,
                     dims=(0, 1)[:min(ndim, 2)],
                     it=1000,
                     prior_transform=g.prior_transform,
                     show_live=False)
    dyplot.boundplot(results,
                     dims=(0, 1)[:min(ndim, 2)],
                     it=3000,
                     prior_transform=g.prior_transform,
                     show_live=False)
    dyplot.cornerbound(results,
                       it=3000,
                       prior_transform=g.prior_transform,
                       show_live=False)
