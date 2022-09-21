import numpy as np
from numpy import linalg
from utils import get_rstate, get_printing

import dynesty  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = get_printing()

# GAUSSIAN TEST


class Gaussian:

    def __init__(self, corr=.95, prior_win=10):
        self.ndim = 3
        self.mean = np.linspace(-1, 1, self.ndim)
        self.cov = np.identity(self.ndim)  # set covariance to identity matrix
        self.cov[self.cov ==
                 0] = corr  # set off-diagonal terms (strongly correlated)
        self.cov_inv = linalg.inv(self.cov)  # precision matrix
        self.lnorm = -0.5 * (np.log(2 * np.pi) * self.ndim +
                             np.log(linalg.det(self.cov)))
        self.prior_win = prior_win  # +/- on both sides
        self.logz_truth = self.ndim * (-np.log(2 * self.prior_win))

    # 3-D correlated multivariate normal log-likelihood
    def loglikelihood(self, x):
        """Multivariate normal log-likelihood."""

        ret = -0.5 * np.dot(
            (x - self.mean), np.dot(self.cov_inv,
                                    (x - self.mean))) + self.lnorm
        # notice here we overwrite the input array just to test
        # that this is not a problem
        blob = x * 1
        x[:] = -np.ones(len(x))
        return ret, blob

    # prior transform
    def prior_transform(self, u):
        """Flat prior between -10. and 10."""
        ret = self.prior_win * (2. * u - 1.)

        # notice here we overwrite the input array just to test
        # that this is not a problem
        u[:] = -np.ones(len(u))

        return ret


def test_gaussian():
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    blob=True)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    (res['blob'])
