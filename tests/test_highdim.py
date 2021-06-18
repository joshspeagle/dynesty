import numpy as np
from scipy import linalg
import scipy.stats
import dynesty
import multiprocessing as mp
from utils import get_rstate
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 5000
printing = False


def get_covar(rstate, ndim):
    eigval = 10**np.linspace(-3, 0, ndim)
    M = scipy.stats.ortho_group.rvs(dim=ndim, random_state=rstate)
    ret = M @ np.diag(eigval**2) @ M.T
    return ret


class Config:
    def __init__(self, rstate, ndim_gau):
        self.ndim_gau = ndim_gau
        self.mean_gau = np.linspace(-1, 1, ndim_gau)
        cov_gau = get_covar(rstate, ndim_gau)
        self.cov_gau = cov_gau
        self.cov_inv_gau = linalg.pinvh(cov_gau)  # precision matrix
        logdet = np.linalg.slogdet(cov_gau)[1]
        self.lnorm_gau = -0.5 * (np.log(2 * np.pi) * ndim_gau + logdet)
        self.prior_win = 1000  # +/- 10 on both sides
        self.logz_truth_gau = ndim_gau * (-np.log(2 * self.prior_win))
        assert (np.isfinite(self.lnorm_gau))
        assert (np.isfinite(self.logz_truth_gau))


class si:
    config = None


# 3-D correlated multivariate normal log-likelihood
class LogL:
    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        """Multivariate normal log-likelihood."""
        co = self.config
        x1 = x - co.mean_gau
        return -0.5 * x1 @ co.cov_inv_gau @ x1 + co.lnorm_gau


# prior transform


class Prior:
    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        """Flat prior between -10. and 10."""
        return self.config.prior_win * (2. * x - 1.)


def do_gaussian(co, sample=None, bound=None, rstate=None):
    curlogl = LogL(co)
    curprior = Prior(co)
    sampler = dynesty.DynamicNestedSampler(curlogl,
                                           curprior,
                                           co.ndim_gau,
                                           nlive=nlive,
                                           bound=bound,
                                           sample=sample,
                                           rstate=rstate,
                                           vol_dec=.25)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    return res.logz[-1], res.logzerr[-1]


def do_gaussians(sample='rslice', bound='single'):
    pool = mp.Pool(36)
    res = []
    for ndim in range(2, 33):
        rstate = np.random.default_rng(ndim)
        co = Config(rstate, ndim)
        res.append(
            (ndim, co,
             pool.apply_async(do_gaussian, (co, ),
                              dict(sample=sample, bound=bound,
                                   rstate=rstate))))
    for ndim, co, curres in res:
        curres = curres.get()
        print('RESULTS', ndim, curres, co.logz_truth_gau)
