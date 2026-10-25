import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools

from utils import get_rstate, get_printing
import dynesty.internal_samplers as dysa
import dynesty  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = get_printing()


def bootstrap_tol(results, rstate):
    """ Compute the uncertainty of means/covs by doing bootstrapping """
    n = len(results['logz'])
    niter = 50
    pos = results.samples
    wts = results.importance_weights()
    means = []
    covs = []

    for i in range(niter):
        sub = rstate.uniform(size=n) < wts / wts.max()
        ind0 = np.nonzero(sub)[0]
        ind1 = rstate.choice(ind0, size=len(ind0), replace=True)
        mean = pos[ind1].mean(axis=0)
        cov = np.cov(pos[ind1].T)
        means.append(mean)
        covs.append(cov)
    return np.std(means, axis=0), np.std(covs, axis=0)


def check_results(results,
                  mean_truth,
                  cov_truth,
                  logz_truth,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=4):
    """ Check if means and covariances match match expectations
    within the tolerances

    """
    results.summary()
    pos = results.samples
    wts = np.exp(results['logwt'] - results['logz'][-1])
    assert np.allclose(results.importance_weights(), wts)
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz = results['logz'][-1]
    logzerr = results['logzerr'][-1]
    assert logzerr < 10  # check that it is not too large
    npt.assert_array_less(np.abs(mean - mean_truth), sig * mean_tol)
    npt.assert_array_less(np.abs(cov - cov_truth), sig * cov_tol)
    npt.assert_array_less(np.abs((logz_truth - logz)), sig * logz_tol)


class TestSampler(dysa.InternalSampler):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_sampler(self,
                        loglstar=None,
                        points=None,
                        axes=None,
                        seeds=None,
                        prior_transform=None,
                        loglikelihood=None,
                        nested_sampler=None):
        """

        """
        pass

    @staticmethod
    def sample(args):
        """

        """

        pass


class Gaussian:

    def __init__(self, corr=.95, prior_win=10, ndim=3):
        self.ndim = ndim
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
        x[:] = -np.ones(len(x))
        return ret

    # prior transform
    def prior_transform(self, u):
        """Flat prior between -10. and 10."""
        ret = self.prior_win * (2. * u - 1.)

        # notice here we overwrite the input array just to test
        # that this is not a problem
        u[:] = -np.ones(len(u))

        return ret

    # gradient (no jacobian)
    def grad_x(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, (x - self.mean))

    # gradient (with jacobian)
    def grad_u(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, x - self.mean) * 2 * self.prior_win


def check_results_gau(results, g, rstate, sig=4, logz_tol=None):
    if logz_tol is None:
        logz_tol = sig * results['logzerr'][-1]
    mean_tol, cov_tol = bootstrap_tol(results, rstate)
    # just check that resample_equal works
    dyfunc.resample_equal(results.samples,
                          np.exp(results['logwt'] - results['logz'][-1]))
    results.samples_equal()
    check_results(results,
                  g.mean,
                  g.cov,
                  g.logz_truth,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=sig)


# try all combinations
@pytest.mark.parametrize(
    "bound,sample",
    list(
        itertools.product(['single', 'multi', 'balls', 'cubes', 'none'],
                          ['unif', 'rwalk', 'slice', 'rslice'])))
def test_bounding_sample(bound, sample):
    # check various bounding methods

    rstate = get_rstate()
    if bound == 'none':
        if sample != 'unif':
            g = Gaussian(0.1)
        else:
            g = Gaussian(corr=0., prior_win=10)
            # make live easy if bound is none
            # but also not too easy so propose_point() is exercised
    else:
        g = Gaussian()
    sample = {
        'unif': dysa.UniformBoundSampler(),
        'rslice': dysa.RSliceSampler(),
        'rwalk': dysa.RWalkSampler(ncdim=g.ndim),
        'slice': dysa.SliceSampler()
    }[sample]
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)
    print(sampler.citations)


# try all combinations
@pytest.mark.parametrize("sample", ['rwalk', 'slice', 'rslice'])
@pytest.mark.parametrize("typ", [0, 1])
def test_walks_slices(sample, typ):
    # This tests if the walks= slices= options are used

    bound = 'single'
    g = Gaussian(0.1, ndim=2)
    res = []
    for i in range(2):
        # number of steps loop
        steps = {0: 10, 1: 20}[i]
        rstate = get_rstate()
        if typ == 0:
            cur_sample = {
                'rwalk': dysa.RWalkSampler(walks=steps),
                'slice': dysa.SliceSampler(slices=steps),
                'rslice': dysa.RSliceSampler(slices=steps),
            }[sample]
            kw = {}
        else:
            cur_sample = sample
            kw = {
                'rwalk': dict(walks=steps),
                'slice': dict(slices=steps),
                'rslice': dict(slices=steps),
            }[sample]
        sampler = dynesty.NestedSampler(g.loglikelihood,
                                        g.prior_transform,
                                        g.ndim,
                                        nlive=nlive,
                                        bound=bound,
                                        sample=cur_sample,
                                        rstate=rstate,
                                        **kw)
        sampler.run_nested(print_progress=printing)
        res.append(sampler.ncall)

    assert (res[1] > res[0])


@pytest.mark.parametrize("ndim,kw,typ,key,value",
                         [(12, {}, dysa.RWalkSampler, 'walks', 32),
                          (12, dict(walks=50), dysa.RWalkSampler, 'walks', 50),
                          (25, {}, dysa.RSliceSampler, 'slices', 28),
                          (25, dict(slices=7), dysa.RSliceSampler, 'slices',
                           7)])
def test_auto_walks_slices(ndim, kw, typ, key, value):
    # check that sample='auto' uses the walks= slices= options
    rstate = get_rstate()
    g = Gaussian(0.1, ndim=ndim)
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    sample='auto',
                                    rstate=rstate,
                                    **kw)
    # the configured sampler is used after the first bound update
    internal_sampler = sampler.internal_sampler_next
    assert isinstance(internal_sampler, typ)
    assert internal_sampler.sampler_kwargs[key] == value


def test_auto_walks_slices_mismatch():
    # slices= option with sample='auto' resolving to rwalk should warn
    rstate = get_rstate()
    g = Gaussian(0.1, ndim=12)
    with pytest.warns(UserWarning, match='does not make sense'):
        dynesty.NestedSampler(g.loglikelihood,
                              g.prior_transform,
                              g.ndim,
                              nlive=nlive,
                              sample='auto',
                              rstate=rstate,
                              slices=7)


class ParamSampler(dysa.RWalkSampler):
    """
    Custom sampler with its own constructor argument
    """

    def __init__(self, my_param=10, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param


def _get_sampler(sample, ndim=3, **kwargs):
    g = Gaussian(0.1, ndim=ndim)
    return dynesty.NestedSampler(g.loglikelihood,
                                 g.prior_transform,
                                 g.ndim,
                                 nlive=nlive,
                                 sample=sample,
                                 rstate=get_rstate(),
                                 **kwargs)


@pytest.mark.parametrize("positional", [False, True])
def test_custom_sampler_args(positional):
    # the arguments of custom sampler subclasses must be preserved
    # when dynesty configures the sampler for the problem
    if positional:
        cur_sample = ParamSampler(20, walks=7)
    else:
        cur_sample = ParamSampler(my_param=20, walks=7)
    sampler = _get_sampler(cur_sample)
    internal_sampler = sampler.internal_sampler_next
    assert isinstance(internal_sampler, ParamSampler)
    assert internal_sampler.my_param == 20
    assert internal_sampler.sampler_kwargs['walks'] == 7
    assert internal_sampler.ndim == 3


def test_custom_sampler_pickle():
    # configured samplers must survive pickling (i.e. checkpointing)
    import pickle
    sampler = _get_sampler(ParamSampler(my_param=20))
    internal_sampler = pickle.loads(
        pickle.dumps(sampler.internal_sampler_next))
    assert internal_sampler.my_param == 20
    sampler = _get_sampler(internal_sampler)
    assert sampler.internal_sampler_next.my_param == 20


@pytest.mark.parametrize("kw,nested_kw", [(dict(ndim=5), {}),
                                          (dict(periodic=[0]), {}),
                                          (dict(periodic=[0]),
                                           dict(periodic=[1]))])
def test_sampler_incompatible_params(kw, nested_kw):
    # the problem parameters of a sampler instance must agree with the
    # ones of the nested sampler
    with pytest.raises(ValueError, match='incompatible'):
        _get_sampler(dysa.RWalkSampler(**kw), **nested_kw)


def test_sampler_compatible_params():
    sampler = _get_sampler(dysa.RWalkSampler(ndim=3, periodic=(0, )),
                           periodic=[0])
    assert list(sampler.internal_sampler_next.sampler_kwargs['periodic']) == [
        0
    ]


@pytest.mark.parametrize("typ,kw", [(dysa.RWalkSampler, dict(walk=5)),
                                    (dysa.SliceSampler, dict(walks=5)),
                                    (dysa.RSliceSampler, dict(facc=0.3)),
                                    (dysa.UniformBoundSampler, dict(slices=5))
                                    ])
def test_sampler_unknown_kwargs(typ, kw):
    # unknown or meaningless options are not silently ignored
    with pytest.raises(TypeError):
        typ(**kw)


@pytest.mark.parametrize("sample,typ,key", [
    ('rwalk', dysa.RWalkSampler, 'walks'),
    ('slice', dysa.SliceSampler, 'slices'),
    ('rslice', dysa.RSliceSampler, 'slices'),
])
def test_sampler_instance_defaults(sample, typ, key):
    # sampler instances must have the same defaults as the string options
    ndim = 4
    val1 = _get_sampler(sample, ndim=ndim).internal_sampler_next.sampler_kwargs[key]
    val2 = _get_sampler(typ(), ndim=ndim).internal_sampler_next.sampler_kwargs[key]
    assert val1 == val2
    # and these depend on dimension
    assert val1 == {'rwalk': ndim + 20, 'slice': 3, 'rslice': 3 + ndim}[sample]


@pytest.mark.parametrize("typ,kw", [(dysa.RWalkSampler, dict(walks=50)),
                                    (dysa.SliceSampler, dict(slices=5)),
                                    (dysa.RWalkSampler, dict(slices=5)),
                                    (dysa.RWalkSampler, dict(facc=0.3))])
def test_sampler_instance_walks_slices(typ, kw):
    # walks/slices/facc options are ambiguous when a sampler instance is given
    with pytest.raises(ValueError, match='sampler instance'):
        _get_sampler(typ(), **kw)


@pytest.mark.parametrize("sample,kw,value",
                         [('rwalk', {}, 0.5), ('rwalk', dict(facc=0.3), 0.3),
                          (dysa.RWalkSampler(), {}, 0.5),
                          (dysa.RWalkSampler(facc=0.3), {}, 0.3)])
def test_sampler_facc(sample, kw, value):
    # facc is used both with the string option and the sampler instance
    sampler = _get_sampler(sample, **kw)
    assert sampler.internal_sampler_next.facc == value
