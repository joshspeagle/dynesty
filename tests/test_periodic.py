import numpy as np
import dynesty
import scipy.stats
from dynesty import bounding
from dynesty.internal_samplers import SamplerArgument, UniformBoundSampler
import pytest
from scipy.special import erf, i0e
from utils import get_rstate, get_printing

nlive = 100
printing = get_printing()
win = 100
ndim = 2


def loglike(x):
    return -0.5 * x[1]**2


def prior_transform(x):
    return (2 * x - 1) * win


@pytest.mark.parametrize("sampler,dynamic", [('rwalk', True), ('unif', True),
                                             ('rslice', True),
                                             ('unif', False)])
def test_periodic(sampler, dynamic):
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    logz_true = np.log(np.sqrt(2 * np.pi) * erf(win / np.sqrt(2)) / (2 * win))
    thresh = 8
    # This is set up to higher level
    # becasue of failures at ~5ssigma level
    # this needs to be investigated
    rstate = get_rstate()
    if dynamic:
        dns = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           periodic=[0],
                                           rstate=rstate,
                                           sample=sampler)
        dns.run_nested(dlogz_init=1, print_progress=printing)
    else:
        dns = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    periodic=[0],
                                    rstate=rstate,
                                    sample=sampler)
        dns.run_nested(dlogz=1, print_progress=printing)
    assert (np.abs(dns.results['logz'][-1] - logz_true)
            < thresh * dns.results['logzerr'][-1])


def test_error():
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[22],
                                     rstate=rstate)


def test_error2():
    # check you cant combine periodic/reflective for one var
    rstate = get_rstate()
    with pytest.raises(ValueError):
        dynesty.DynamicNestedSampler(loglike,
                                     prior_transform,
                                     ndim,
                                     nlive=nlive,
                                     periodic=[1],
                                     reflective=[1],
                                     rstate=rstate)


def prior_transform_strict(x):
    # identity prior transform that refuses points outside the cube
    if x.min() < 0 or x.max() > 1:
        raise ValueError(f'point outside the unit cube {x}')
    return x


@pytest.mark.parametrize("kind", ['periodic', 'reflective'])
def test_unif_nonbounded_edge(kind):
    # Here the likelihood depends on the periodic/reflective variable
    # and is peaked at the edge of the cube. The uniform sampler
    # used to accept points outside the cube in those dimensions and
    # pass them to the prior transform, which could strongly bias logz
    # (i.e. by ~+4 with a single elongated ellipsoid reaching u=-0.5)
    kappa = 20
    sig = 0.1

    def loglike_edge(x):
        # normalized gaussian in x0 times normalized von Mises
        # (periodic) or exponential (reflective) distributions in x1
        ret = -0.5 * ((x[0] - 0.5) / sig)**2 - np.log(np.sqrt(2 * np.pi) *
                                                      sig)
        if kind == 'periodic':
            ret += kappa * np.cos(2 * np.pi * x[1]) - np.log(
                i0e(kappa)) - kappa
        else:
            ret += -x[1] / sig - np.log(sig * (1 - np.exp(-1 / sig)))
        return ret

    rstate = get_rstate()
    dns = dynesty.NestedSampler(loglike_edge,
                                prior_transform_strict,
                                ndim,
                                nlive=nlive,
                                sample='unif',
                                rstate=rstate,
                                **{kind: [1]})
    dns.run_nested(print_progress=printing)
    res = dns.results
    assert res['samples_u'].min() >= 0 and res['samples_u'].max() <= 1
    # the true logz is zero (up to the negligible truncation of the gaussian)
    assert np.abs(res['logz'][-1]) < 5 * res['logzerr'][-1]


@pytest.mark.parametrize("kind", ['periodic', 'reflective'])
def test_unif_nonbounded_uniform(kind):
    # The uniform sampler maps the points of the bound that are outside
    # the cube in periodic/reflective dimensions into the cube.
    # Here the bound overlaps with its mapped image, so the sampling will
    # only be uniform if the mapped points are not double counted
    rstate = get_rstate()
    # ellipse covering -0.5..0.7 in x0 and 0.2..0.8 in x1
    ell = bounding.Ellipsoid(2,
                             ctr=np.array([0.1, 0.5]),
                             cov=np.diag([0.6**2, 0.3**2]))
    args = SamplerArgument(u=None,
                           loglstar=-np.inf,
                           axes=None,
                           scale=1,
                           prior_transform=lambda x: x,
                           loglikelihood=lambda x: 0.,
                           rseed=rstate,
                           kwargs=dict(bound=ell,
                                       ndim=2,
                                       n_cluster=2,
                                       nonbounded=np.array([False, True]),
                                       **{kind: [0]}))
    nsamp = 5000
    samps = np.array(
        [UniformBoundSampler.sample(args).u for _ in range(nsamp)])
    assert samps.min() >= 0 and samps.max() <= 1
    # reference sample from the cube within the bound or its mapped image
    ref = rstate.uniform(size=(20 * nsamp, 2))
    alt = ref.copy()
    if kind == 'periodic':
        alt[:, 0] = np.where(ref[:, 0] < 0.5, ref[:, 0] + 1, ref[:, 0] - 1)
    else:
        alt[:, 0] = np.where(ref[:, 0] < 0.5, -ref[:, 0], 2 - ref[:, 0])
    good = np.array([ell.contains(a) or ell.contains(b)
                     for a, b in zip(ref, alt)])
    ref = ref[good]
    for i in range(2):
        assert scipy.stats.ks_2samp(samps[:, i], ref[:, i]).pvalue > 1e-3
