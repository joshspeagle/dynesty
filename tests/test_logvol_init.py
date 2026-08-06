"""
Regression tests for the treatment of the initial prior volume
(logvol_init) of runs where part of the prior has zero (-inf)
likelihood. They cover the final integral recomputation in static
run_nested(), the jitter_run()/resample_run() realizations and
merge_runs().
"""
import numpy as np
import pytest
from scipy.special import logsumexp
import dynesty
from dynesty import utils as dyfunc
from utils import get_rstate, get_printing

printing = get_printing()

# 2-d problem: logl = 0.1 * x0 for x1 < 0.35 and -inf otherwise
# Z = 0.35 * int_0^1 exp(0.1 x) dx = 0.35 * (e^0.1 - 1) / 0.1
LOGZ_TRUTH_2D = np.log(0.35 * (np.exp(0.1) - 1) / 0.1)


def loglike_2d(x):
    if x[1] < 0.35:
        return 0.1 * x[0]
    return -np.inf


def prior_transform(u):
    return u


@pytest.fixture(scope='module')
def restricted_run():
    # With a 0.35 finite-likelihood fraction the live-point
    # initialization takes several attempts (logvol_init < 0) and for
    # most seeds (including the default one) every live point ends up
    # with a finite likelihood. Without _LOWL_VAL filler points the
    # first dead point carries an appreciable likelihood, which is the
    # configuration where a wrong initial volume maximally biases the
    # first integration interval.
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_2d,
                                    prior_transform,
                                    2,
                                    nlive=100,
                                    sample='unif',
                                    bound='multi',
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    res = sampler.results
    # make sure the initialization really was restricted
    assert sampler.logvol_init < -0.5
    return res


def test_static_recompute(restricted_run):
    # the final recomputation of the integrals in run_nested() must not
    # discard the restricted initial prior volume
    assert np.abs(restricted_run['logz'][-1] - LOGZ_TRUTH_2D) < 0.35


def test_results_logvol_init_key(restricted_run):
    # the initial volume must be stored in the results
    assert restricted_run['logvol_init'] < -0.5


@pytest.mark.parametrize('func', [dyfunc.jitter_run, dyfunc.resample_run])
def test_jitter_resample(restricted_run, func):
    # realizations of the run must scatter around the true evidence
    # rather than be systematically shifted by -logvol_init
    rstate = get_rstate()
    logzs = np.array([
        func(restricted_run, rstate=rstate)['logz'][-1] for _ in range(20)
    ])
    assert np.abs(logzs.mean() - LOGZ_TRUTH_2D) < 0.35


def loglike_1d(x):
    if x[0] < 0.1:
        return x[0]
    return -np.inf


def test_merge():
    # merging runs with a restricted initial prior volume must pool the
    # initial volumes instead of restarting from unit volume
    logz_truth = np.log(np.exp(0.1) - 1)
    rstate = get_rstate()
    runs = []
    for i in range(2):
        sampler = dynesty.NestedSampler(loglike_1d,
                                        prior_transform,
                                        1,
                                        nlive=50,
                                        sample='unif',
                                        bound='none',
                                        rstate=rstate)
        sampler.run_nested(print_progress=printing)
        runs.append(sampler.results)
    merged = dyfunc.merge_runs(runs, print_progress=printing)
    assert np.abs(merged['logz'][-1] - logz_truth) < 0.5
    assert merged['logvol_init'] < -1


def test_dynamic_batch_combine():
    # a batch sampled from the whole prior carries its own initial
    # volume estimate which must be combined with the baseline one
    # using the same equations as merge_runs()
    rstate = get_rstate()
    dns = dynesty.DynamicNestedSampler(loglike_2d,
                                       prior_transform,
                                       2,
                                       sample='unif',
                                       bound='multi',
                                       rstate=rstate)
    for _ in dns.sample_initial(nlive=100, dlogz=0.01):
        pass
    lvi_base = dns.logvol_init
    assert lvi_base < -0.5
    for _ in dns.sample_batch(nlive_new=50, logl_bounds=(-np.inf, 0.05)):
        pass
    lvi_batch = dns.new_logvol_init
    dns.combine_runs()
    res = dns.results
    expected = np.log(150) - logsumexp(
        [np.log(100) - lvi_base, np.log(50) - lvi_batch])
    assert np.isclose(res['logvol_init'], expected)
    assert np.abs(res['logz'][-1] - LOGZ_TRUTH_2D) < 0.35


def test_compute_integrals():
    # with an initial volume X0 the quadrature must integrate over
    # [X_last, X0] exactly (the likelihood at X0 is padded with ~zero)
    logvol_init = -1.5
    logvol = logvol_init + np.log(np.linspace(0.9, 0.01, 30))
    logl = np.linspace(0., 1., 30)
    logwt, logz, logzvar, h = dyfunc.compute_integrals(
        logl=logl, logvol=logvol, logvol_init=logvol_init)
    x = np.concatenate([[np.exp(logvol_init)], np.exp(logvol)])
    lik = np.concatenate([[0], np.exp(logl)])
    expected = np.log(np.sum(0.5 * (lik[1:] + lik[:-1]) * (x[:-1] - x[1:])))
    assert np.abs(logz[-1] - expected) < 1e-6
