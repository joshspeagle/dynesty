import numpy as np
from numpy import linalg
from utils import get_rstate, get_printing
import pytest

import dynesty
import dynesty.pool as dypool
from dynesty import utils as dyfunc

nlive = 500
printing = get_printing()


class Gaussian:

    def __init__(self, corr=.95, prior_win=10):
        self.ndim = 3
        self.mean = np.linspace(-1, 1, self.ndim)
        self.cov = np.identity(self.ndim)
        self.cov[self.cov ==
                 0] = corr
        self.cov_inv = linalg.inv(self.cov)
        self.lnorm = -0.5 * (np.log(2 * np.pi) * self.ndim +
                             np.log(linalg.det(self.cov)))
        self.prior_win = prior_win
        self.logz_truth = self.ndim * (-np.log(2 * self.prior_win))

    def loglikelihood(self, x):
        ret = -0.5 * np.dot(
            (x - self.mean), np.dot(self.cov_inv,
                                    (x - self.mean))) + self.lnorm
        return ret

    def prior_transform(self, u):
        ret = self.prior_win * (2. * u - 1.)
        return ret


def test_static_proposal_stats():
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    rstate=rstate,
                                    sample='rwalk',
                                    blob=False) # blob=False to isolate proposal_stats
    sampler.run_nested(print_progress=printing, maxiter=100)
    res = sampler.results
    assert 'proposal_stats' in res
    assert res['proposal_stats'] is not None
    assert len(res['proposal_stats']) == len(res.samples)
    assert any(ps is not None for ps in res['proposal_stats'])
    for ps in res['proposal_stats']:
        if ps is not None:
            assert isinstance(ps, dict)
            assert 'n_proposals' in ps


def test_dynamic_proposal_stats():
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.DynamicNestedSampler(g.loglikelihood,
                                           g.prior_transform,
                                           g.ndim,
                                           nlive=nlive,
                                           rstate=rstate,
                                           sample='unif',
                                           blob=False) # blob=False to isolate proposal_stats
    sampler.run_nested(print_progress=printing, dlogz_init=1, maxiter_init=100)
    res = sampler.results
    assert 'proposal_stats' in res
    assert res['proposal_stats'] is not None
    assert len(res['proposal_stats']) == len(res.samples)
    assert any(ps is not None for ps in res['proposal_stats'])
    for ps in res['proposal_stats']:
        if ps is not None:
            assert isinstance(ps, dict)
            assert 'n_proposals' in ps


def test_proposal_stats_length_consistency():
    """Test that proposal_stats array has same length as samples array"""
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive//10,  # smaller for faster test
                                    rstate=rstate,
                                    sample='rwalk',
                                    blob=False)
    sampler.run_nested(print_progress=printing, maxiter=50)
    res = sampler.results
    
    # Check length consistency
    assert len(res['proposal_stats']) == len(res.samples)


def test_proposal_stats_different_samplers():
    """Test proposal_stats with different sampling methods"""
    rstate = get_rstate()
    g = Gaussian()
    
    # Test different sampling methods
    sampling_methods = ['unif', 'rwalk', 'slice', 'rslice']
    
    for sample_method in sampling_methods:
        sampler = dynesty.NestedSampler(g.loglikelihood,
                                        g.prior_transform,
                                        g.ndim,
                                        nlive=nlive//10,  # smaller for faster test
                                        rstate=rstate,
                                        sample=sample_method,
                                        blob=False)
        sampler.run_nested(print_progress=printing, maxiter=30)
        res = sampler.results
        
        # Basic checks
        assert 'proposal_stats' in res
        assert res['proposal_stats'] is not None
        assert len(res['proposal_stats']) == len(res['logl'])
        
        # Check that we have some non-None proposal_stats
        non_none_count = sum(1 for ps in res['proposal_stats'] if ps is not None)
        assert non_none_count > 0, f"No proposal_stats found for {sample_method}"
        
        # Check structure of non-None entries
        for ps in res['proposal_stats']:
            if ps is not None:
                assert isinstance(ps, dict)
                assert 'n_proposals' in ps
                assert isinstance(ps['n_proposals'], (int, np.integer))
                assert ps['n_proposals'] > 0


def test_proposal_stats_nested_vs_dynamic():
    """Test proposal_stats consistency between nested and dynamic nested samplers"""
    rstate = get_rstate()
    g = Gaussian()
    
    # Test with nested sampler
    sampler_nested = dynesty.NestedSampler(g.loglikelihood,
                                          g.prior_transform,
                                          g.ndim,
                                          nlive=nlive//10,
                                          rstate=rstate,
                                          sample='rwalk',
                                          blob=False)
    sampler_nested.run_nested(print_progress=printing, maxiter=30)
    res_nested = sampler_nested.results
    
    # Test with dynamic nested sampler
    rstate = get_rstate()  # Reset to same state
    sampler_dynamic = dynesty.DynamicNestedSampler(g.loglikelihood,
                                                  g.prior_transform,
                                                  g.ndim,
                                                  nlive=nlive//10,
                                                  rstate=rstate,
                                                  sample='rwalk',
                                                  blob=False)
    sampler_dynamic.run_nested(print_progress=printing, 
                              dlogz_init=1, 
                              maxiter_init=30)
    res_dynamic = sampler_dynamic.results
    
    # Both should have proposal_stats
    assert 'proposal_stats' in res_nested
    assert 'proposal_stats' in res_dynamic
    assert res_nested['proposal_stats'] is not None
    assert res_dynamic['proposal_stats'] is not None
    
    # Both should have consistent lengths
    assert len(res_nested['proposal_stats']) == len(res_nested['logl'])
    assert len(res_dynamic['proposal_stats']) == len(res_dynamic['logl'])
    
    # Both should have some non-None proposal_stats
    assert any(ps is not None for ps in res_nested['proposal_stats'])
    assert any(ps is not None for ps in res_dynamic['proposal_stats'])


def test_proposal_stats_content_validity():
    """Test that proposal_stats contain valid data structures and values"""
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive//10,
                                    rstate=rstate,
                                    sample='rwalk',
                                    blob=False)
    sampler.run_nested(print_progress=printing, maxiter=30)
    res = sampler.results
    
    # Check content of proposal_stats
    for i, ps in enumerate(res['proposal_stats']):
        if ps is not None:
            # Should be a dictionary
            assert isinstance(ps, dict), f"proposal_stats[{i}] is not a dict"
            
            # Should contain n_proposals
            assert 'n_proposals' in ps, f"proposal_stats[{i}] missing 'n_proposals'"
            
            # n_proposals should be a positive integer
            n_prop = ps['n_proposals']
            assert isinstance(n_prop, (int, np.integer)), f"n_proposals is not an integer at index {i}"
            assert n_prop > 0, f"n_proposals should be positive at index {i}, got {n_prop}"
            
            # If there are other keys, they should have reasonable values
            for key, value in ps.items():
                if key != 'n_proposals':
                    # Any additional stats should be numeric and finite
                    if isinstance(value, (int, float, np.number)):
                        assert np.isfinite(value), f"Non-finite value in proposal_stats[{i}]['{key}']"


def test_proposal_stats_with_blob():
    """Test that proposal_stats work correctly when blob is also enabled"""
    def loglikelihood_with_blob(x):
        logl = -0.5 * np.sum(x**2)
        blob = {'param_sum': np.sum(x), 'param_max': np.max(x)}
        return logl, blob
    
    def prior_transform(u):
        return 4 * (u - 0.5)  # uniform from -2 to 2
    
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_with_blob,
                                    prior_transform,
                                    3,  # ndim
                                    nlive=nlive//10,
                                    rstate=rstate,
                                    sample='rwalk',
                                    blob=True)
    sampler.run_nested(print_progress=printing, maxiter=30)
    res = sampler.results
    
    # Should have both blob and proposal_stats
    assert 'blob' in res
    assert 'proposal_stats' in res
    assert res['blob'] is not None
    assert res['proposal_stats'] is not None
    
    # Both arrays should have same length as samples
    assert len(res['blob']) == len(res.samples)
    assert len(res['proposal_stats']) == len(res.samples)
    assert len(res['proposal_stats']) == len(res['logl'])
    
    # Check that both contain valid data
    assert any(b is not None for b in res['blob'])
    assert any(ps is not None for ps in res['proposal_stats'])
