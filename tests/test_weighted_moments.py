"""Weighted posterior moments must not depend on the units of the weights."""

import numpy as np
import pytest

from dynesty.utils import mean_and_cov


@pytest.mark.parametrize("dtype,scale", [
    (np.float32, 1.0), (np.float32, 1e-30), (np.float32, 1e30),
    (np.float64, 1.0), (np.float64, 1e-200), (np.float64, 1e200),
    (np.int64, 1), (np.int64, 2**32), (np.int64, 2**60),
])
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_weight_scale_invariance(dtype, scale, ndim, zero_weight):
    samples = np.array([[1., 2.], [3., 0.], [5., 4.], [0., -1.]])[:, :ndim]
    reference_weights = np.array([1., 2., 3., 4.])
    if zero_weight:
        reference_weights[0] = 0
    weights = reference_weights.astype(dtype) * scale
    original_weights = weights.copy()
    original_samples = samples.copy()
    expected_mean = np.average(samples, axis=0, weights=reference_weights)
    expected_cov = np.atleast_2d(np.cov(samples, rowvar=False, aweights=reference_weights))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        mean, cov = mean_and_cov(samples, weights)
    assert np.isfinite(mean).all()
    assert np.isfinite(cov).all()
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(weights, original_weights)
    np.testing.assert_array_equal(samples, original_samples)


@pytest.mark.parametrize("nsamples", [0, 3])
def test_zero_total_weight_remains_an_error(nsamples):
    samples = np.zeros((nsamples, 2))
    with pytest.raises(ZeroDivisionError):
        mean_and_cov(samples, np.zeros(nsamples))
