"""Integer sample weights have the same CDF as their floating equivalents."""

import warnings

import numpy as np
import pytest

from dynesty.utils import quantile, resample_equal
from utils import get_rstate, NullContextManager

WEIGHT_TYPES = [
    (np.int16, 1), (np.int32, 1), (np.int64, 1), (np.uint64, 1),
    (np.int64, 2**60), (np.float32, 1), (np.float64, 1),
]


@pytest.mark.parametrize("dtype,scale", WEIGHT_TYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_quantile_weight_dtype(dtype, scale, reverse):
    samples = np.array([0., 10., 20., 30.])
    weights = np.array([4, 3, 2, 1], dtype=dtype) * scale
    if reverse:
        samples = samples[::-1]
        weights = weights[::-1]
    original_weights = weights.copy()
    levels = [0., 0.25, 0.5, 0.75, 1.]
    expected = quantile(samples, levels, weights=weights.astype(np.float64))
    actual = quantile(samples, levels, weights=weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("dtype,scale", WEIGHT_TYPES)
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("one_hot", [False, True])
@pytest.mark.parametrize("seed", [56432, 991])
def test_resampling_weight_dtype(dtype, scale, ndim, one_hot, seed):
    samples = np.arange(4)
    if ndim == 2:
        samples = np.column_stack([samples, samples + 10])
    raw_weights = [0, 1, 0, 0] if one_hot else [4, 3, 2, 1]
    weights = np.array(raw_weights, dtype=dtype) * (1 if one_hot else scale)
    original_weights = weights.copy()
    normalized = weights.astype(np.float64)
    normalized /= normalized.sum()
    expected = resample_equal(samples, normalized, rstate=get_rstate(seed))
    context = (NullContextManager(None) if one_hot else
               pytest.warns(UserWarning, match="renormalized"))
    with context:
        actual = resample_equal(samples, weights, rstate=get_rstate(seed))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(weights, original_weights)
    assert actual.dtype == samples.dtype
    if one_hot:
        np.testing.assert_array_equal(actual, np.repeat(samples[1:2], 4, axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalized_floating_weights_do_not_warn(dtype):
    samples = np.arange(4)
    weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        resample_equal(samples, weights, rstate=get_rstate())
    assert not caught
