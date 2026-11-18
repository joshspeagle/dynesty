"""Merging independent runs must retain independent particle strands."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import utils as dyutil
from utils import get_rstate


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('nruns', [2, 5])
def test_merge_independent_strands(dynamic, nruns):
    rstate = get_rstate()
    runs = []
    for _ in range(nruns):
        cls = DynamicNestedSampler if dynamic else NestedSampler
        sampler = cls(lambda x: -.5 * np.dot(x, x),
                       lambda u: 20 * u - 10, 1, nlive=20, rstate=rstate)
        kwargs = dict(maxbatch=0) if dynamic else {}
        sampler.run_nested(print_progress=False, **kwargs)
        runs.append(sampler.results)
    original_ids = [r.samples_id.copy() for r in runs]
    merged = dyutil.merge_runs(runs, print_progress=False)
    assert len(np.unique(merged.samples_id)) == 20 * nruns
    assert_array_equal(merged.batch_nlive, [20 * nruns])
    # Each merged strand must contain exactly one original strand.
    for run in runs:
        for idx in np.unique(run.samples_id):
            original = run.samples[run.samples_id == idx, 0]
            rows = np.isin(merged.samples[:, 0], original)
            ids = np.unique(merged.samples_id[rows])
            assert len(ids) == 1
            assert_array_equal(merged.samples[merged.samples_id == ids[0], 0],
                               original)
    for run, ids in zip(runs, original_ids):
        assert_array_equal(run.samples_id, ids)
    # Bootstrapping the merged run must still have one strand per live slot.
    boot = dyutil.resample_run(merged, rstate=get_rstate())
    assert boot.samples_n.max() == 20 * nruns
