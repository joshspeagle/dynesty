"""Static conversion must distinguish dead points from final live points."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from dynesty import NestedSampler
from dynesty import utils as dyutil
from dynesty import plotting as dyplot
from utils import get_rstate


@pytest.mark.parametrize('add_live', [False, True])
@pytest.mark.parametrize('bootstrap', [False, True])
def test_merge_single_partial_niter(add_live, bootstrap):
    sampler = NestedSampler(lambda x: -.5 * np.dot(x, x),
                             lambda u: 20 * u - 10, 1,
                             nlive=30, rstate=get_rstate())
    with pytest.warns(UserWarning, match='stopped short'):
        sampler.run_nested(maxiter=5, add_live=add_live,
                            print_progress=False)
    res = sampler.results
    if bootstrap:
        res = dyutil.resample_run(res, rstate=get_rstate())
    merged = dyutil.merge_runs([res], print_progress=False)
    assert not merged.isdynamic()
    if not add_live:
        assert merged.niter == len(res.logl)
        assert merged.nlive == 30
    else:
        assert len(merged.logl) == merged.niter + merged.nlive
    fig, _ = dyplot.runplot(merged, kde=False)
    plt.close(fig)


def test_merge_bootstrapped_single_strand():
    # A completed one-live-point run has samples_n == 1 everywhere.
    # Both static-form checks match, but its final point still counts as live.
    sampler = NestedSampler(lambda x: -.5 * np.dot(x, x),
                             lambda u: 20 * u - 10, 1,
                             nlive=30, rstate=get_rstate())
    sampler.run_nested(dlogz=.1, print_progress=False)
    strand = dyutil.unravel_run(sampler.results, print_progress=False)[0]
    boot = dyutil.resample_run(strand, rstate=get_rstate())
    merged = dyutil.merge_runs([boot], print_progress=False)
    assert merged.nlive == 1
    assert merged.niter == len(merged.logl) - 1
