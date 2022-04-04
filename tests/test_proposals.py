import itertools

import numpy as np
import dynesty
from dynesty.proposals import register_proposal, _proposal_map, ELLIPSOID_PROPOSALS, ENSEMBLE_PROPOSALS
import pytest
from scipy.special import erf
from utils import get_rstate, get_printing

nlive = 100
printing = get_printing()
win = 100
ndim = 2


def loglike(x):
    return -0.5 * x[1]**2


def prior_transform(x):
    return (2 * x - 1) * win


def dummy_proposal(x):
    return x + 1


@pytest.mark.parametrize(
    "proposal,dynamic",
    itertools.product(
        ["diff", "stretch", "walk", "snooker", "axis", "volumetric", "chi", "normal"],
        [True, False]
    )
)
def test_proposals(proposal, dynamic):
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    logz_true = np.log(np.sqrt(2 * np.pi) * erf(win / np.sqrt(2)) / (2 * win))
    sampler = "rwalk"
    rstate = get_rstate()
    if dynamic:
        dns = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           periodic=[0],
                                           rstate=rstate,
                                           proposals=[proposal],
                                           sample=sampler)
        dns.run_nested(dlogz_init=1, print_progress=printing)
    else:
        dns = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    periodic=[0],
                                    rstate=rstate,
                                    proposals=[proposal],
                                    sample=sampler)
        dns.run_nested(dlogz=1, print_progress=printing)
    assert (np.abs(logz_true - dns.results.logz[-1]) <
            5. * dns.results.logzerr[-1])


@pytest.mark.parametrize(
    "name,kind", [("ellipse", "ellipsoid"), ("ensemble", "ensemble"), ("neither", None)]
)
def test_register_proposal(name, kind):
    register_proposal(name, proposal=dummy_proposal, kind=kind)
    assert _proposal_map[name](1) == 2
    ellipse = False
    ensemble = False
    if kind == "ellipsoid":
        ellipse = True
        assert name in ELLIPSOID_PROPOSALS
    elif kind == "ensemble":
        ensemble = True
    assert ellipse == (name in ELLIPSOID_PROPOSALS)
    assert ensemble == (name in ENSEMBLE_PROPOSALS)
