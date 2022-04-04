import numpy as np

ELLIPSOID_PROPOSALS = {"normal", "volumetric", "axis", "chi"}
ENSEMBLE_PROPOSALS = {"diff", "stretch", "walk", "snooker"}
DEFAULT_PROPOSALS = {"diff", "stretch", "normal"}


def register_proposal(name, proposal, kind=None):
    f"""
    Add a new proposal function to the known set.

    Parameters
    ----------
    name: str
        Name of the proposal, this is what is passed to :code:`dynesty`,
        e.g., :code:`diff`
    proposal: callable
        The callable that takes in the standard proposal inputs
        :code:`u, rstate, axis, live, scale` and returns either
        proposed point and possibly a natural log Jacobian associated with the
        step.
    kind: str, optional
        Either :code:`ellipsoid`, :code:`ensemble` or :code:`None`. This tells
        the sampler whether bounding ellipsoids need to be computed for the
        proposal or the current live points need to be passed respectively.

    Raises
    ------
    ValueError
        A value error is raised if the new proposal name already exists in the
        known registered proposals.
        The pre-defined proposal names are {", ".join(_proposal_map.keys())}.
    """
    if name in _proposal_map:
        raise ValueError(f"Proposal {name} already exists.")
    if kind == "ellipsoid":
        ELLIPSOID_PROPOSALS.add(name)
    elif kind == "ensemble":
        ENSEMBLE_PROPOSALS.add(name)
    _proposal_map[name] = proposal


_acceptances = dict()
_failures = dict()


def propose_diff_evo(u, live, rstate, **kwargs):
    r"""
    Propose a new point using ensemble differential evolution.

    .. math::

        u_{\rm prop} = u + \gamma (v_{a} - v_{b})

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    rstate: RandomState
        The random state to use to generate random numbers.
    kwargs: dict, unused
        Required for flexibility of the interface

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    nlive, n = live.shape
    first, second = rstate.choice(nlive, 2, replace=False)
    diff = live[second] - live[first]
    if rstate.uniform(0, 1) < 0.5:
        diff *= 2.38 / n**0.5
        diff *= (100**rstate.uniform(0, 1)) / 10
    u_prop = u + diff
    return u_prop


def propose_ensemble_snooker(u, live, rstate, **kwargs):
    r"""
    Propose a new point using ensemble differential evolution.

    .. math::

        u_{\rm prop} = u + \gamma (v_{a} - v_{b})

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    rstate: RandomState
        The random state to use to generate random numbers.
    kwargs: dict, unused
        Required for flexibility of the interface

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    ln_jacobian: float
        The natural log transition asymmetry for the proposed step.
    """
    nlive, n = live.shape
    choices = rstate.choice(nlive, 3, replace=False)
    z = live[choices[0]]
    z1 = live[choices[1]]
    z2 = live[choices[2]]
    delta = u - z
    norm = np.linalg.norm(delta)
    delta /= norm
    u_prop = u + 1.7 * delta * (np.dot(u, z1) - np.dot(u, z2))
    ln_jacobian = (n - 1.0) * np.log(np.linalg.norm(u_prop - z) / norm)
    return u_prop, ln_jacobian


def propose_ensemble_stretch(u, live, rstate, **kwargs):
    r"""
    Propose a new point using ensemble differential evolution.

    .. math::

        u_{\rm prop} = v + \gamma (u - v)

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    rstate: RandomState
        The random state to use to generate random numbers.
    kwargs: dict, unused
        Required for flexibility of the interface

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    ln_jacobian: float
        The natural log transition asymmetry for the proposed step.
    """
    nlive, n = live.shape
    max_scale = 3
    scale = ((max_scale - 1.0) * rstate.uniform() + 1) ** 2.0 / max_scale

    other = rstate.choice(nlive)
    other = live[other]

    u_prop = other + scale * (u - other)
    ln_jacobian = np.log(scale) * (n - 1)
    return u_prop, ln_jacobian


def propose_ensemble_walk(u, live, rstate, nsamples=3, **kwargs):
    r"""
    Propose a new point using ensemble differential evolution.

    .. math::

        u_{\rm prop} = u + \frac{1}{N_{\rm samples}}
        \sum^{N_{\rm samples}}_{i=1} \gamma_{i} (v_{i} - \hat{v})

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    rstate: RandomState
        The random state to use to generate random numbers.
    kwargs: dict, unused
        Required for flexibility of the interface

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    nlive, n = live.shape
    choices = rstate.choice(nlive, nsamples, replace=False)
    center_of_mass = np.mean(live[choices], axis=0)
    scales = rstate.normal(0, 1, nsamples)[:, np.newaxis]
    diff = np.sum(scales * (live[choices] - center_of_mass), axis=0)
    return u + diff


def propose_along_axis(u, axes, rstate, scale=1, **kwargs):
    """
    Propose a new point along one of the principal axes of the bounding
    ellipsoid.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    axes: np.ndarray
        The principal component decomposition of the bounding ellipsoid.
    rstate: RandomState
        The random state to use to generate random numbers.
    scale: float
        The scale of the step to take.
    kwargs: dict, unused
        Required for flexibility of the interface.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    n = len(u)
    idx = rstate.choice(n)
    axis = axes[idx]
    scale *= rstate.normal(0, 1)
    return u + axis * scale


def propose_chi(u, axes, rstate, scale=1, **kwargs):
    """
    Propose a new point drawn using the principal axes of the bounding
    ellipsoid. The scale of the proposal is drawn from a chi distribution
    with n_dim degrees of freedom.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    axes: np.ndarray
        The principal component decomposition of the bounding ellipsoid.
    rstate: RandomState
        The random state to use to generate random numbers.
    scale: float
        The scale of the step to take.
    kwargs: dict, unused
        Required for flexibility of the interface.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    scale *= rstate.chisquare(len(u)) / len(u)
    return _axes_proposal(u=u, axes=axes, rstate=rstate, scale=scale)


def propose_normal(u, axes, rstate, scale=1, **kwargs):
    """
    Propose a new point drawn using the principal axes of the bounding
    ellipsoid. The scale of the proposal is drawn from a unit normal
    distribution.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    axes: np.ndarray
        The principal component decomposition of the bounding ellipsoid.
    rstate: RandomState
        The random state to use to generate random numbers.
    scale: float
        The scale of the step to take.
    kwargs: dict, unused
        Required for flexibility of the interface.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    scale *= rstate.normal(0, 1)
    return _axes_proposal(u=u, axes=axes, rstate=rstate, scale=scale)


def propose_volumetric(u, axes, rstate, scale=1, **kwargs):
    """
    Propose a new point drawn using the principal axes of the bounding
    ellipsoid. The scale of the proposal is drawn uniformly from the ellipsoid.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    axes: np.ndarray
        The principal component decomposition of the bounding ellipsoid.
    rstate: RandomState
        The random state to use to generate random numbers.
    scale: float
        The scale of the step to take.
    kwargs: dict, unused
        Required for flexibility of the interface.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    n = len(u)
    scale *= rstate.uniform(0, 1) ** (1.0 / n)
    return _axes_proposal(u=u, axes=axes, rstate=rstate, scale=scale)


def _axes_proposal(u, axes, rstate, scale):
    n = len(u)
    drhat = rstate.normal(0, 1, n)
    drhat /= np.linalg.norm(drhat)
    du = np.dot(axes, drhat)
    return u + scale * du


_proposal_map = dict(
    diff=propose_diff_evo,
    stretch=propose_ensemble_stretch,
    walk=propose_ensemble_walk,
    snooker=propose_ensemble_snooker,
    axis=propose_along_axis,
    volumetric=propose_volumetric,
    chi=propose_chi,
    normal=propose_normal,
)
