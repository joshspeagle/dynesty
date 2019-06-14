#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The top-level interface (defined natively upon initialization) that
provides access to the two main sampler "super-classes" via
:meth:`NestedSampler` and :meth:`DynamicNestedSampler`.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

from .nestedsamplers import (UnitCubeSampler, SingleEllipsoidSampler,
                             MultiEllipsoidSampler, RadFriendsSampler,
                             SupFriendsSampler)
from .sampling import (sample_unif, sample_rwalk, sample_rstagger,
                       sample_slice, sample_rslice, sample_hslice)
from .dynamicsampler import DynamicSampler

__all__ = ["NestedSampler", "DynamicNestedSampler", "_function_wrapper"]

_SAMPLERS = {'none': UnitCubeSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler,
             'balls': RadFriendsSampler,
             'cubes': SupFriendsSampler}
_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'rstagger': sample_rstagger,
             'slice': sample_slice,
             'rslice': sample_rslice,
             'hslice': sample_hslice}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def NestedSampler(loglikelihood, prior_transform, ndim, nlive=500,
                  bound='multi', sample='auto', periodic=None,
                  update_interval=None, first_update=None,
                  npdim=None, rstate=None, queue_size=None, pool=None,
                  use_pool=None, live_points=None,
                  logl_args=None, logl_kwargs=None,
                  ptform_args=None, ptform_kwargs=None,
                  gradient=None, grad_args=None, grad_kwargs=None,
                  compute_jac=False,
                  enlarge=None, bootstrap=0, vol_dec=0.5, vol_check=2.0,
                  walks=25, facc=0.5, slices=5, fmove=0.9, max_move=100,
                  **kwargs):
    """
    Initializes and returns a sampler object for Static Nested Sampling.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d `~numpy` array with length `ndim`, where
        each value is in the range [0, 1). The return value should also be a
        1-d `~numpy` array with length `ndim`, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be::

            def prior_transform(u):
                return 2.0 * u

    ndim : int
        Number of parameters returned by `prior_transform` and accepted by
        `loglikelihood`.

    nlive : int, optional
        Number of "live" points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is `500`.

    bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}, optional
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to
        propose new live points. Choices are no bound (`'none'`), a single
        bounding ellipsoid (`'single'`), multiple bounding ellipsoids
        (`'multi'`), balls centered on each live point (`'balls'`), and
        cubes centered on each live point (`'cubes'`). Default is `'multi'`.

    sample : {`'auto'`, `'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Unique methods available are:
        uniform sampling within the bounds(`'unif'`),
        random walks with fixed proposals (`'rwalk'`),
        random walks with variable ("staggering") proposals (`'rstagger'`),
        multivariate slice sampling along preferred orientations (`'slice'`),
        "random" slice sampling along all orientations (`'rslice'`), and
        "Hamiltonian" slices along random trajectories (`'hslice'`).
        `'auto'` selects the sampling method based on the dimensionality
        of the problem (from `ndim`).
        When `ndim < 10`, this defaults to `'unif'`.
        When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
        When `ndim > 20`, this defaults to `'hslice'` if a `gradient` is
        provided and `'slice'` otherwise. `'rstagger'` and `'rslice'`
        are provided as alternatives for `'rwalk'` and `'slice'`, respectively.
        Default is `'auto'`.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. It is assumed that their periodicity
        is dealt with in the `prior_transform` and/or `loglikelihood`
        functions. Default is `None` (i.e. no periodic boundary conditions).

    update_interval : int or float, optional
        If an integer is passed, only update the proposal distribution every
        `update_interval`-th likelihood call. If a float is passed, update the
        proposal after every `round(update_interval * nlive)`-th likelihood
        call. Larger update intervals larger can be more efficient
        when the likelihood function is quick to evaluate. Default behavior
        is to target a roughly constant change in prior volume, with
        `1.5` for `'unif'`, `0.15 * walks` for `'rwalk'` and `'rstagger'`,
        `0.9 * ndim * slices` for `'slice'`, `2.0 * slices` for `'rslice'`,
        and `25.0 * slices` for `'hslice'`.

    first_update : dict, optional
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube (`'none'`)
        to the one specified by `sample`. Options are the minimum number of
        likelihood calls (`'min_ncall'`) and the minimum allowed overall
        efficiency in percent (`'min_eff'`). Defaults are `2 * nlive` and
        `10.`, respectively.

    npdim : int, optional
        Number of parameters accepted by `prior_transform`. This might differ
        from `ndim` in the case where a parameter of loglikelihood is dependent
        upon multiple independently distributed parameters, some of which may
        be nuisance parameters.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    queue_size : int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) `queue_size` many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated. If no value is passed, this defaults to `pool.size` (if
        a `pool` has been provided) and `1` otherwise (no parallelism).

    pool : user-provided pool, optional
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where a pool should be used to
        execute operations in parallel. These govern whether `prior_transform`
        is executed in parallel during initialization (`'prior_transform'`),
        `loglikelihood` is executed in parallel during initialization
        (`'loglikelihood'`), live points are proposed in parallel during a run
        (`'propose_point'`), and bounding distributions are updated in
        parallel during a run (`'update_bound'`). Default is `True` for all
        options.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        A set of live points used to initialize the nested sampling run.
        Contains `live_u`, the coordinates on the unit cube, `live_v`, the
        transformed variables, and `live_logl`, the associated loglikelihoods.
        By default, if these are not provided the initial set of live points
        will be drawn uniformly from the unit `npdim`-cube.
        **WARNING: It is crucial that the initial set of live points have been
        sampled from the prior. Failure to provide a set of valid live points
        will result in incorrect results.**

    logl_args : iterable, optional
        Additional arguments that can be passed to `loglikelihood`.

    logl_kwargs : dict, optional
        Additional keyword arguments that can be passed to `loglikelihood`.

    ptform_args : iterable, optional
        Additional arguments that can be passed to `prior_transform`.

    ptform_kwargs : dict, optional
        Additional keyword arguments that can be passed to `prior_transform`.

    gradient : function, optional
        A function which returns the gradient corresponding to
        the provided `loglikelihood` *with respect to the unit cube*.
        If provided, this will be used when computing reflections
        when sampling with `'hslice'`. If not provided, gradients are
        approximated numerically using 2-sided differencing.

    grad_args : iterable, optional
        Additional arguments that can be passed to `gradient`.

    grad_kwargs : dict, optional
        Additional keyword arguments that can be passed to `gradient`.

    compute_jac : bool, optional
        Whether to compute and apply the Jacobian `dv/du`
        from the target space `v` to the unit cube `u` when evaluating the
        `gradient`. If `False`, the gradient provided is assumed to be
        already defined with respect to the unit cube. If `True`, the gradient
        provided is assumed to be defined with respect to the target space
        so the Jacobian needs to be numerically computed and applied. Default
        is `False`.

    enlarge : float, optional
        Enlarge the volumes of the specified bounding object(s) by this
        fraction. The preferred method is to determine this organically
        using bootstrapping. If `bootstrap > 0`, this defaults to `1.0`.
        If `bootstrap = 0`, this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes. Can
        lead to unstable bounding ellipsoids. Default is `0` (no bootstrap).

    vol_dec : float, optional
        For the `'multi'` bounding option, the required fractional reduction
        in volume after splitting an ellipsoid in order to to accept the split.
        Default is `0.5`.

    vol_check : float, optional
        For the `'multi'` bounding option, the factor used when checking if
        the volume of the original bounding ellipsoid is large enough to
        warrant `> 2` splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is `2.0`.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling
        options, the number of times to execute a "slice update"
        before proposing a new live point. Default is `5`.
        Note that `'slice'` cycles through **all dimensions**
        when executing a "slice update".

    fmove : float, optional
        The target fraction of samples that are proposed along a trajectory
        (i.e. not reflecting) for the `'hslice'` sampling option.
        Default is `0.9`.

    max_move : int, optional
        The maximum number of timesteps allowed for `'hslice'`
        per proposal forwards and backwards in time. Default is `100`.

    Returns
    -------
    sampler : sampler from :mod:`~dynesty.nestedsamplers`
        An initialized instance of the chosen sampler specified via `bound`.

    """

    # Prior dimensions.
    if npdim is None:
        npdim = ndim

    # Bounding method.
    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    # Sampling method.
    if sample == 'auto':
        if npdim < 10:
            sample = 'unif'
        elif 10 <= npdim <= 20:
            sample = 'rwalk'
        else:
            if gradient is None:
                sample = 'slice'
            else:
                sample = 'hslice'
    if sample not in _SAMPLING:
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    # Dimensional warning check.
    if nlive <= 2 * ndim:
        warnings.warn("Beware! Having `nlive <= 2 * ndim` is extremely risky!")
    elif nlive < ndim * (ndim + 1) // 2 and bound in ['single', 'multi']:
        warnings.warn("A note of caution: "
                      "having `nlive < ndim * (ndim + 1) // 2` may result in "
                      "unconstrained bounding distributions.")

    # Gather non-periodic boundary conditions.
    if periodic is not None:
        nonperiodic = np.ones(npdim, dtype='bool')
        nonperiodic[periodic] = False
    else:
        nonperiodic = None
    kwargs['nonperiodic'] = nonperiodic

    # Update interval for bounds.
    if update_interval is None:
        if sample == 'unif':
            update_interval = 1.5
        elif sample == 'rwalk' or sample == 'rstagger':
            update_interval = 0.15 * walks
        elif sample == 'slice':
            update_interval = 0.9 * npdim * slices
        elif sample == 'rslice':
            update_interval = 2.0 * slices
        elif sample == 'hslice':
            update_interval = 25.0 * slices
        else:
            raise ValueError("Unknown sampling method: '{0}'".format(sample))
    if bound == 'none':
        update_interval = sys.maxsize  # no need to update with no bounds
    if isinstance(update_interval, float):
        update_interval = max(1, round(update_interval * nlive))

    # Keyword arguments controlling the first update.
    if first_update is None:
        first_update = dict()

    # Random state.
    if rstate is None:
        rstate = np.random

    # Log-likelihood.
    if logl_args is None:
        logl_args = []
    if logl_kwargs is None:
        logl_kwargs = {}

    # Prior transform.
    if ptform_args is None:
        ptform_args = []
    if ptform_kwargs is None:
        ptform_kwargs = {}

    # gradient
    if grad_args is None:
        grad_args = []
    if grad_kwargs is None:
        grad_kwargs = {}

    # Bounding distribution modifications.
    if enlarge is not None:
        kwargs['enlarge'] = enlarge
    if bootstrap is not None:
        kwargs['bootstrap'] = bootstrap
    if vol_dec is not None:
        kwargs['vol_dec'] = vol_dec
    if vol_check is not None:
        kwargs['vol_check'] = vol_check

    # Sampling.
    if walks is not None:
        kwargs['walks'] = walks
    if facc is not None:
        kwargs['facc'] = facc
    if slices is not None:
        kwargs['slices'] = slices
    if fmove is not None:
        kwargs['fmove'] = fmove
    if max_move is not None:
        kwargs['max_move'] = max_move

    # Set up parallel (or serial) evaluation.
    if queue_size is not None and queue_size < 1:
        raise ValueError("The queue must contain at least one element!")
    elif (queue_size == 1) or (pool is None and queue_size is None):
        M = map
        queue_size = 1
    elif pool is not None:
        M = pool.map
        if queue_size is None:
            try:
                queue_size = pool.size
            except:
                raise ValueError("Cannot initialize `queue_size` because "
                                 "`pool.size` has not been provided. Please"
                                 "define `pool.size` or specify `queue_size` "
                                 "explicitly.")
    else:
        raise ValueError("`queue_size > 1` but no `pool` provided.")
    if use_pool is None:
        use_pool = dict()

    # Wrap functions.
    ptform = _function_wrapper(prior_transform, ptform_args, ptform_kwargs,
                               name='prior_transform')
    loglike = _function_wrapper(loglikelihood, logl_args, logl_kwargs,
                                name='loglikelihood')

    # Add in gradient.
    if gradient is not None:
        grad = _function_wrapper(gradient, grad_args, grad_kwargs,
                                 name='gradient')
        kwargs['grad'] = grad
        kwargs['compute_jac'] = compute_jac

    # Initialize live points and calculate log-likelihoods.
    if live_points is None:
        live_u = rstate.rand(nlive, npdim)  # positions in unit cube
        if use_pool.get('prior_transform', True):
            live_v = np.array(list(M(ptform,
                                     np.array(live_u))))  # real parameters
        else:
            live_v = np.array(list(map(ptform,
                                       np.array(live_u))))
        if use_pool.get('loglikelihood', True):
            live_logl = np.array(list(M(loglike,
                                        np.array(live_v))))  # log likelihood
        else:
            live_logl = np.array(list(map(loglike,
                                          np.array(live_v))))
        live_points = [live_u, live_v, live_logl]

    # Convert all `-np.inf` log-likelihoods to finite large numbers.
    # Necessary to keep estimators in our sampler from breaking.
    for i, logl in enumerate(live_points[2]):
        if not np.isfinite(logl):
            if np.sign(logl) < 0:
                live_points[2][i] = -1e300
            else:
                raise ValueError("The log-likelihood ({0}) of live point {1} "
                                 "located at u={2} v={3} is invalid."
                                 .format(logl, i, live_points[0][i],
                                         live_points[1][i]))

    # Initialize our nested sampler.
    sampler = _SAMPLERS[bound](loglike, ptform, npdim,
                               live_points, sample, update_interval,
                               first_update, rstate, queue_size, pool,
                               use_pool, kwargs)

    return sampler


def DynamicNestedSampler(loglikelihood, prior_transform, ndim,
                         bound='multi', sample='auto', periodic=None,
                         update_interval=None, first_update=None,
                         npdim=None, rstate=None, queue_size=None, pool=None,
                         use_pool=None, logl_args=None, logl_kwargs=None,
                         ptform_args=None, ptform_kwargs=None,
                         gradient=None, grad_args=None, grad_kwargs=None,
                         compute_jac=False,
                         enlarge=None, bootstrap=0,
                         vol_dec=0.5, vol_check=2.0,
                         walks=25, facc=0.5,
                         slices=5, fmove=0.9, max_move=100,
                         **kwargs):
    """
    Initializes and returns a sampler object for Dynamic Nested Sampling.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d `~numpy` array with length `ndim`, where
        each value is in the range [0, 1). The return value should also be a
        1-d `~numpy` array with length `ndim`, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be::

            def prior_transform(u):
                return 2.0 * u

    ndim : int
        Number of parameters returned by `prior_transform` and accepted by
        `loglikelihood`.

    bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}, optional
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to
        propose new live points. Choices are no bound (`'none'`), a single
        bounding ellipsoid (`'single'`), multiple bounding ellipsoids
        (`'multi'`), balls centered on each live point (`'balls'`), and
        cubes centered on each live point (`'cubes'`). Default is `'multi'`.

    sample : {`'auto'`, `'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Unique methods available are:
        uniform sampling within the bounds(`'unif'`),
        random walks with fixed proposals (`'rwalk'`),
        random walks with variable ("staggering") proposals (`'rstagger'`),
        multivariate slice sampling along preferred orientations (`'slice'`),
        "random" slice sampling along all orientations (`'rslice'`), and
        "Hamiltonian" slices along random trajectories (`'hslice'`).
        `'auto'` selects the sampling method based on the dimensionality
        of the problem (from `ndim`).
        When `ndim < 10`, this defaults to `'unif'`.
        When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
        When `ndim > 20`, this defaults to `'hslice'` if a `gradient` is
        provided and `'slice'` otherwise. `'rstagger'` and `'rslice'`
        are provided as alternatives for `'rwalk'` and `'slice'`, respectively.
        Default is `'auto'`.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. It is assumed that their periodicity
        is dealt with in the `prior_transform` and/or `loglikelihood`
        functions. Default is `None` (i.e. no periodic boundary conditions).

    update_interval : int or float, optional
        If an integer is passed, only update the proposal distribution every
        `update_interval`-th likelihood call. If a float is passed, update the
        proposal after every `round(update_interval * nlive)`-th likelihood
        call. Larger update intervals larger can be more efficient
        when the likelihood function is quick to evaluate. Default behavior
        is to target a roughly constant change in prior volume, with
        `1.5` for `'unif'`, `0.15 * walks` for `'rwalk'` and `'rstagger'`,
        `0.9 * ndim * slices` for `'slice'`, `2.0 * slices` for `'rslice'`,
        and `25.0 * slices` for `'hslice'`.

    first_update : dict, optional
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube (`'none'`)
        to the one specified by `sample`. Options are the minimum number of
        likelihood calls (`'min_ncall'`) and the minimum allowed overall
        efficiency in percent (`'min_eff'`). Defaults are `2 * nlive` and
        `10.`, respectively.

    npdim : int, optional
        Number of parameters accepted by `prior_transform`. This might differ
        from `ndim` in the case where a parameter of loglikelihood is dependent
        upon multiple independently distributed parameters, some of which may
        be nuisance parameters.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    queue_size : int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) `queue_size` many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated. If no value is passed, this defaults to `pool.size` (if
        a `pool` has been provided) and `1` otherwise (no parallelism).

    pool : user-provided pool, optional
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where a pool should be used to
        execute operations in parallel. These govern whether `prior_transform`
        is executed in parallel during initialization (`'prior_transform'`),
        `loglikelihood` is executed in parallel during initialization
        (`'loglikelihood'`), live points are proposed in parallel during a run
        (`'propose_point'`), bounding distributions are updated in
        parallel during a run (`'update_bound'`), and the stopping criteria
        is evaluated in parallel during a run (`'stop_function'`).
        Default is `True` for all options.

    logl_args : iterable, optional
        Additional arguments that can be passed to `loglikelihood`.

    logl_kwargs : dict, optional
        Additional keyword arguments that can be passed to `loglikelihood`.

    ptform_args : iterable, optional
        Additional arguments that can be passed to `prior_transform`.

    ptform_kwargs : dict, optional
        Additional keyword arguments that can be passed to `prior_transform`.

    gradient : function, optional
        A function which returns the gradient corresponding to
        the provided `loglikelihood` *with respect to the unit cube*.
        If provided, this will be used when computing reflections
        when sampling with `'hslice'`. If not provided, gradients are
        approximated numerically using 2-sided differencing.

    grad_args : iterable, optional
        Additional arguments that can be passed to `gradient`.

    grad_kwargs : dict, optional
        Additional keyword arguments that can be passed to `gradient`.

    compute_jac : bool, optional
        Whether to compute and apply the Jacobian `dv/du`
        from the target space `v` to the unit cube `u` when evaluating the
        `gradient`. If `False`, the gradient provided is assumed to be
        already defined with respect to the unit cube. If `True`, the gradient
        provided is assumed to be defined with respect to the target space
        so the Jacobian needs to be numerically computed and applied. Default
        is `False`.

    enlarge : float, optional
        Enlarge the volumes of the specified bounding object(s) by this
        fraction. The preferred method is to determine this organically
        using bootstrapping. If `bootstrap > 0`, this defaults to `1.0`.
        If `bootstrap = 0`, this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes. Can lead
        to unstable bounding ellipsoids. Default is `0` (no bootstrap).

    vol_dec : float, optional
        For the `'multi'` bounding option, the required fractional reduction
        in volume after splitting an ellipsoid in order to to accept the split.
        Default is `0.5`.

    vol_check : float, optional
        For the `'multi'` bounding option, the factor used when checking if
        the volume of the original bounding ellipsoid is large enough to
        warrant `> 2` splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is `2.0`.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling
        options, the number of times to execute a "slice update"
        before proposing a new live point. Default is `5`.
        Note that `'slice'` cycles through **all dimensions**
        when executing a "slice update".

    fmove : float, optional
        The target fraction of samples that are proposed along a trajectory
        (i.e. not reflecting) for the `'hslice'` sampling option.
        Default is `0.9`.

    max_move : int, optional
        The maximum number of timesteps allowed for `'hslice'`
        per proposal forwards and backwards in time. Default is `100`.

    Returns
    -------
    sampler : a :class:`dynesty.DynamicSampler` instance
        An initialized instance of the dynamic nested sampler.

    """

    # Prior dimensions.
    if npdim is None:
        npdim = ndim

    # Bounding method.
    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    # Sampling method.
    if sample == 'auto':
        if npdim < 10:
            sample = 'unif'
        elif 10 <= npdim <= 20:
            sample = 'rwalk'
        else:
            if gradient is None:
                sample = 'slice'
            else:
                sample = 'hslice'
    if sample not in _SAMPLING:
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    # Gather non-periodic boundary conditions.
    if periodic is not None:
        nonperiodic = np.ones(npdim, dtype='bool')
        nonperiodic[periodic] = False
    else:
        nonperiodic = None
    kwargs['nonperiodic'] = nonperiodic

    # Update interval for bounds.
    if update_interval is None:
        if sample == 'unif':
            update_interval = 1.5
        elif sample == 'rwalk' or sample == 'rstagger':
            update_interval = 0.15 * walks
        elif sample == 'slice':
            update_interval = 0.9 * npdim * slices
        elif sample == 'rslice':
            update_interval = 2.0 * slices
        elif sample == 'hslice':
            update_interval = 25.0 * slices
        else:
            raise ValueError("Unknown sampling method: '{0}'".format(sample))
    if bound == 'none':
        update_interval = sys.maxsize  # no need to update with no bounds

    # Keyword arguments controlling the first update.
    if first_update is None:
        first_update = dict()

    # Random state.
    if rstate is None:
        rstate = np.random

    # Log-likelihood.
    if logl_args is None:
        logl_args = []
    if logl_kwargs is None:
        logl_kwargs = {}

    # Prior transform.
    if ptform_args is None:
        ptform_args = []
    if ptform_kwargs is None:
        ptform_kwargs = dict()

    # gradient
    if grad_args is None:
        grad_args = []
    if grad_kwargs is None:
        grad_kwargs = {}

    # Bounding distribution modifications.
    if enlarge is not None:
        kwargs['enlarge'] = enlarge
    if bootstrap is not None:
        kwargs['bootstrap'] = bootstrap
    if vol_dec is not None:
        kwargs['vol_dec'] = vol_dec
    if vol_check is not None:
        kwargs['vol_check'] = vol_check

    # Sampling.
    if walks is not None:
        kwargs['walks'] = walks
    if facc is not None:
        kwargs['facc'] = facc
    if slices is not None:
        kwargs['slices'] = slices
    if fmove is not None:
        kwargs['fmove'] = fmove
    if max_move is not None:
        kwargs['max_move'] = max_move

    # Set up parallel (or serial) evaluation.
    if queue_size is not None and queue_size < 1:
        raise ValueError("The queue must contain at least one element!")
    elif (queue_size == 1) or (pool is None and queue_size is None):
        queue_size = 1
    elif pool is not None:
        if queue_size is None:
            try:
                queue_size = pool.size
            except:
                raise ValueError("Cannot initialize `queue_size` because "
                                 "`pool.size` has not been provided. Please "
                                 "define `pool.size` or specify `queue_size` "
                                 "explicitly.")
    else:
        raise ValueError("`queue_size > 1` but no `pool` provided.")
    if use_pool is None:
        use_pool = dict()

    # Wrap functions.
    ptform = _function_wrapper(prior_transform, ptform_args, ptform_kwargs,
                               name='prior_transform')
    loglike = _function_wrapper(loglikelihood, logl_args, logl_kwargs,
                                name='loglikelihood')

    # Add in gradient.
    if gradient is not None:
        grad = _function_wrapper(gradient, grad_args, grad_kwargs,
                                 name='gradient')
        kwargs['grad'] = grad
        kwargs['compute_jac'] = compute_jac

    # Initialize our nested sampler.
    sampler = DynamicSampler(loglike, ptform, npdim,
                             bound, sample, update_interval, first_update,
                             rstate, queue_size, pool, use_pool, kwargs)

    return sampler


class _function_wrapper(object):
    """
    A hack to make functions pickleable when `args` or `kwargs` are
    also included. Based on the implementation in
    `emcee <http://dan.iel.fm/emcee/>`_.

    """

    def __init__(self, func, args, kwargs, name='input'):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self, x):
        try:
            return self.func(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("Exception while calling {0} function:".format(self.name))
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
