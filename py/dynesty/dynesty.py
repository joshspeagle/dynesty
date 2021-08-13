#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The top-level interface (defined natively upon initialization) that
provides access to the two main sampler "super-classes" via
:meth:`NestedSampler` and :meth:`DynamicNestedSampler`.

"""

import sys
import warnings
import math
import numpy as np

from .nestedsamplers import _SAMPLING
from .dynamicsampler import (DynamicSampler, __get_update_interval_ratio,
                             _SAMPLERS)
from .utils import LogLikelihood, get_random_generator

__all__ = ["NestedSampler", "DynamicNestedSampler", "_function_wrapper"]


_CITES = {'default':  # default set of citations
          "Code and Methods:\n================\n"
          "Speagle (2020): "
          "ui.adsabs.harvard.edu/abs/2019arXiv190402180S\n\n"
          "Nested Sampling:\n===============\n"
          "Skilling (2004): "
          "ui.adsabs.harvard.edu/abs/2004AIPC..735..395S\n"
          "Skilling (2006): "
          "projecteuclid.org/euclid.ba/1340370944\n",
          'dynamic':  # dynamic nested sampling
          "Dynamic Nested Sampling:\n=======================\n"
          "Higson et al. (2019): "
          "doi.org/10.1007/s11222-018-9844-0\n",
          'none': "Bounding Method:\n===============\n",  # no bound
          'single':  # single ellipsoid
          "Bounding Method:\n===============\n"
          "Mukherjee, Parkinson & Liddle (2006): "
          "ui.adsabs.harvard.edu/abs/2006ApJ...638L..51M\n",
          'multi':  # multiple ellipsoids
          "Bounding Method:\n===============\n"
          "Feroz, Hobson & Bridges (2009): "
          "ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F\n",
          'balls':  # overlapping balls
          "Bounding Method:\n===============\n"
          "Buchner (2016): "
          "ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B\n"
          "Buchner (2017): "
          "ui.adsabs.harvard.edu/abs/2017arXiv170704476B\n",
          'cubes':  # overlapping cubes
          "Bounding Method:\n===============\n"
          "Buchner (2016): "
          "ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B\n"
          "Buchner (2017): "
          "ui.adsabs.harvard.edu/abs/2017arXiv170704476B\n",
          'unif': "Sampling Method:\n===============\n",  # uniform sampling
          'rwalk':  # random walk
          "Sampling Method:\n===============\n"
          "Skilling (2006): "
          "projecteuclid.org/euclid.ba/1340370944\n",
          'rstagger':  # random stagger
          "Sampling Method:\n===============\n"
          "Skilling (2006): "
          "projecteuclid.org/euclid.ba/1340370944\n",
          'slice':  # multivariate slice
          "Sampling Method:\n===============\n"
          "Neal (2003): "
          "projecteuclid.org/euclid.aos/1056562461\n"
          "Handley, Hobson & Lasenby (2015a): "
          "ui.adsabs.harvard.edu/abs/2015MNRAS.450L..61H\n"
          "Handley, Hobson & Lasenby (2015b): "
          "ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H\n",
          'rslice':  # random slice
          "Sampling Method:\n===============\n"
          "Neal (2003): "
          "projecteuclid.org/euclid.aos/1056562461\n"
          "Handley, Hobson & Lasenby (2015a): "
          "ui.adsabs.harvard.edu/abs/2015MNRAS.450L..61H\n"
          "Handley, Hobson & Lasenby (2015b): "
          "ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H\n",
          'hslice':  # "hamiltonian" slice
          "Sampling Method:\n===============\n"
          "Neal (2003): "
          "projecteuclid.org/euclid.aos/1056562461\n"
          "Skilling (2012): "
          "aip.scitation.org/doi/abs/10.1063/1.3703630\n"
          "Feroz & Skilling (2013): "
          "ui.adsabs.harvard.edu/abs/2013AIPC.1553..106F\n"
          "Speagle (2020): "
          "ui.adsabs.harvard.edu/abs/2019arXiv190402180S\n"}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def __get_auto_sample(ndim, gradient):
    """ Decode which sampling method to use

    Arguments:
    ndim: int (dimensionality)
    gradient: (None or function/true)
    Returns: sampler string
    """
    if ndim < 10:
        sample = 'unif'
    elif 10 <= ndim <= 20:
        sample = 'rwalk'
    else:
        if gradient is None:
            sample = 'rslice'
        else:
            sample = 'hslice'
    return sample


def __get_walks_slices(walks0, slices0, sample, ndim):
    """
    Get the best number of steps for random walk/slicing based on 
    the type of sampler and dimension
    
    Arguments:
    walks0: integer (provided by user or none for auto)
    slices0: integer (provided by user or none for auto)
    sample: string (sampler type)
    ndim: int (dimensionality)
    Returns the tuple with number of walk steps, number of slice steps
    """
    walks, slices = None, None
    # see https://github.com/joshspeagle/dynesty/issues/289
    if sample in ['hslice', 'rslice']:
        slices = 3 + ndim
    elif sample == 'slice':
        slices = 3
        # we don't add dimensions, since we loop over them
    elif sample in ['rwalk', 'rstagger']:
        # this is technically incorrect a we need to add ndim **2
        walks = 20 + ndim
    slices = slices0 or slices
    walks = walks0 or walks
    return walks, slices


def NestedSampler(loglikelihood,
                  prior_transform,
                  ndim,
                  nlive=500,
                  bound='multi',
                  sample='auto',
                  periodic=None,
                  reflective=None,
                  update_interval=None,
                  first_update=None,
                  npdim=None,
                  rstate=None,
                  queue_size=None,
                  pool=None,
                  use_pool=None,
                  live_points=None,
                  logl_args=None,
                  logl_kwargs=None,
                  ptform_args=None,
                  ptform_kwargs=None,
                  gradient=None,
                  grad_args=None,
                  grad_kwargs=None,
                  compute_jac=False,
                  enlarge=None,
                  bootstrap=0,
                  walks=None,
                  facc=0.5,
                  slices=None,
                  fmove=0.9,
                  max_move=100,
                  update_func=None,
                  ncdim=None,
                  save_history=False,
                  history_filename=None):
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
              `'slice'`, `'rslice'`, `'hslice'`, callable}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Unique methods available are:
        uniform sampling within the bounds(`'unif'`),
        random walks with fixed proposals (`'rwalk'`),
        random walks with variable ("staggering") proposals (`'rstagger'`),
        multivariate slice sampling along preferred orientations (`'slice'`),
        "random" slice sampling along all orientations (`'rslice'`),
        "Hamiltonian" slices along random trajectories (`'hslice'`), and
        any callable function which follows the pattern of the sample methods
        defined in dynesty.sampling.
        `'auto'` selects the sampling method based on the dimensionality
        of the problem (from `ndim`).
        When `ndim < 10`, this defaults to `'unif'`.
        When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
        When `ndim > 20`, this defaults to `'hslice'` if a `gradient` is
        provided and `'rslice'` otherwise. `'rstagger'` and `'slice'`
        are provided as alternatives for `'rwalk'` and `'rslice'`,
        respectively.
        Default is `'auto'`.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. Default is `None` (i.e. no periodic
        boundary conditions).

    reflective : iterable, optional
        A list of indices for parameters with reflective boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may reflect at the edge. Default is `None` (i.e. no reflective
        boundary conditions).

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

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance. If not given, the
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

    update_func : function, optional
        Any callable function which takes in a `blob` and `scale`
        as input and returns a modification to the internal `scale` as output.
        Must follow the pattern of the update methods defined
        in dynesty.nestedsamplers. If provided, this will supersede the
        default functions used to update proposals. In the case where a custom
        callable function is passed to `sample` but no similar function is
        passed to `update_func`, this will default to no update.

    ncdim: int, optional
        The number of clustering dimensions. The first ncdim dimensions will
        be sampled using the sampling method, the remaining dimensions will
        just sample uniformly from the prior distribution.
        If this is `None` (default), this will default to npdim.

    Returns
    -------
    sampler : sampler from :mod:`~dynesty.nestedsamplers`
        An initialized instance of the chosen sampler specified via `bound`.

    """

    # Prior dimensions.
    if npdim is None:
        npdim = ndim
    if ncdim is None:
        ncdim = npdim

    # Bounding method.
    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    # Sampling method.
    if sample == 'auto':
        sample = __get_auto_sample(ndim, gradient)

    walks, slices = __get_walks_slices(walks, slices, sample, ndim)

    if ncdim != npdim and sample in ['slice', 'hslice', 'rslice']:
        raise ValueError('ncdim unsupported for slice sampling')

    # Custom sampling function.
    if sample not in _SAMPLING and not callable(sample):
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    kwargs = {}
    # Custom updating function.
    if update_func is not None and not callable(update_func):
        raise ValueError("Unknown update function: '{0}'".format(update_func))
    kwargs['update_func'] = update_func

    # Citation generator.
    kwargs['cite'] = (_CITES['default'] + "\n" + _CITES[bound] + "\n" +
                      _CITES[sample])

    # Dimensional warning check.
    if nlive <= 2 * ndim:
        warnings.warn("Beware! Having `nlive <= 2 * ndim` is extremely risky!")

    # Gather boundary conditions.
    if periodic is not None and reflective is not None:
        if np.intersect1d(periodic, reflective) != 0:
            raise ValueError("You have specified a parameter as both "
                             "periodic and reflective.")
    nonbounded = np.ones(npdim, dtype='bool')
    if periodic is not None:
        nonbounded[periodic] = False
    if reflective is not None:
        nonbounded[reflective] = False
    kwargs['nonbounded'] = nonbounded
    kwargs['periodic'] = periodic
    kwargs['reflective'] = reflective

    # Keyword arguments controlling the first update.
    if first_update is None:
        first_update = dict()

    # Random state.
    if rstate is None:
        rstate = get_random_generator()

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

    update_interval_ratio = __get_update_interval_ratio(
        update_interval, sample, bound, nlive, ndim, slices, walks)
    update_interval = int(
        max(min(np.round(update_interval_ratio * nlive), sys.maxsize), 1))

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
            except AttributeError:
                raise ValueError("Cannot initialize `queue_size` because "
                                 "`pool.size` has not been provided. Please"
                                 "define `pool.size` or specify `queue_size` "
                                 "explicitly.")
    else:
        raise ValueError("`queue_size > 1` but no `pool` provided.")
    if use_pool is None:
        use_pool = dict()

    # Wrap functions.
    ptform = _function_wrapper(prior_transform,
                               ptform_args,
                               ptform_kwargs,
                               name='prior_transform')
    if use_pool.get('loglikelihood') or True:
        pool_logl = pool
    else:
        pool_logl = None
    loglike = LogLikelihood(_function_wrapper(loglikelihood,
                                              logl_args,
                                              logl_kwargs,
                                              name='loglikelihood'),
                            ndim,
                            save=save_history,
                            history_filename=history_filename
                            or 'dynesty_logl_history.h5',
                            pool=pool_logl)

    # Add in gradient.
    if gradient is not None:
        grad = _function_wrapper(gradient,
                                 grad_args,
                                 grad_kwargs,
                                 name='gradient')
        kwargs['grad'] = grad
        kwargs['compute_jac'] = compute_jac

    # Initialize live points and calculate log-likelihoods.
    if live_points is None:
        # If no live points are provided, propose them by randomly sampling
        # from the unit cube.
        for attempt in range(100):
            live_u = rstate.uniform(size=(nlive,
                                          npdim))  # positions in unit cube
            if use_pool.get('prior_transform', True):
                live_v = np.array(list(M(ptform,
                                         np.array(live_u))))  # parameters
            else:
                live_v = np.array(list(map(ptform, np.array(live_u))))
            live_logl = loglike.map(np.array(live_v))  # log-like
            live_points = [live_u, live_v, live_logl]

            # Convert all `-np.inf` log-likelihoods to finite large numbers.
            # Necessary to keep estimators in our sampler from breaking.
            for i, logl in enumerate(live_points[2]):
                if not np.isfinite(logl):
                    if np.sign(logl) < 0:
                        live_points[2][i] = -1e300
                    else:
                        raise ValueError("The log-likelihood ({0}) of live "
                                         "point {1} located at u={2} v={3} "
                                         "is invalid.".format(
                                             logl, i, live_points[0][i],
                                             live_points[1][i]))

            # Check to make sure there is at least one finite log-likelihood
            # value within the initial set of live points.
            if any(live_points[2] != -1e300):
                break
        else:
            # If we found nothing after many attempts, raise the alarm.
            raise RuntimeError("After many attempts, not a single live point "
                               "had a valid log-likelihood! Please check your "
                               "prior transform and/or log-likelihood.")
    else:
        # If live points were provided, convert the log-likelihoods and then
        # run a quick safety check.
        for i, logl in enumerate(live_points[2]):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    live_points[2][i] = -1e300
                else:
                    raise ValueError("The log-likelihood ({0}) of live "
                                     "point {1} located at u={2} v={3} "
                                     "is invalid.".format(
                                         logl, i, live_points[0][i],
                                         live_points[1][i]))
        if all(live_points[2] == -1e300):
            raise ValueError("Not a single provided live point has a valid "
                             "log-likelihood!")

    # Initialize our nested sampler.
    sampler = _SAMPLERS[bound](loglike,
                               ptform,
                               npdim,
                               live_points,
                               sample,
                               update_interval,
                               first_update,
                               rstate,
                               queue_size,
                               pool,
                               use_pool,
                               kwargs,
                               ncdim=ncdim)

    return sampler


def DynamicNestedSampler(loglikelihood,
                         prior_transform,
                         ndim,
                         nlive=None,
                         bound='multi',
                         sample='auto',
                         periodic=None,
                         reflective=None,
                         update_interval=None,
                         first_update=None,
                         npdim=None,
                         rstate=None,
                         queue_size=None,
                         pool=None,
                         use_pool=None,
                         logl_args=None,
                         logl_kwargs=None,
                         ptform_args=None,
                         ptform_kwargs=None,
                         gradient=None,
                         grad_args=None,
                         grad_kwargs=None,
                         compute_jac=False,
                         enlarge=None,
                         bootstrap=0,
                         walks=None,
                         facc=0.5,
                         slices=None,
                         fmove=0.9,
                         max_move=100,
                         update_func=None,
                         ncdim=None,
                         save_history=False,
                         history_filename=None):
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
        "random" slice sampling along all orientations (`'rslice'`),
        "Hamiltonian" slices along random trajectories (`'hslice'`), and
        any callable function which follows the pattern of the sample methods
        defined in dynesty.sampling.
        `'auto'` selects the sampling method based on the dimensionality
        of the problem (from `ndim`).
        When `ndim < 10`, this defaults to `'unif'`.
        When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
        When `ndim > 20`, this defaults to `'hslice'` if a `gradient` is
        provided and `'rslice'` otherwise. `'rstagger'` and `'slice'`
        are provided as alternatives for `'rwalk'` and `'rslice'`,
        respectively.
        Default is `'auto'`.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. Default is `None` (i.e. no periodic
        boundary conditions).

    reflective : iterable, optional
        A list of indices for parameters with reflective boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may reflect at the edge. Default is `None` (i.e. no reflective
        boundary conditions).

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

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance. If not given, the
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

    update_func : function, optional
        Any callable function which takes in a `blob` and `scale`
        as input and returns a modification to the internal `scale` as output.
        Must follow the pattern of the update methods defined
        in dynesty.nestedsamplers. If provided, this will supersede the
        default functions used to update proposals. In the case where a custom
        callable function is passed to `sample` but no similar function is
        passed to `update_func`, this will default to no update.

    ncdim: int, optional
        The number of clustering dimensions. The first ncdim dimensions will
        be sampled using the sampling method, the remaining dimensions will
        just sample uniformly from the prior distribution.
        If this is `None` (default), this will default to npdim.

    Returns
    -------
    sampler : a :class:`dynesty.DynamicSampler` instance
        An initialized instance of the dynamic nested sampler.

    """

    # Prior dimensions.
    if npdim is None:
        npdim = ndim
    if ncdim is None:
        ncdim = npdim

    nlive = nlive or 500

    # Bounding method.
    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    # Sampling method.
    if sample == 'auto':
        sample = __get_auto_sample(ndim, gradient)

    walks, slices = __get_walks_slices(walks, slices, sample, ndim)

    if ncdim != npdim and sample in ['slice', 'hslice', 'rslice']:
        raise ValueError('ncdim unsupported for slice sampling')

    update_interval_ratio = __get_update_interval_ratio(
        update_interval, sample, bound, 1, ndim, slices, walks)

    kwargs = {}

    # Custom sampling function.
    if sample not in _SAMPLING and not callable(sample):
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    # Custom updating function.
    if update_func is not None and not callable(update_func):
        raise ValueError("Unknown update function: '{0}'".format(update_func))
    kwargs['update_func'] = update_func

    # Citation generator.
    kwargs['cite'] = (_CITES['default'] + "\n" + _CITES['dynamic'] + "\n" +
                      _CITES[bound] + "\n" + _CITES[sample])

    # Gather boundary conditions.
    if periodic is not None and reflective is not None:
        if np.intersect1d(periodic, reflective) != 0:
            raise ValueError("You have specified a parameter as both "
                             "periodic and reflective.")
    nonbounded = np.ones(npdim, dtype='bool')
    if periodic is not None:
        nonbounded[periodic] = False
    if reflective is not None:
        nonbounded[reflective] = False
    kwargs['nonbounded'] = nonbounded
    kwargs['periodic'] = periodic
    kwargs['reflective'] = reflective

    # Keyword arguments controlling the first update.
    if first_update is None:
        first_update = dict()

    # Random state.
    if rstate is None:
        rstate = get_random_generator()

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
            except AttributeError:
                raise ValueError("Cannot initialize `queue_size` because "
                                 "`pool.size` has not been provided. Please "
                                 "define `pool.size` or specify `queue_size` "
                                 "explicitly.")
    else:
        raise ValueError("`queue_size > 1` but no `pool` provided.")
    if use_pool is None:
        use_pool = dict()

    # Wrap functions.
    ptform = _function_wrapper(prior_transform,
                               ptform_args,
                               ptform_kwargs,
                               name='prior_transform')

    if use_pool.get('loglikelihood') or True:
        pool_logl = pool
    else:
        pool_logl = None
    loglike = LogLikelihood(_function_wrapper(loglikelihood,
                                              logl_args,
                                              logl_kwargs,
                                              name='loglikelihood'),
                            ndim,
                            pool=pool_logl,
                            history_filename=history_filename
                            or 'dynesty_logl_history.h5',
                            save=save_history)

    # Add in gradient.
    if gradient is not None:
        grad = _function_wrapper(gradient,
                                 grad_args,
                                 grad_kwargs,
                                 name='gradient')
        kwargs['grad'] = grad
        kwargs['compute_jac'] = compute_jac

    # Initialize our nested sampler.
    sampler = DynamicSampler(loglike, ptform, npdim, bound, sample,
                             update_interval_ratio, first_update, rstate,
                             queue_size, pool, use_pool, ncdim, nlive, kwargs)

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
        except:  # noqa
            import traceback
            print("Exception while calling {0} function:".format(self.name))
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
