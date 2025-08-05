#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The top-level interface (defined natively upon initialization) that
provides access to the two main sampler "super-classes" via
:meth:`NestedSampler` and :meth:`DynamicNestedSampler`.

"""

import sys
import warnings
import traceback
import numpy as np
from .sampling import (INTERNAL_SAMPLER_LIST, InternalSampler, RSliceSampler,
                       UniformBoundSampler, RWalkSampler, SliceSampler)
from .sampler import Sampler, _initialize_live_points
from .bounding import BOUND_LIST
from . import bounding
from .dynamicsampler import DynamicSampler
from .utils import (LogLikelihood, get_random_generator, get_nonbounded)

__all__ = ["NestedSampler", "DynamicNestedSampler", "_function_wrapper"]


def _get_citations(nested_type, bound, sampler):
    """
    Return a string of citations for given dynesty run

    Parameters
    ----------
    nested_type: string
        Either dynamic or static
    bound: string
        Bound type used
    sampler: string
        Internal sampler type

    Returns
    -------
    citations: string
         The long printable string of citations
    """

    # In all cases the references are organized as
    # a tuple of reference string and url
    # or a list of tuples

    # Main references for the code
    default_refs = [
        ("Speagle (2020)", "ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S"),
        ("Koposov et al. (2023)", "doi.org/10.5281/zenodo.3348367")
    ]

    # Basics of nested sampling algorithm
    nested_refs = [
        ("Skilling (2004)", "ui.adsabs.harvard.edu/abs/2004AIPC..735..395S"),
        ("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944")
    ]

    # The dynamic nested sampling
    dynamic_refs = [("Higson et al. (2019)",
                     "doi.org/10.1007/s11222-018-9844-0")]

    # citations for different bounds in a dictionary
    bound_refs = {
        'none':
        '',
        'single': ("Mukherjee, Parkinson & Liddle (2006)",
                   "ui.adsabs.harvard.edu/abs/2006ApJ...638L..51M"),
        'multi': ("Feroz, Hobson & Bridges (2009)",
                  "ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F"),
        'balls':
        [("Buchner (2016)", "ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B"),
         ("Buchner (2017)", "ui.adsabs.harvard.edu/abs/2017arXiv170704476B")],
        'cubes':
        [("Buchner (2016)", "ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B"),
         ("Buchner (2017)", "ui.adsabs.harvard.edu/abs/2017arXiv170704476B")]
    }

    def reflist_tostring(x):
        """ internal function to convert reference lists to
        a printable string
        """
        if isinstance(x, str):
            return x
        if isinstance(x, tuple):
            return x[0] + ': ' + x[1]
        if isinstance(x, list):
            return '\n'.join([_[0] + ': ' + _[1] for _ in x])
        else:
            return str(x)

    default_citations = reflist_tostring(default_refs)
    nested_citations = reflist_tostring(nested_refs)
    # if using a custom sampler, dynesty does not know the citations
    bound_citations = reflist_tostring(bound_refs.get(bound, ""))
    sampler_citations = reflist_tostring(sampler.citations)

    assert nested_type in ['dynamic', 'static']
    if nested_type == 'dynamic':
        dynamic_citations = reflist_tostring(dynamic_refs)
        dynamic_citations = ("Dynamic Nested Sampling:\n"
                             "=======================\n" + dynamic_citations)
    else:
        dynamic_citations = ""

    citations = f"""Code and Methods:\n ================
{default_citations}

Nested Sampling:\n===============
{nested_citations}
{dynamic_citations}

Bounding Method:\n===============
{bound_citations}

Sampling Method:\n===============
{sampler_citations}
"""
    return citations


def _get_internal_sampler(sampling, ndim, ncdim, periodic, reflective, walks,
                          slices, facc):
    default_steps = {'rwalk': ndim + 20, 'slice': 3, 'rslice': 3 + ndim}
    if sampling == 'auto':
        if ndim < 10:
            sampling = UniformBoundSampler(ndim=ndim)
        elif 10 <= ndim <= 20:
            sampling = RWalkSampler(ndim=ndim, walks=default_steps['rwalk'])
        else:
            sampling = RSliceSampler(ndim=ndim, slices=default_steps['rslice'])

    nonbounded = get_nonbounded(ndim, periodic, reflective)
    sampler_kw = dict(ncdim=ncdim,
                      ndim=ndim,
                      nonbounded=nonbounded,
                      periodic=periodic,
                      reflective=reflective,
                      facc=facc)
    if sampling == 'rslice':
        sampler_kw['slices'] = slices or default_steps['rslice']
        internal_sampler = RSliceSampler(**sampler_kw)
    elif sampling == 'slice':
        sampler_kw['slices'] = slices or default_steps['slice']
        internal_sampler = SliceSampler(**sampler_kw)
    elif sampling == 'rwalk':
        sampler_kw['walks'] = walks or default_steps['rwalk']
        internal_sampler = RWalkSampler(**sampler_kw)
    elif sampling == 'unif':
        internal_sampler = UniformBoundSampler(**sampler_kw)
    elif isinstance(sampling, InternalSampler):
        # todo check what to do with the options
        internal_sampler = sampling._new_from_template(sampler_kw)
    else:
        raise ValueError(f'Unsupported Sampler {sampling}')
    if sampling == 'rwalk' and slices is not None or (
            sampling in ['rslice', 'slice'] and walks is not None):
        warnings.warn('Specifying slice option while using rwalk sampler or '
                      ' walks option with a slice sampler'
                      ' does not make sense')

    return internal_sampler


def _get_enlarge_bootstrap(sample, enlarge, bootstrap):
    """
    Determine the enlarge, bootstrap for a given run
    """
    # we should make it dimension dependent I think...
    DEFAULT_ENLARGE = 1.25
    DEFAULT_UNIF_BOOTSTRAP = 5
    if enlarge is not None and bootstrap is None:
        # If enlarge is specified and bootstrap is not we just use enlarge
        # with no nootstrapping
        assert enlarge >= 1
        return enlarge, 0
    elif enlarge is None and bootstrap is not None:
        # If bootstrap is specified but enlarge is not we just use bootstrap
        # And if we allow zero bootstrap if we want to force no bootstrap
        assert ((bootstrap > 1) or (bootstrap == 0))
        return 1, bootstrap
    elif enlarge is None and bootstrap is None:
        # If neither enlarge or bootstrap are specified we are doing
        # things in auto-mode. I.e. use enlarge unless the uniform
        # sampler is selected
        if isinstance(sample, UniformBoundSampler):
            return 1, DEFAULT_UNIF_BOOTSTRAP
        else:
            return DEFAULT_ENLARGE, 0
    else:
        # Both enlarge and bootstrap were specified
        if bootstrap == 0 or enlarge == 1:
            return enlarge, bootstrap
        else:
            raise ValueError('Enlarge and bootstrap together do not make '
                             'sense unless bootstrap=0 or enlarge = 1')


def _parse_pool_queue(pool, queue_size):
    """
    Common functionality of interpretign the pool and queue_size
    arguments to Dynamic and static nested samplers
    """
    if queue_size is not None and queue_size < 1:
        raise ValueError("The queue must contain at least one element!")
    elif (queue_size == 1) or (pool is None and queue_size is None):
        mapper = map
        queue_size = 1
    elif pool is not None:
        mapper = pool.map
        if queue_size is None:
            try:
                queue_size = pool.size
            except AttributeError as e:
                raise ValueError("Cannot initialize `queue_size` because "
                                 "`pool.size` has not been provided. Please"
                                 "define `pool.size` or specify `queue_size` "
                                 "explicitly.") from e
    else:
        raise ValueError("`queue_size > 1` but no `pool` provided.")

    return mapper, queue_size


def _check_first_update(first_update):
    """
    Verify that the first_update dictionary is valid
    Specifically that it doesn't have unrecognized keywords
    """
    for k in first_update.keys():
        if k not in ['min_ncall', 'min_eff']:
            raise ValueError('Unrecognized keywords in first_update')


def _get_update_interval_ratio(update_interval, sample, nlive):
    """
    Get the update_interval (i.e. boundary update interval)
    divided by the number of live points.
    """

    if update_interval is None:
        if isinstance(sample, InternalSampler):
            update_interval_ratio = sample.update_bound_interval_ratio
        else:
            update_interval_ratio = 1
            warnings.warn(
                "No update_interval set with unknown sampling method."
                ". Defaulting to no 1 update per nlive points.")
    elif isinstance(update_interval, float):
        update_interval_ratio = update_interval
    elif isinstance(update_interval, int):
        update_interval_ratio = update_interval * 1. / nlive
    else:
        raise RuntimeError(f'Strange update_interval value {update_interval}')
    return update_interval_ratio


def _assemble_sampler_docstring(dynamic):
    """
    Assemble the docstring for the NestedSampler and DynamicNestedSampler
    We do that to avoid duplicating the parameter descriptions
    """
    common = """
        Parameters
        ----------
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        prior_transform : function
            Function translating a unit cube to the parameter space according
            to the prior. The input is a 1-d `~numpy` array with length
            `ndim`, where each value is in the range [0, 1). The return
            value should also be a 1-d `~numpy` array with length `ndim`,
            where each value is a parameter.
            The return value is passed to the loglikelihood function. For
            example, for a 2 parameter model with flat priors in the range
            [0, 2), the function would be::

                def prior_transform(u):
                    return 2.0 * u

        ndim : int
            Number of parameters returned by `prior_transform` and accepted by
            `loglikelihood`.

        nlive : int, optional
            Number of "live" points. Larger numbers result in a more finely
            sampled posterior (more accurate evidence), but also a larger
            number of iterations required to converge. Default is `500`.

        bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}, \
optional
            Method used to approximately bound the prior using the current
            set of live points. Conditions the sampling methods used to
            propose new live points. Choices are no bound (`'none'`), a single
            bounding ellipsoid (`'single'`), multiple bounding ellipsoids
            (`'multi'`), balls centered on each live point (`'balls'`), and
            cubes centered on each live point (`'cubes'`). Default is
            `'multi'`.

        sample : {`'auto'`, `'unif'`, `'rwalk'`, `'slice'`, `'rslice'`},
            optional
            Method used to sample uniformly within the likelihood constraint,
            conditioned on the provided bounds. Unique methods available are:
            uniform sampling within the bounds(`'unif'`),
            random walks with fixed proposals (`'rwalk'`),
            multivariate slice sampling along preferred orientations
            (`'slice'`),
            "random" slice sampling along all orientations (`'rslice'`),
            `'auto'` selects the sampling method based on the dimensionality
            of the problem (from `ndim`).
            When `ndim < 10`, this defaults to `'unif'`.
            When `10 <= ndim <= 20`, this defaults to `'rwalk'`.
            When `ndim > 20`, this defaults to`'rslice'`. `'slice'`
            is provided as alternative for`'rslice'`.
            Default is `'auto'`.

        periodic : iterable, optional
            A list of indices for parameters with periodic boundary conditions.
            These parameters *will not* have their positions constrained to be
            within the unit cube, enabling smooth behavior for parameters
            that may wrap around the edge. Default is `None` (i.e. no periodic
            boundary conditions).

        reflective : iterable, optional
            A list of indices for parameters with reflective boundary
            conditions.
            These parameters *will not* have their positions constrained to be
            within the unit cube, enabling smooth behavior for parameters
            that may reflect at the edge. Default is `None` (i.e. no reflective
            boundary conditions).

        update_interval : int or float, optional
            If an integer is passed, only update the proposal distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the proposal after every
            `round(update_interval * nlive)`-th likelihood
            call. Larger update intervals larger can be more efficient
            when the likelihood function is quick to evaluate. Default behavior
            is to target a roughly constant change in prior volume, with
            `1.5` for `'unif'`, `0.15 * walks` for `'rwalk'`.
            `0.9 * ndim * slices` for `'slice'`, `2.0 * slices` for `'rslice'`.

        first_update : dict, optional
            A dictionary containing parameters governing when the sampler
            should
            first update the bounding distribution from the unit cube
            (`'none'`)
            to the one specified by `sample`. Options are the minimum number of
            likelihood calls (`'min_ncall'`) and the minimum allowed overall
            efficiency in percent (`'min_eff'`). Defaults are `2 * nlive` and
            `10.`, respectively.

        rstate : `~numpy.random.Generator`, optional
            `~numpy.random.Generator` instance. If not given, the
             global random state of the `~numpy.random` module will be used.

        queue_size : int, optional
            Carry out likelihood evaluations in parallel by queueing up new
            live point proposals using (at most) `queue_size` many threads.
            Each thread independently proposes new live points until the
            proposal distribution
            is updated. If no value is passed, this defaults to `pool.size` (if
            a `pool` has been provided) and `1` otherwise (no parallelism).

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        use_pool : dict, optional
            A dictionary containing flags indicating where a pool should be
            used to execute operations in parallel. These govern whether
            `prior_transform`
            is executed in parallel during initialization
            (`'prior_transform'`),
            `loglikelihood` is executed in parallel during initialization
            (`'loglikelihood'`), live points are proposed in parallel during
            a run
            (`'propose_point'`), and bounding distributions are updated in
            parallel during a run (`'update_bound'`). Default is `True` for all
            options.

        live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods.
            By default, if these are not provided the initial set of live
            points will be drawn uniformly from the unit `ndim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points
            will result in incorrect results.**

        logl_args : iterable, optional
            Additional arguments that can be passed to `loglikelihood`.

        logl_kwargs : dict, optional
            Additional keyword arguments that can be passed to `loglikelihood`.

        ptform_args : iterable, optional
            Additional arguments that can be passed to `prior_transform`.

        ptform_kwargs : dict, optional
            Additional keyword arguments that can be passed to
            `prior_transform`.

        enlarge : float, optional
            Enlarge the volumes of the specified bounding object(s) by this
            fraction. The preferred method is to determine this organically
            using bootstrapping. If `bootstrap > 0`, this defaults to `1.0`.
            If `bootstrap = 0`, this instead defaults to `1.25`.

        bootstrap : int, optional
            Compute this many bootstrapped realizations of the bounding
            objects. Use the maximum distance found to the set of points left
            out during each iteration to enlarge the resulting volumes. Can
            lead to unstable bounding ellipsoids. Default is `None` (no
            bootstrap
            unless the sampler is uniform). If bootstrap is set to zero,
            bootstrap is disabled.

        walks : int, optional
            For the `'rwalk'` sampling option, the minimum number of steps
            (minimum 2) before proposing a new live point. Default is `25`.

        facc : float, optional
            The target acceptance fraction for the `'rwalk'` sampling option.
            Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

        slices : int, optional
            For the `'slice'`, `'rslice'` sampling
            options, the number of times to execute a "slice update"
            before proposing a new live point. Default is 3 for
            `'slice'` and 3+ndim for rslice.
            Note that `'slice'` cycles through **all dimensions**
            when executing a "slice update".

        ncdim: int, optional
            The number of clustering dimensions. The first ncdim dimensions
            will be sampled using the sampling method, the remaining
            dimensions will
            just sample uniformly from the prior distribution.
            If this is `None` (default), this will default to ndim.

        blob: bool, optional
            The default value is False. If it is true, then the log-likelihood
            should return the tuple of logl and a numpy-array "blob" that will
            stored as part of the chain. That blob can contain auxiliary
            information computed inside the likelihood function.

    """

    static_docstring = f"""
        Initializes and returns a sampler object for Static Nested Sampling.
{common}
        Returns
        -------
        sampler : sampler from :mod:`~dynesty.nestedsamplers`
            An initialized instance of the chosen sampler specified via
            `bound`.
        """

    dynamic_docstring = f"""
        Initializes a sampler object for Dynamic Nested Sampling.

{common}
        Returns
        -------
        sampler : a :class:`dynesty.DynamicSampler` instance
            An initialized instance of the dynamic nested sampler.

        """
    if dynamic:
        return dynamic_docstring
    else:
        return static_docstring


def _common_sampler_init(*,
                         nlive,
                         ndim,
                         prior_transform,
                         loglikelihood,
                         ncdim=None,
                         bound=None,
                         sample=None,
                         walks=None,
                         slices=None,
                         rstate=None,
                         periodic=None,
                         reflective=None,
                         bootstrap=None,
                         enlarge=None,
                         first_update=None,
                         facc=None,
                         blob=None,
                         ptform_args=None,
                         ptform_kwargs=None,
                         logl_args=None,
                         logl_kwargs=None,
                         use_pool=None,
                         pool=None,
                         queue_size=None,
                         save_history=None,
                         history_filename=None,
                         save_evaluation_history=None,
                         update_interval=None,
                         dynamic=False):
    ret = {}

    ncdim = ncdim or ndim
    ret['ncdim'] = ncdim
    # Dimensional warning check.
    if nlive <= 2 * ndim:
        warnings.warn("Beware! Having `nlive <= 2 * ndim` is extremely risky!")

    # Bounding method.
    if bound not in BOUND_LIST and not isinstance(bound, bounding.Bound):
        raise ValueError(f"Unknown bounding method: {bound}")
    # Sampling method.
    sample = _get_internal_sampler(sample, ndim, ncdim, periodic, reflective,
                                   walks, slices, facc)

    # Custom sampler
    if sample not in INTERNAL_SAMPLER_LIST and not isinstance(
            sample, InternalSampler):
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    if ncdim != ndim and (isinstance(sample, SliceSampler)
                          or isinstance(sample, RSliceSampler)):
        raise ValueError('ncdim unsupported for slice sampling')

    ret['sample'] = sample

    # Random state.
    if rstate is None:
        rstate = get_random_generator()
    ret['rstate'] = rstate

    # Keyword arguments controlling the first update.
    if first_update is None:
        first_update = {}
    else:
        _check_first_update(first_update)
    ret['first_bound_update'] = first_update

    # Prior transform.
    ptform_args = ptform_args or []
    ptform_kwargs = ptform_kwargs or {}
    # Wrap functions.
    prior_transform_wrap = _function_wrapper(prior_transform,
                                             ptform_args,
                                             ptform_kwargs,
                                             name='prior_transform')
    ret['prior_transform_wrap'] = prior_transform_wrap

    # Set up parallel (or serial) evaluation.
    mapper, queue_size = _parse_pool_queue(pool, queue_size)
    use_pool = use_pool or {}
    ret['use_pool'] = use_pool
    ret['mapper'] = mapper
    ret['queue_size'] = queue_size
    ret['pool'] = pool
    if use_pool.get('loglikelihood', True):
        pool_logl = pool
    else:
        pool_logl = None

    # Log-likelihood.
    logl_args = logl_args or []
    logl_kwargs = logl_kwargs or {}
    save_history = save_history or False
    save_evaluation_history = save_evaluation_history or False
    blob = blob or False
    default_logl_history_name = 'dynesty_logl_history.h5'
    loglikelihood_wrap = LogLikelihood(
        _function_wrapper(loglikelihood,
                          logl_args,
                          logl_kwargs,
                          name='loglikelihood'),
        ndim,
        pool=pool_logl,
        history_filename=history_filename or default_logl_history_name,
        save=save_history,
        blob=blob,
        save_evaluation_history=save_evaluation_history)
    ret['loglikelihood_wrap'] = loglikelihood_wrap

    update_interval_ratio = _get_update_interval_ratio(update_interval, sample,
                                                       nlive)
    ret['update_interval_ratio'] = update_interval_ratio

    # Citation generator.
    if dynamic:
        ret['cite'] = _get_citations('dynamic', bound, sample)
    else:
        ret['cite'] = _get_citations('static', bound, sample)

    # Bounding distribution modifications.
    enlarge, bootstrap = _get_enlarge_bootstrap(sample, enlarge, bootstrap)
    ret['bound_enlarge'] = enlarge
    ret['bound_bootstrap'] = bootstrap

    return ret


class NestedSampler(Sampler):
    """
    The main class performing the static nested sampling.
    It inherits all the methods of the dynesty.sampler.Sampler.
    """

    def __new__(cls,
                loglikelihood,
                prior_transform,
                ndim,
                nlive=500,
                bound='multi',
                sample='auto',
                periodic=None,
                reflective=None,
                update_interval=None,
                first_update=None,
                rstate=None,
                queue_size=None,
                pool=None,
                use_pool=None,
                live_points=None,
                logl_args=None,
                logl_kwargs=None,
                ptform_args=None,
                ptform_kwargs=None,
                enlarge=None,
                bootstrap=None,
                walks=None,
                facc=0.5,
                slices=None,
                ncdim=None,
                blob=False,
                save_history=False,
                history_filename=None,
                save_evaluation_history=False):

        params = _common_sampler_init(
            nlive=nlive,
            ndim=ndim,
            ncdim=ncdim,
            bound=bound,
            sample=sample,
            walks=walks,
            slices=slices,
            rstate=rstate,
            periodic=periodic,
            reflective=reflective,
            bootstrap=bootstrap,
            enlarge=enlarge,
            first_update=first_update,
            blob=blob,
            facc=facc,
            prior_transform=prior_transform,
            ptform_args=ptform_args,
            ptform_kwargs=ptform_kwargs,
            loglikelihood=loglikelihood,
            logl_args=logl_args,
            logl_kwargs=logl_kwargs,
            use_pool=use_pool,
            pool=pool,
            queue_size=queue_size,
            save_history=save_history,
            history_filename=history_filename,
            save_evaluation_history=save_evaluation_history,
            update_interval=update_interval,
            dynamic=False)

        update_interval = int(
            max(
                min(np.round(params['update_interval_ratio'] * nlive),
                    sys.maxsize), 1))

        live_points, logvol_init, init_ncalls = _initialize_live_points(
            live_points,
            params['prior_transform_wrap'],
            params['loglikelihood_wrap'],
            params['mapper'],
            nlive=nlive,
            ndim=ndim,
            rstate=params['rstate'],
            blob=blob,
            use_pool_ptform=params['use_pool'].get('prior_transform', True))

        # Initialize our nested sampler.
        sampler = super().__new__(Sampler)
        sampler.__init__(params['loglikelihood_wrap'],
                         params['prior_transform_wrap'],
                         ndim,
                         live_points,
                         params['sample'],
                         bound,
                         ncdim=params['ncdim'],
                         rstate=params['rstate'],
                         pool=params['pool'],
                         use_pool=params['use_pool'],
                         queue_size=params['queue_size'],
                         bound_update_interval=update_interval,
                         first_bound_update=params['first_bound_update'],
                         bound_bootstrap=params['bound_bootstrap'],
                         bound_enlarge=params['bound_enlarge'],
                         cite=params['cite'],
                         blob=blob,
                         logvol_init=logvol_init)
        sampler.ncall = init_ncalls
        return sampler


NestedSampler.__new__.__doc__ = _assemble_sampler_docstring(False)
NestedSampler.__init__.__doc__ = _assemble_sampler_docstring(False)


class DynamicNestedSampler(DynamicSampler):
    """
    The main class for performing dynamic nested sampling.
    It inherits all the methods from dynesty.dynamicsampler.DynamicSampler
    """

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 ndim,
                 nlive=500,
                 bound='multi',
                 sample='auto',
                 periodic=None,
                 reflective=None,
                 update_interval=None,
                 first_update=None,
                 rstate=None,
                 queue_size=None,
                 pool=None,
                 use_pool=None,
                 logl_args=None,
                 logl_kwargs=None,
                 ptform_args=None,
                 ptform_kwargs=None,
                 enlarge=None,
                 bootstrap=None,
                 walks=None,
                 facc=0.5,
                 slices=None,
                 ncdim=None,
                 blob=False,
                 save_history=False,
                 history_filename=None,
                 save_evaluation_history=False):

        params = _common_sampler_init(
            nlive=nlive,
            ndim=ndim,
            ncdim=ncdim,
            bound=bound,
            sample=sample,
            walks=walks,
            slices=slices,
            rstate=rstate,
            periodic=periodic,
            reflective=reflective,
            bootstrap=bootstrap,
            enlarge=enlarge,
            first_update=first_update,
            blob=blob,
            facc=facc,
            prior_transform=prior_transform,
            ptform_args=ptform_args,
            ptform_kwargs=ptform_kwargs,
            loglikelihood=loglikelihood,
            logl_args=logl_args,
            logl_kwargs=logl_kwargs,
            use_pool=use_pool,
            pool=pool,
            queue_size=queue_size,
            save_history=save_history,
            history_filename=history_filename,
            save_evaluation_history=save_evaluation_history,
            update_interval=update_interval,
            dynamic=True)

        # Initialize our nested sampler.
        super().__init__(
            params['loglikelihood_wrap'],
            params['prior_transform_wrap'],
            ndim,
            params['sample'],
            bound,
            nlive0=nlive,
            ncdim=params['ncdim'],
            rstate=params['rstate'],
            pool=params['pool'],
            use_pool=params['use_pool'],
            queue_size=params['queue_size'],
            bound_update_interval_ratio=params['update_interval_ratio'],
            first_bound_update=params['first_bound_update'],
            bound_bootstrap=params['bound_bootstrap'],
            bound_enlarge=params['bound_enlarge'],
            cite=params['cite'],
            blob=blob)


DynamicNestedSampler.__init__.__doc__ = _assemble_sampler_docstring(True)


class _function_wrapper:
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
            # IMPORTANT
            # Here we make a copy if the input vector just to ensure
            # that users can safely modify in-place the arguments to
            # say prior_transform or likelihood
            # This comes at performance cost, but it's worthwhile
            # as it may lead to hard to diagnose weird behaviour
            return self.func(np.asarray(x).copy(), *self.args, **self.kwargs)
        except:  # noqa
            print(f"Exception while calling {self.name} function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
