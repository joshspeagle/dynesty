#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains the dynamic nested sampler class :class:`DynamicSampler` used to
dynamically allocate nested samples. Note that :class:`DynamicSampler`
implicitly wraps a sampler from :mod:`~dynesty.nestedsamplers`. Also contains
the weight function :meth:`weight_function` and stopping function
:meth:`stopping_function`. These are used by default within
:class:`DynamicSampler` if corresponding functions are not provided
by the user.

"""

import sys
import warnings
import math
import copy
from enum import Enum
import numpy as np
from scipy.special import logsumexp
from .nestedsamplers import (UnitCubeSampler, SingleEllipsoidSampler,
                             MultiEllipsoidSampler, RadFriendsSampler,
                             SupFriendsSampler)
from .results import Results
from .utils import (get_seed_sequence, get_print_func, _kld_error,
                    compute_integrals, IteratorResult, IteratorResultShort,
                    get_enlarge_bootstrap, RunRecord, get_neff_from_logwt,
                    DelayTimer, save_sampler, restore_sampler, _LOWL_VAL)

__all__ = [
    "DynamicSampler",
    "weight_function",
    "stopping_function",
]

_SAMPLERS = {
    'none': UnitCubeSampler,
    'single': SingleEllipsoidSampler,
    'multi': MultiEllipsoidSampler,
    'balls': RadFriendsSampler,
    'cubes': SupFriendsSampler
}


class DynamicSamplerStatesEnum(Enum):
    """ """
    INIT = 1  # after the constructor
    LIVEPOINTSINIT = 2  # after generating livepoints
    INBASE = 3  # during base runs
    BASE_DONE = 4  # base run done
    INBATCH = 5  # after at least one batch
    BATCH_DONE = 6  # after at least one batch
    INBASEADDLIVE = 7  # during addition of livepoints in the
    INBATCHADDLIVE = 8  # during addition of livepoints in the
    RUN_DONE = 9  # The run has ended
    # end of the base run


def compute_weights(results):
    """ Derive evidence and posterior weights.
    return two arrays, evidence weights and posterior weights
    """
    logl = results.logl
    logz = results.logz  # final ln(evidence)
    logvol = results.logvol
    logwt = results.logwt
    samples_n = results.samples_n

    if np.ptp(logz) == 0:
        # this pathological case can happen if all logl are very small
        # and all logz are very small and the same
        # then the calculation below failse
        warnings.warn('''The calculation of weights is seeing same
logz values associated with all the samples. It may mean somethings is
wrong with your likelihood.''')
        zweight = np.ones(len(logl)) / len(logl)
    else:
        # TODO the logic here needs to be verified
        logz_remain = logl[-1] + logvol[-1]  # remainder
        logz_tot = np.logaddexp(logz[-1], logz_remain)  # estimated upper bound
        lzones = np.ones_like(logz)
        logzin = logsumexp([lzones * logz_tot, logz],
                           axis=0,
                           b=[lzones, -lzones])  # ln(remaining evidence)
        logzweight = logzin - np.log(samples_n)  # ln(evidence weight)
        logzweight -= logsumexp(logzweight)  # normalize
        zweight = np.exp(logzweight)  # convert to linear scale

    # Derive posterior weights.
    pweight = np.exp(logwt - logz[-1])  # importance weight
    pweight /= np.sum(pweight)  # normalize
    return zweight, pweight


def weight_function(results, args=None, return_weights=False):
    """
    The default weight function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns each point a weight based on a weighted average of the
    posterior and evidence information content::

        weight = pfrac * pweight + (1. - pfrac) * zweight

    where `pfrac` is the fractional importance placed on the posterior,
    the evidence weight `zweight` is based on the estimated remaining
    posterior mass, and the posterior weight `pweight` is the sample's
    importance weight.

    Returns a set of log-likelihood bounds set by the earliest/latest
    samples where `weight > maxfrac * max(weight)`, with additional
    left/right padding based on `pad`.

    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the log-likelihood bounds used for sampling,
        as described above. Default values are `pfrac = 0.8`, `maxfrac = 0.8`,
        and `pad = 1`.

    return_weights : bool, optional
        Whether to return the individual weights (and their components) used
        to compute the log-likelihood bounds. Default is `False`.

    Returns
    -------
    logl_bounds : tuple with shape (2,)
        Log-likelihood bounds `(logl_min, logl_max)` determined by the weights.

    weights : tuple with shape (3,), optional
        The individual weights `(pweight, zweight, weight)` used to determine
        `logl_bounds`.

    """

    # Initialize hyperparameters.
    if args is None:
        args = {}
    pfrac = args.get('pfrac', 0.8)
    if not 0. <= pfrac <= 1.:
        raise ValueError(
            f"The provided `pfrac` {pfrac} is not between 0. and 1.")
    maxfrac = args.get('maxfrac', 0.8)
    if not 0. <= maxfrac <= 1.:
        raise ValueError(
            f"The provided `maxfrac` {maxfrac} is not between 0. and 1.")
    lpad = args.get('pad', 1)
    if lpad < 0:
        raise ValueError(f"`lpad` {lpad} is less than zero.")

    zweight, pweight = compute_weights(results)

    # Compute combined weights.
    weight = (1. - pfrac) * zweight + pfrac * pweight

    # Compute logl bounds
    # we pad by lpad on each side (2lpad total)
    # if this brings us outside the range on on side, I add it on another
    nsamps = len(weight)
    bounds = np.nonzero(weight > maxfrac * np.max(weight))[0]
    bounds = (bounds[0] - lpad, bounds[-1] + lpad)
    logl = results.logl
    if bounds[1] > nsamps - 1:
        # overflow on the RHS, so we move the left side
        bounds = [bounds[0] - (bounds[1] - (nsamps - 1)), nsamps - 1]
    if bounds[0] <= 0:
        # if we overflow on the leftside we set the edge to -inf and expand
        # the RHS
        logl_min = -np.inf
        logl_max = logl[min(bounds[1] - bounds[0], nsamps - 1)]
    else:
        logl_min, logl_max = logl[bounds[0]], logl[bounds[1]]

    if bounds[1] == nsamps - 1:
        logl_max = np.inf
    if return_weights:
        return (logl_min, logl_max), (pweight, zweight, weight)

    return (logl_min, logl_max)


def _get_update_interval_ratio(update_interval, sample, bound, ndim, nlive,
                               slices, walks):
    """
    Get the update_interval divided by the number of live points
    """
    if update_interval is None:
        if sample == 'unif':
            update_interval_frac = 1.5
        elif sample == 'rwalk':
            update_interval_frac = 0.15 * walks
        elif sample == 'slice':
            update_interval_frac = 0.9 * ndim * slices
        elif sample == 'rslice':
            update_interval_frac = 2.0 * slices
        elif sample == 'hslice':
            update_interval_frac = 25.0 * slices
        else:
            update_interval_frac = np.inf
            warnings.warn(
                "No update_interval set with unknown sampling method: "
                f"'{sample}'. Defaulting to no updates.")
    elif isinstance(update_interval, float):
        update_interval_frac = update_interval
    elif isinstance(update_interval, int):
        update_interval_frac = update_interval * 1. / nlive
    else:
        raise RuntimeError(f'Strange update_interval value {update_interval}')
    if bound == 'none':
        update_interval_frac = np.inf
    return update_interval_frac


def stopping_function(results,
                      args=None,
                      rstate=None,
                      M=None,
                      return_vals=False):
    """
    The default stopping function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns the run a stopping value based on a weighted average of the
    stopping values for the posterior and evidence::

        stop = pfrac * stop_post + (1.- pfrac) * stop_evid

    The evidence stopping value is based on the estimated evidence error
    (i.e. standard deviation) relative to a given threshold::

        stop_evid = evid_std / evid_thresh

    The posterior stopping value is based on the estimated effective number
    of samples.

        stop_post = target_n_effective / n_effective

    Estimates of the mean and standard deviation are computed using `n_mc`
    realizations of the input using a provided `'error'` keyword (either
    `'jitter'` or `'resample'`, which call related functions :meth:`jitter_run`
    and :meth:`resample_run` in :mod:`dynesty.utils`, respectively.

    Returns the boolean `stop <= 1`. If `True`, the :class:`DynamicSampler`
    will stop adding new samples to our results.

    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the stopping values. Default values are
        `pfrac = 1.0`, `evid_thresh = 0.1`, `target_n_effective = 10000`,
        `n_mc = 0`, `error = 'jitter'`, and `approx = True`.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    M : `map` function, optional
        An alias to a `map`-like function. This allows users to pass
        functions from pools (e.g., `pool.map`) to compute realizations in
        parallel. By default the standard `map` function is used.

    return_vals : bool, optional
        Whether to return the stopping value (and its components). Default
        is `False`.

    Returns
    -------
    stop_flag : bool
        Boolean flag indicating whether we have passed the desired stopping
        criteria.

    stop_vals : tuple of shape (3,), optional
        The individual stopping values `(stop_post, stop_evid, stop)` used
        to determine the stopping criteria.

    """

    # Initialize values.
    if args is None:
        args = {}
    if M is None:
        M = map

    # Initialize hyperparameters.
    pfrac = args.get('pfrac', 1.0)
    if not 0. <= pfrac <= 1.:
        raise ValueError(
            f"The provided `pfrac` {pfrac} is not between 0. and 1.")
    evid_thresh = args.get('evid_thresh', 0.1)
    if pfrac < 1. and evid_thresh < 0.:
        raise ValueError(
            f"The provided `evid_thresh` {evid_thresh} is not non-negative "
            f"even though `pfrac` is {pfrac}.")
    target_n_effective = args.get('target_n_effective', 10000)

    if pfrac > 0. and target_n_effective < 0.:
        raise ValueError(
            f"The provided `target_n_effective` {target_n_effective} " +
            f"is not non-negative even though `pfrac` is {pfrac}")
    n_mc = args.get('n_mc', 0)
    if n_mc < 0:
        raise ValueError(f"The number of realizations {n_mc} must be greater "
                         "or equal to zero.")
    if n_mc > 0 and n_mc < 20:
        warnings.warn("Using a small number of realizations might result in "
                      "excessively noisy stopping value estimates.")
    error = args.get('error', 'jitter')
    if error not in {'jitter', 'resample'}:
        raise ValueError(f"The chosen `'error'` option {error} is not valid.")
    approx = args.get('approx', True)

    if n_mc > 1:
        # Compute realizations of ln(evidence) and the KL divergence.
        rlist = [results for i in range(n_mc)]
        error_list = [error for i in range(n_mc)]
        approx_list = [approx for i in range(n_mc)]
        seeds = get_seed_sequence(rstate, n_mc)
        args = zip(rlist, error_list, approx_list, seeds)
        outputs = list(M(_kld_error, args))
        lnz_arr = np.array([res[1].logz[-1] for res in outputs])
        # Evidence stopping value.
        lnz_std = np.std(lnz_arr)
    else:
        lnz_std = results.logzerr[-1]

    stop_evid = lnz_std / evid_thresh

    n_effective = get_neff_from_logwt(results.logwt)
    stop_post = target_n_effective / n_effective

    # Effective stopping value.
    stop = pfrac * stop_post + (1. - pfrac) * stop_evid

    if return_vals:
        return stop <= 1., (stop_post, stop_evid, stop)
    else:
        return stop <= 1.


def _initialize_live_points(live_points,
                            prior_transform,
                            loglikelihood,
                            M,
                            nlive=None,
                            ndim=None,
                            rstate=None,
                            blob=False,
                            use_pool_ptform=None):
    """
    Initialize the first set of live points before starting the sampling

    Parameters
    ----------
    live_points: tuple of arrays or None
        This can be either none or
        tuple of 3 arrays (u, v, logl) or
        tuple of 4 arrays (u, v, logl, blobs), i.e.
        points location in cube coordinates,
        point slocation in original coordinates,
        logl values and optionally blobs associated

    prior_transform: function

    log_likelihood: function

    M: function
        The function supporting parallel calls like M(func, list)

    nlive: int
        Number of live-points

    ndim: int
        Number of dimensions

    rstate: :class: numpy.random.RandomGenerator

    blob: bool
        If true we also keep track of blobs returned by likelihood

    use_pool_ptform: bool or None
        The flag to perform prior transform using multiprocessing pool or not

    Returns
    -------
    (live_u, live_v, live_logl, blobs), logvol_init, ncalls : tuple
        The first tuple consist of:
        live_u Unit cube coordinates of points
        live_v Original coordinates.
        live_logl log-likelihood values of points
        blobs - Array of blobs associated with logl calls (or None)
        The other arguments are
        logvol_init Log(volume) associated with returned points.
               It will be zero, if all the log(l) values were finite
        ncalls Integer number of function calls
    """
    logvol_init = 0
    ncalls = 0
    if live_points is None:
        # If no live points are provided, propose them by randomly
        # sampling from the unit cube.
        n_attempts = 1000

        min_npoints = min(nlive, max(ndim + 1, min(nlive - 20, 100)))
        # the minimum number points we want with finite logl
        # we want want at least ndim+1, because we want
        # to be able to constraint the ellipsoid
        # Note that if nlive <ndim+ 1 this doesn't really make sense
        # but we should have warned the user earlier, so they are on their own
        # And the reason we have max(ndim+1, X ) is that we'd like to get at
        # least X points as otherwise the poisson estimate of the volume will
        # be too large.
        # The reason why X is min(nlive-20, 100) is that we want at least 100
        # to have reasonable volume accuracy of ~ 10%
        # and the reason for nlive-20 is because if nlive is 100, we don't want
        # all points with finite logl, because this leads to issues with
        # integrals and batch sampling in plateau edge tests
        # The formula probably should be simplified

        live_u = np.zeros((nlive, ndim))
        live_v = np.zeros((nlive, ndim))
        live_logl = np.zeros(nlive)
        ngoods = 0  # counter for how many finite logl we have found
        live_blobs = []
        iattempt = 0
        while True:
            iattempt += 1

            # simulate nlive points by uniform sampling
            cur_live_u = rstate.random(size=(nlive, ndim))
            if use_pool_ptform:
                cur_live_v = M(prior_transform, np.asarray(cur_live_u))
            else:
                cur_live_v = map(prior_transform, np.asarray(cur_live_u))
            cur_live_v = np.array(list(cur_live_v))
            cur_live_logl = loglikelihood.map(np.asarray(cur_live_v))
            if blob:
                cur_live_blobs = np.array([_.blob for _ in cur_live_logl])
            cur_live_logl = np.array([_.val for _ in cur_live_logl])
            ncalls += nlive

            # Convert all `-np.inf` log-likelihoods to finite large
            # numbers. Necessary to keep estimators in our sampler from
            # breaking.
            finite = np.isfinite(cur_live_logl)
            not_finite = ~finite
            neg_infinite = np.isneginf(cur_live_logl)
            if np.any(not_finite & (~neg_infinite)):
                raise ValueError("The log-likelihood of live "
                                 "point is invalid.")
            cur_live_logl[not_finite] = _LOWL_VAL

            # how many finite logl values we have
            cur_ngood = finite.sum()
            if cur_ngood > 0:
                # append them to our list
                nextra = min(nlive - ngoods, cur_ngood)
                assert nextra >= 0
                cur_ind = np.nonzero(finite)[0][:nextra]
                live_logl[ngoods:ngoods + nextra] = cur_live_logl[cur_ind]
                live_u[ngoods:ngoods + nextra] = cur_live_u[cur_ind]
                live_v[ngoods:ngoods + nextra] = cur_live_v[cur_ind]
                if blob:
                    live_blobs.extend(cur_live_blobs[cur_ind])
                ngoods += nextra

            # Check if we have more than the minimum required number
            # after that we will stop
            if ngoods >= min_npoints:
                # we need to fill the rest with points with
                # not finite logl
                nextra = nlive - ngoods
                if nextra > 0:
                    cur_ind = np.nonzero(not_finite)[0][:nextra]
                    assert len(cur_ind) == nextra
                    live_logl[ngoods:ngoods + nextra] = cur_live_logl[cur_ind]
                    live_u[ngoods:ngoods + nextra] = cur_live_u[cur_ind]
                    live_v[ngoods:ngoods + nextra] = cur_live_v[cur_ind]
                    if blob:
                        live_blobs.extend(cur_live_blobs[cur_ind])
                logvol_init = -np.log(iattempt)
                # The logic is the following:
                # if we have n live points and we sampled N attempts
                # and we have k points above LOWL_VAL
                # then the volume associated with pts above LOWL_VAL
                # can be estimated as k/(Nn)
                # the rest of the points have 1/Nn volume per pt
                # Since we quit with k points above LOWL_VAL and
                # (n-k)  LOWL points
                # The volume is k/(Nn) + (n-k)/(Nn) = 1/N
                break
            if iattempt == n_attempts:
                if ngoods == 0:
                    # If we found nothing after many attempts, raise the alarm.
                    raise RuntimeError(
                        f"After {n_attempts} attempts, we cound not "
                        "find a single point "
                        "that have a valid log-likelihood! Please "
                        "check your prior transform and/or "
                        "log-likelihood.")
                else:
                    # If we found nothing after many attempts, raise the alarm.
                    warnings.warn(f"After {n_attempts} attempts, we cound not "
                                  f"find at least {min_npoints} points "
                                  "that have a valid log-likelihood! "
                                  "The initial sampling is very inefficient!")

    else:
        # If live points were provided, convert the log-likelihoods and
        # then run a quick safety check.
        live_u, live_v, live_logl = live_points[:3]
        if blob:
            live_blobs = live_points[3]
        live_logl = np.asarray(live_logl)
        for i, logl in enumerate(live_logl):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    live_logl[i] = _LOWL_VAL
                else:
                    raise ValueError("The log-likelihood ({0}) of live "
                                     "point {1} located at u={2} v={3} "
                                     " is invalid.".format(
                                         logl, i, live_u[i], live_v[i]))
        if np.all(live_logl == _LOWL_VAL):
            raise ValueError("Not a single provided live point has a "
                             "valid log-likelihood!")
    if np.ptp(live_logl) == 0:
        warnings.warn(
            'All the initial likelihood values are the same. '
            'You likely have a plateau in the likelihood. '
            'Nested sampling may not be the best sampler in this case.',
            RuntimeWarning)
    if not blob:
        live_blobs = None
    return (live_u, live_v, live_logl, live_blobs), logvol_init, ncalls


def _configure_batch_sampler(main_sampler,
                             nlive_new,
                             update_interval,
                             logl_bounds=None,
                             save_bounds=None):
    """
    This is a utility method that construct a new internal
    sampler that will sample one batch.
    Since the setting up requires us coming up with a set of
    of starting live points we also prepare the first list points
    that will be yielded by the "master" sampler (but those are
    yielded just for printing).

    This method should not modify the parent sampler at all other
    than using the parent's random number generator

    Parameters
    ----------
    main_sampler: DynamicNestedSampler
        The parent sampler that we'll use to initialize a new sampler

    nlive_new: integer
        The number of live-points in the new sampler

    update_interval : int
        Only update the bounding distribution
        every `update_interval`-th likelihood call. Note that
        here it must be integer

    logl_bounds: tuple
        Tuple of bounds in loglikelihood for the batch

    save_bounds: bool
        If true bounds will be preserved

    Returns
    -------
    batch_sampler: Sampler
        The sampler that will actually execute the batch.
        It will also have a first_points attribute with the list
        of IteratorResultShorts
    ncall: integer
        Number of likelihood calls throughout
    niter: integer
        Number of iterations
    logl_min: float
        actually used logl_min as we may modify logl_bound
    logl_max: float
        actually used logl_max
    """

    # Things to consider in this method
    # Because we are not yielding samples from this method directly, we
    # are not expecting that the .save() will save a state
    # before or in the middle of running this method

    # Counters of calls and iterations throughout.
    ncall = 0
    niter = 0

    # Grab results from saved run.
    saved_u = np.array(main_sampler.saved_run['u'])
    saved_v = np.array(main_sampler.saved_run['v'])
    saved_logl = np.array(main_sampler.saved_run['logl'])
    saved_logvol = np.array(main_sampler.saved_run['logvol'])
    saved_scale = np.array(main_sampler.saved_run['scale'])
    saved_blobs = np.array(main_sampler.saved_run['blob'])
    first_points = []

    # This will be a list of first points yielded from
    # this batch before we start proper sampling
    batch_sampler = _SAMPLERS[main_sampler.bounding](
        main_sampler.loglikelihood,
        main_sampler.prior_transform,
        main_sampler.ndim,
        main_sampler.live_init,  # this is not used at all
        # as we replace the starting points
        main_sampler.method,
        update_interval,
        main_sampler.first_update,
        main_sampler.rstate,
        main_sampler.queue_size,
        main_sampler.pool,
        main_sampler.use_pool,
        ncdim=main_sampler.ncdim,
        kwargs=main_sampler.kwargs,
        blob=main_sampler.blob)
    batch_sampler.save_bounds = save_bounds
    batch_sampler.logl_first_update = main_sampler.sampler.logl_first_update

    # Initialize ln(likelihood) bounds.
    if logl_bounds is None:
        # the reason we set logl_max to not the highest logl
        # is because the last few points are always added in the end
        # without sampling through add_live_points()
        # so here I pick up the first point where the volume is
        # Vmin*nlive where Vmin is the smallest volume previously seen.
        logl_max_pos = np.nonzero(saved_logvol < (saved_logvol[-1] +
                                                  np.log(nlive_new)))[0]
        if len(logl_max_pos) > 0:
            logl_max_pos = logl_max_pos[-1]
        else:
            logl_max_pos = len(saved_logl) - 1
        logl_min, logl_max = -np.inf, saved_logl[logl_max_pos]
    else:
        logl_min, logl_max = logl_bounds
    # IMPORTANT we update these in the process

    # Check whether the lower bound encompasses all previous saved
    # samples.
    psel = np.all(saved_logl > logl_min)
    if psel:
        # If the lower bound encompasses all saved samples, we want
        # to propose a new set of points from the unit cube.
        (live_u, live_v, live_logl,
         live_blobs), logvol0, init_ncalls = _initialize_live_points(
             None,
             main_sampler.prior_transform,
             main_sampler.loglikelihood,
             main_sampler.M,
             nlive=nlive_new,
             ndim=main_sampler.ndim,
             rstate=main_sampler.rstate,
             blob=main_sampler.blob,
             use_pool_ptform=main_sampler.use_pool_ptform)
        live_bound = np.zeros(nlive_new, dtype=int)
        live_it = np.zeros(nlive_new, dtype=int)
        live_nc = np.ones(nlive_new, dtype=int)
        # we should have evaluated the function once per point

        ncall += init_ncalls
        # Return live points in generator format.
        for i in range(nlive_new):
            # TODO is the self.eff the right efficiency to use here
            # check worst_it
            first_points.append(
                IteratorResultShort(worst=-i - 1,
                                    ustar=live_u[i],
                                    vstar=live_v[i],
                                    loglstar=live_logl[i],
                                    nc=1,
                                    worst_it=live_it[i] + main_sampler.it,
                                    boundidx=0,
                                    bounditer=0,
                                    eff=main_sampler.eff))
        batch_sampler.update_bound_if_needed(logl_min)
        # Trigger an update of the internal bounding distribution based
        # on the "new" set of live points.
    else:
        # If the lower bound doesn't encompass all base samples,
        # we need to create a uniform sample from the prior subject
        # to the likelihood boundary constraint

        subset0 = np.nonzero(saved_logl > logl_min)[0]

        if len(subset0) == 0:
            raise RuntimeError('Could not find live points in the '
                               'required logl interval. Please report!\n'
                               f'Diagnostics. logl_min: {logl_min} '
                               f'logl_bounds: {logl_bounds} '
                               f'saved_loglmax: {saved_logl.max()}')

        # Also if we don't have enough live points above the boundary
        # we simply go down to collect our nlive_new points
        if len(subset0) < nlive_new:
            if len(saved_logl) < nlive_new:
                # It means we don't even have nlive_new points
                # in our base runs so we just take everything
                subset0 = np.arange(len(saved_logl))
            else:
                # otherwise we just move the boundary down
                # to collect our nlive_new points
                subset0 = np.arange(subset0[-1] - nlive_new + 1,
                                    subset0[-1] + 1)
            # IMPORTANT We have to update the lower bound for sampling
            # otherwise some of our live points do not satisfy it

            # we want our points to be strictly above the logl_min
            if subset0[0] > 0:
                logl_min = saved_logl[subset0[0] - 1]
            else:
                logl_min = -np.inf

        live_scale = saved_scale[subset0[0]]
        # set the scale based on the lowest point

        # we are weighting each point by X_i to ensure
        # uniformyish sampling within boundary volume
        # It doesn't have to be super uniform as we'll sample
        # again, but still
        cur_log_uniwt = saved_logvol[subset0]
        cur_uniwt = np.exp(cur_log_uniwt - cur_log_uniwt.max())
        cur_uniwt = cur_uniwt / cur_uniwt.sum()
        # I normalize in linear space rather then using logsumexp
        # because cur_uniwt.sum() needs to be 1 for random.choice

        # we are now randomly sampling with weights
        # notice that since we are sampling without
        # replacement we aren't guaranteed to be able
        # to get nlive_new points
        # so we get min(nlive_new,subset.sum())
        # in that case the sample technically won't be
        # uniform
        n_pos_weight = (cur_uniwt > 0).sum()

        subset = main_sampler.rstate.choice(subset0,
                                            size=min(nlive_new, n_pos_weight),
                                            p=cur_uniwt,
                                            replace=False)
        # subset will now have indices of selected points from
        # saved_* arrays
        cur_nlive = len(subset)
        if cur_nlive == 1:
            raise RuntimeError('Only one live point is selected\n' +
                               'Please report the error on github!' +
                               f'Diagnostics nlive_new: {nlive_new} ' +
                               f'cur_nlive: {cur_nlive}' +
                               f'n_pos_weight: {n_pos_weight}' +
                               f'cur_wt: {cur_uniwt}')
        # We are doing copies here, because live_* stuff is
        # updated in place
        live_u = saved_u[subset, :].copy()
        live_v = saved_v[subset, :].copy()
        live_logl = saved_logl[subset].copy()
        live_blobs = saved_blobs[subset].copy()

        # Hack the internal sampler by overwriting the live points
        # and scale factor.
        batch_sampler.nlive = cur_nlive
        batch_sampler.live_u = live_u
        batch_sampler.live_v = live_v
        batch_sampler.live_logl = live_logl
        batch_sampler.scale = live_scale
        batch_sampler.live_blobs = live_blobs

        batch_sampler.update_bound_if_needed(logl_min)
        # Trigger an update of the internal bounding distribution based
        # on the "new" set of live points.

        live_u = np.empty((nlive_new, main_sampler.ndim))
        live_v = np.empty((nlive_new, saved_v.shape[1]))
        live_logl = np.empty(nlive_new)
        live_bound = np.zeros(nlive_new, dtype=int)
        live_it = np.zeros(nlive_new, dtype=int)

        live_nc = np.empty(nlive_new, dtype=int)
        if main_sampler.blob:
            live_blobs = []
        else:
            live_blobs = None

        # Sample a new batch of `nlive_new` live points using the
        # internal sampler given the `logl_min` constraint.
        for i in range(nlive_new):
            newpt = batch_sampler._new_point(logl_min)
            (live_u[i], live_v[i], live_logl[i], live_nc[i]) = newpt
            if main_sampler.blob:
                blob = newpt[2].blob
                live_blobs.append(blob)
            else:
                blob = None

            ncall += live_nc[i]

            # Return live points in generator format.
            # these won't be saved but just used for printing
            first_points.append(
                IteratorResultShort(
                    worst=-i - 1,
                    ustar=live_u[i],
                    vstar=live_v[i],
                    loglstar=live_logl[i],
                    nc=live_nc[i],
                    worst_it=live_it[i] + main_sampler.it,
                    boundidx=live_bound[i],
                    bounditer=live_bound[i],
                    eff=main_sampler.eff,
                ))
    niter += nlive_new
    # Overwrite the previous set of live points in our internal sampler
    # with the new batch of points we just generated.
    batch_sampler.nlive = nlive_new

    # All the arrays are newly created in this function
    # We don't need to worry about them being parts of other arrays
    batch_sampler.live_u = live_u
    batch_sampler.live_v = live_v
    batch_sampler.live_logl = live_logl
    batch_sampler.live_bound = live_bound
    batch_sampler.live_blobs = live_blobs
    batch_sampler.live_it = live_it

    if psel:
        batch_sampler.logvol_init = logvol0

    # Figure out where the new run would would join the previous run
    if logl_min == -np.inf:
        vol_idx = 0
    else:
        vol_idx = np.argmin(np.abs(saved_logl - logl_min)) + 1

    # truncate information in the saver of the internal sampler
    # to make it look like we are just continuing
    for k in batch_sampler.saved_run.keys():
        batch_sampler.saved_run[k] = main_sampler.saved_run[k][:vol_idx]

    batch_sampler.dlv = math.log((nlive_new + 1.) / nlive_new)

    batch_sampler.first_points = first_points
    # We save these points in the object to ensure we can
    # resume from an interrupted run
    return batch_sampler, ncall, niter, logl_min, logl_max


class DynamicSampler:
    """
    A dynamic nested sampler that allocates live points adaptively during
    a single run according to a specified weight function until a specified
    stopping criteria is reached.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    ndim : int, optional
        Number of parameters accepted by `prior_transform`.

    bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}, optional
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to
        propose new live points.

    method : {`'unif'`, `'rwalk'`,
        `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    ncdim: int
        Number of clustered dimensions

    nlive0: int
        Default number of live points to use

    kwargs : dict, optional
        A dictionary of additional parameters (described below).
    """

    def __init__(self, loglikelihood, prior_transform, ndim, bound, method,
                 update_interval_ratio, first_update, rstate, queue_size, pool,
                 use_pool, ncdim, nlive0, kwargs):

        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.ncdim = ncdim
        self.blob = kwargs.get('blob') or False
        # bounding/sampling
        self.bounding = bound
        self.method = method
        self.update_interval_ratio = update_interval_ratio
        self.first_update = first_update

        # internal sampler object
        self.sampler = None

        # extra arguments
        self.kwargs = kwargs

        self.enlarge, self.bootstrap = get_enlarge_bootstrap(
            method, kwargs.get('enlarge'), kwargs.get('bootstrap'))

        self.walks = self.kwargs.get('walks', 25)
        self.slices = self.kwargs.get('slices', 3)
        self.cite = self.kwargs.get('cite')
        self.custom_update = self.kwargs.get('update_func')

        # random state
        self.rstate = rstate

        # parallelism
        self.queue_size = queue_size
        self.pool = pool
        if self.pool is None:
            self.M = map
        else:
            self.M = pool.map

        self.use_pool = use_pool  # provided flags for when to use the pool
        self.use_pool_ptform = use_pool.get('prior_transform', True)
        self.use_pool_logl = use_pool.get('loglikelihood', True)
        self.use_pool_evolve = use_pool.get('propose_point', True)
        self.use_pool_update = use_pool.get('update_bound', True)
        self.use_pool_stopfn = use_pool.get('stop_function', True)

        # sampling details
        self.it = 1  # number of iterations
        self.batch = 0  # number of batches allocated dynamically
        self.ncall = 0  # number of function calls
        self.bound = []  # initial states used to compute bounds
        self.eff = 1.  # sampling efficiency
        self.base = False  # base run complete
        self.nlive0 = nlive0
        self.internal_state = DynamicSamplerStatesEnum.INIT

        self.saved_run = RunRecord(dynamic=True)
        self.base_run = RunRecord(dynamic=True)
        self.new_run = None

        self.new_logl_min, self.new_logl_max = -np.inf, np.inf
        # logl bounds of latest "new" run

        # these are set-up during sampling
        self.live_u = None
        self.live_v = None
        self.live_it = None
        self.live_bound = None
        self.live_logl = None
        self.live_init = None
        self.nlive_init = None
        self.batch_sampler = None
        self.checkpoint_timer = None
        # the reason why we need a global object is to
        # preserve the timer betweeen batch calls
        self.live_blobs = None

    def __setstate__(self, state):
        self.__dict__ = state
        self.pool = None
        self.M = map

    def __getstate__(self):
        """Get state information for pickling."""

        state = self.__dict__.copy()

        # deal with pool
        del state['pool']  # remove pool
        del state['M']  # remove `pool.map` function hook

        return state

    def save(self, fname):
        """
        Save the state of the dynamic sampler in a file

        Parameters
        ----------
        fname: string
            Filename of the save file.

        """
        save_sampler(self, fname)

    @staticmethod
    def restore(fname, pool=None):
        """
        Restore the dynamic sampler from a file.
        It is assumed that the file was created using .save() method
        of DynamicNestedSampler or as a result of checkpointing during
        run_nested()

        Parameters
        ----------
        fname: string
            Filename of the save file.
        pool: object(optional)
            The multiprocessing pool-like object that supports map()
            calls that will be used in the restored object.

        """
        return restore_sampler(fname, pool=pool)

    def __get_update_interval(self, update_interval, nlive):
        if not isinstance(update_interval, int):
            if isinstance(update_interval, float):
                cur_update_interval_ratio = update_interval
            elif update_interval is None:
                cur_update_interval_ratio = self.update_interval_ratio
            else:
                raise RuntimeError(
                    str.format('Weird update_interval value {}',
                               update_interval))
            update_interval = int(
                max(
                    min(np.round(cur_update_interval_ratio * nlive),
                        sys.maxsize), 1))
        return update_interval

    def reset(self):
        """Re-initialize the sampler."""

        # sampling
        self.it = 1
        self.batch = 0
        self.ncall = 0
        self.bound = []
        self.eff = 1.
        self.base = False

        self.saved_run = RunRecord(dynamic=True)
        self.base_run = RunRecord(dynamic=True)
        self.new_run = None
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

    @property
    def results(self):
        """Saved results from the dynamic nested sampling run. All saved
        bounds are also returned."""
        d = {}
        for k in [
                'nc', 'v', 'id', 'batch', 'it', 'u', 'n', 'logwt', 'logl',
                'logvol', 'logz', 'logzvar', 'h', 'batch_nlive',
                'batch_bounds', 'blob'
        ]:
            d[k] = np.array(self.saved_run[k])

        # Add all saved samples (and ancillary quantities) to the results.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [('niter', self.it - 1), ('ncall', d['nc']),
                       ('eff', self.eff), ('samples', d['v'])]
            for k in ['id', 'batch', 'it', 'u', 'n']:
                results.append(('samples_' + k, d[k]))
            for k in [
                    'logwt', 'logl', 'logvol', 'logz', 'batch_nlive',
                    'batch_bounds', 'blob'
            ]:
                results.append((k, d[k]))
            results.append(('logzerr', np.sqrt(d['logzvar'])))
            results.append(('information', d['h']))

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.sampler.save_bounds:
            results.append(('bound', copy.deepcopy(self.bound)))
            results.append(
                ('bound_iter', np.array(self.saved_run['bounditer'])))
            results.append(
                ('samples_bound', np.array(self.saved_run['boundidx'])))
            results.append(('scale', np.array(self.saved_run['scale'])))

        return Results(results)

    @property
    def n_effective(self):
        """
        Estimate the effective number of posterior samples using the Kish
        Effective Sample Size (ESS) where `ESS = sum(wts)^2 / sum(wts^2)`.
        Note that this is `len(wts)` when `wts` are uniform and
        `1` if there is only one non-zero element in `wts`.

        """
        logwt = self.saved_run['logwt']
        if len(logwt) == 0 or np.isneginf(np.max(logwt)):
            # If there are no saved weights, or its -inf return 0.
            return 0
        else:
            return get_neff_from_logwt(np.asarray(logwt))

    @property
    def citations(self):
        """
        Return list of papers that should be cited given the specified
        configuration of the sampler.

        """

        return self.cite

    def sample_initial(self,
                       nlive=None,
                       update_interval=None,
                       first_update=None,
                       maxiter=None,
                       maxcall=None,
                       logl_max=np.inf,
                       dlogz=0.01,
                       n_effective=np.inf,
                       live_points=None,
                       save_samples=False,
                       resume=False):
        """
        Generate a series of initial samples from a nested sampling
        run using a fixed number of live points using an internal
        sampler from :mod:`~dynesty.nestedsamplers`. Instantiates a
        generator that will be called by the user.

        Parameters
        ----------
        nlive : int, optional
            The number of live points to use for the baseline nested
            sampling run. Default is either nlive0 parameter of 500

        update_interval : int or float, optional
            If an integer is passed, only update the bounding distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the bound after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        first_update : dict, optional
            A dictionary containing parameters governing when the sampler will
            first update the bounding distribution from the unit cube
            (`'none'`) to the one specified by `sample`.

        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        n_effective: int, optional
            This option is deprecated and will be removed in a future release.

        live_points: list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            and optionally list of blobs associated with these likelihood calls
            (if blob=True in the sampler)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods. By default, if these are not provided the initial
            set of live points will be drawn from the unit `ndim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points will lead to incorrect results.**

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (ndim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        logvol : float
            Ln(prior volume) within the sample.

        logwt : float
            Ln(weight) of the sample.

        logz : float
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : float
            Estimated cumulative variance on `logz` (inclusive).

        h : float
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        delta_logz : float
            The estimated remaining evidence expressed as the ln(ratio) of the
            current evidence.

        """

        # Check for deprecated options
        if n_effective is not np.inf:
            with warnings.catch_warnings():
                warnings.filterwarnings("once")
                warnings.warn(
                    "The n_effective option to DynamicSampler.sample_initial "
                    "is deprecated and will be removed in future releases",
                    DeprecationWarning)

        # Initialize inputs.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        nlive = nlive or self.nlive0
        update_interval = self.__get_update_interval(update_interval, nlive)
        if nlive <= 2 * self.ncdim:
            warnings.warn("Beware: `nlive_init <= 2 * ndim`!")

        if not resume:
            # Reset saved results to avoid any possible conflicts.
            self.reset()

            (self.live_u, self.live_v, self.live_logl,
             blobs), logvol_init, init_ncalls = _initialize_live_points(
                 live_points,
                 self.prior_transform,
                 self.loglikelihood,
                 self.M,
                 nlive=nlive,
                 ndim=self.ndim,
                 rstate=self.rstate,
                 blob=self.blob,
                 use_pool_ptform=self.use_pool_ptform)
            if self.blob:
                self.live_blobs = blobs
            else:
                self.live_blobs = None
            self.nlive_init = len(self.live_u)

            # (Re-)bundle live points.
            live_points = [
                self.live_u, self.live_v, self.live_logl, self.live_blobs
            ]
            self.live_init = [np.array(_) for _ in live_points]
            self.ncall += init_ncalls
            self.live_bound = np.zeros(self.nlive_init, dtype=int)
            self.live_it = np.zeros(self.nlive_init, dtype=int)

            bounding = self.bounding

            if first_update is None:
                first_update = self.first_update
            self.sampler = _SAMPLERS[bounding](self.loglikelihood,
                                               self.prior_transform,
                                               self.ndim,
                                               self.live_init,
                                               self.method,
                                               update_interval,
                                               first_update,
                                               self.rstate,
                                               self.queue_size,
                                               self.pool,
                                               self.use_pool,
                                               ncdim=self.ncdim,
                                               kwargs=self.kwargs,
                                               blob=self.blob,
                                               logvol_init=logvol_init)
            self.bound = self.sampler.bound
            self.internal_state = DynamicSamplerStatesEnum.LIVEPOINTSINIT
            # Run the sampler internally as a generator.
        for it, results in enumerate(
                self.sampler.sample(maxiter=maxiter,
                                    save_samples=save_samples,
                                    maxcall=maxcall,
                                    logl_max=logl_max,
                                    dlogz=dlogz,
                                    resume=resume)):
            # Grab results.

            # Save our base run (which we will use later).
            add_info = dict(id=results.worst,
                            u=results.ustar,
                            v=results.vstar,
                            logl=results.loglstar,
                            logvol=results.logvol,
                            logwt=results.logwt,
                            logz=results.logz,
                            logzvar=results.logzvar,
                            h=results.h,
                            nc=results.nc,
                            it=results.worst_it,
                            n=self.nlive_init,
                            blob=results.blob,
                            boundidx=results.boundidx,
                            bounditer=results.bounditer,
                            scale=self.sampler.scale)

            self.base_run.append(add_info)
            self.saved_run.append(add_info)

            # Increment relevant counters.
            self.ncall += results.nc
            self.eff = 100. * self.it / self.ncall
            self.it += 1
            self.internal_state = DynamicSamplerStatesEnum.INBASE
            yield IteratorResult(worst=results.worst,
                                 ustar=results.ustar,
                                 vstar=results.vstar,
                                 loglstar=results.loglstar,
                                 logvol=results.logvol,
                                 logwt=results.logwt,
                                 logz=results.logz,
                                 logzvar=results.logzvar,
                                 h=results.h,
                                 nc=results.nc,
                                 blob=results.blob,
                                 worst_it=results.worst_it,
                                 boundidx=results.boundidx,
                                 bounditer=results.bounditer,
                                 eff=self.eff,
                                 delta_logz=results.delta_logz)
        self.internal_state = DynamicSamplerStatesEnum.INBASEADDLIVE
        for it, results in enumerate(self.sampler.add_live_points()):
            # Grab results.

            add_info = dict(id=results.worst,
                            u=results.ustar,
                            v=results.vstar,
                            logl=results.loglstar,
                            logvol=results.logvol,
                            logwt=results.logwt,
                            logz=results.logz,
                            logzvar=results.logzvar,
                            h=results.h,
                            blob=results.blob,
                            nc=results.nc,
                            it=results.worst_it,
                            n=self.nlive_init - it,
                            boundidx=results.boundidx,
                            bounditer=results.bounditer,
                            scale=self.sampler.scale)

            self.base_run.append(add_info)
            self.saved_run.append(add_info)

            # Increment relevant counters.
            self.eff = 100. * self.it / self.ncall
            self.it += 1
            yield IteratorResult(worst=results.worst,
                                 ustar=results.ustar,
                                 vstar=results.vstar,
                                 loglstar=results.loglstar,
                                 logvol=results.logvol,
                                 logwt=results.logwt,
                                 logz=results.logz,
                                 logzvar=results.logzvar,
                                 h=results.h,
                                 blob=results.blob,
                                 nc=results.nc,
                                 worst_it=results.worst_it,
                                 boundidx=results.boundidx,
                                 bounditer=results.bounditer,
                                 eff=self.eff,
                                 delta_logz=results.delta_logz)
        new_vals = {}
        (new_vals['logwt'], new_vals['logz'], new_vals['logzvar'],
         new_vals['h']) = compute_integrals(logl=self.saved_run['logl'],
                                            logvol=self.saved_run['logvol'])
        for curk in ['logwt', 'logz', 'logzvar', 'h']:
            self.saved_run[curk] = new_vals[curk].tolist()
            self.base_run[curk] = new_vals[curk].tolist()

        self.saved_run['batch'] = np.zeros(len(self.saved_run['id']),
                                           dtype=int)  # batch

        self.saved_run['batch_nlive'].append(self.nlive_init)  # initial nlive
        self.saved_run['batch_bounds'].append(
            (-np.inf, np.inf))  # initial bounds

        self.base = True  # baseline run complete
        self.internal_state = DynamicSamplerStatesEnum.BASE_DONE

    def sample_batch(self,
                     dlogz=0.01,
                     nlive_new=None,
                     update_interval=None,
                     logl_bounds=None,
                     maxiter=None,
                     maxcall=None,
                     save_bounds=True,
                     resume=False):
        """
        Generate an additional series of nested samples that will be combined
        with the previous set of dead points. Works by hacking the internal
        `sampler` object.
        Instantiates a generator that will be called by the user.

        Parameters
        ----------
        nlive_new : int
            Number of new live points to be added. Default is `500`.

        update_interval : int or float, optional
            If an integer is passed, only update the bounding distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the bound after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        logl_bounds : tuple of size (2,), optional
            The ln(likelihood) bounds used to bracket the run. If `None`,
            the default bounds span the entire range covered by the
            original run.

        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        dlogz : float, optional
            The stopping point in terms of remaining delta(logz)

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample. **Negative values indicate the index
            of a new live point generated when initializing a new batch.**

        ustar : `~numpy.ndarray` with shape (ndim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        """

        # Initialize default values.
        maxcall = maxcall or sys.maxsize
        maxiter = maxiter or sys.maxsize

        nlive_new = nlive_new or self.nlive0

        if nlive_new <= 2 * self.ncdim:
            warnings.warn("Beware: `nlive_batch <= 2 * ndim`!")

        # In the following code we are carefully trying to store everything
        # in attributes of batch_sampler rather than have individual
        # variables as otherwise we can't resume properly
        if not resume:
            update_interval = self.__get_update_interval(
                update_interval, nlive_new)
            (batch_sampler, ncall, niter, logl_min,
             logl_max) = _configure_batch_sampler(
                 self,
                 nlive_new,
                 update_interval=update_interval,
                 logl_bounds=logl_bounds,
                 save_bounds=save_bounds)
            self.batch_sampler = batch_sampler

            # TODO
            # This is not actually correct, and because of that
            # the bounds from base run or added batches are lost
            # Ideally bounds need to be saved somehow not just overwritten
            self.bound = self.batch_sampler.bound

            self.new_logl_min, self.new_logl_max = logl_min, logl_max
            # Reset "new" results.
            self.new_run = RunRecord(dynamic=True)
            self.ncall += ncall
            batch_sampler.it0 = self.it
            it0 = self.it
            # The tricky thing here is that we have here two sets of
            # iterations.
            # We have iterations of the batch_sampler and a parent
            # sampler and we need to make sure we translate one to another
            maxcall_left = maxcall - ncall
            maxiter_left = maxiter - niter
        else:
            batch_sampler = self.batch_sampler
            it0 = batch_sampler.it0
            logl_min, logl_max = self.new_logl_min, self.new_logl_max
            maxcall_left = maxcall
            maxiter_left = maxiter
            # I have decided that maxcall/maxiter_left will not be preserved
            # if interrupted and resumed

        for _ in range(len(batch_sampler.first_points)):
            yield batch_sampler.first_points.pop(0)
            # these yields are just for printing
            # we are not actually storing those in new_run
            # because we're not sampling yet, we've just
            # set up nlive_new points above our logl_min boundary
            # The reason why I'm popping the items is to ensure if we
            # are interrupted and then resume we don't start again

        iterated_batch = False
        # To identify if the loop below was executed or not

        for it, results in enumerate(
                batch_sampler.sample(dlogz=dlogz,
                                     logl_max=logl_max,
                                     maxiter=maxiter_left,
                                     maxcall=maxcall_left,
                                     save_samples=True,
                                     save_bounds=save_bounds,
                                     resume=resume)):
            # Save results.
            D = dict(id=results.worst,
                     u=results.ustar,
                     v=results.vstar,
                     logl=results.loglstar,
                     nc=results.nc,
                     it=results.worst_it + it0,
                     blob=results.blob,
                     n=nlive_new,
                     boundidx=results.boundidx,
                     bounditer=results.bounditer,
                     scale=batch_sampler.scale)
            self.new_run.append(D)
            # Increment relevant counters.
            self.ncall += results.nc
            self.eff = 100. * self.it / self.ncall
            self.it += 1
            maxiter_left -= 1
            maxcall_left -= results.nc
            iterated_batch = True
            self.internal_state = DynamicSamplerStatesEnum.INBATCH

            # These yields will be just for printing
            yield IteratorResultShort(worst=results.worst,
                                      ustar=results.ustar,
                                      vstar=results.vstar,
                                      loglstar=results.loglstar,
                                      nc=results.nc,
                                      worst_it=results.worst_it + it0,
                                      boundidx=results.boundidx,
                                      bounditer=results.bounditer,
                                      eff=self.eff)
        if iterated_batch and results.loglstar < logl_max and np.isfinite(
                logl_max) and maxiter_left > 0 and maxcall_left > 0:
            warnings.warn('Warning. The maximum likelihood not reached '
                          'in the batch. '
                          'You may not have enough livepoints and/or have '
                          'highly multi-modal distribution')
        self.internal_state = DynamicSamplerStatesEnum.INBATCHADDLIVE

        if not iterated_batch and len(batch_sampler.saved_run['logl']) == 0:
            # This is a special case *if* we only sampled the initial
            # livepoints but never did sample after
            batch_sampler.saved_run['logvol'] = [-np.inf]
            batch_sampler.saved_run['logl'] = [logl_min]
            batch_sampler.saved_run['logz'] = [-1e100]
            batch_sampler.saved_run['logzvar'] = [0]
            batch_sampler.saved_run['h'] = [0]
        for it, results in enumerate(batch_sampler.add_live_points()):
            # Save results.
            D = dict(id=results.worst,
                     u=results.ustar,
                     v=results.vstar,
                     logl=results.loglstar,
                     nc=results.nc,
                     it=results.worst_it + it0,
                     n=nlive_new - it,
                     blob=results.blob,
                     boundidx=results.boundidx,
                     bounditer=results.bounditer,
                     scale=batch_sampler.scale)
            self.new_run.append(D)

            # Increment relevant counters.
            self.eff = 100. * self.it / self.ncall
            self.it += 1
            # These yields will be just for printing
            yield IteratorResultShort(worst=results.worst,
                                      ustar=results.ustar,
                                      vstar=results.vstar,
                                      loglstar=results.loglstar,
                                      nc=results.nc,
                                      worst_it=results.worst_it + it0,
                                      boundidx=results.boundidx,
                                      bounditer=results.bounditer,
                                      eff=self.eff)
        del self.batch_sampler
        self.batch_sampler = None

    def combine_runs(self):
        """ Merge the most recent run into the previous (combined) run by
        "stepping through" both runs simultaneously."""

        # Make sure we have a run to add.
        if len(self.new_run['id']) == 0:
            raise ValueError("No new samples are currently saved.")

        # Grab results from saved run.
        saved_d = {}
        new_d = {}

        for k in [
                'id', 'u', 'v', 'logl', 'nc', 'boundidx', 'it', 'bounditer',
                'n', 'scale', 'blob', 'logvol'
        ]:
            saved_d[k] = np.array(self.saved_run[k])
            new_d[k] = np.array(self.new_run[k])

        saved_d['batch'] = np.array(self.saved_run['batch'])
        nsaved = len(saved_d['n'])

        new_d['id'] = new_d['id'] + max(saved_d['id']) + 1
        nnew = len(new_d['n'])
        llmin, llmax = self.new_logl_min, self.new_logl_max

        old_batch_bounds = self.saved_run['batch_bounds']
        old_batch_nlive = self.saved_run['batch_nlive']
        # Reset saved results.
        del self.saved_run
        self.saved_run = RunRecord(dynamic=True)

        # Start our counters at the beginning of each set of dead points.
        idx_saved, idx_new = 0, 0  # start of our dead points
        logl_s, logl_n = saved_d['logl'][idx_saved], new_d['logl'][idx_new]
        nlive_s, nlive_n = saved_d['n'][idx_saved], new_d['n'][idx_new]

        # Iteratively walk through both set of samples to simulate
        # a combined run.
        ntot = nsaved + nnew
        for _ in range(ntot):
            if logl_s > self.new_logl_min:
                # If our saved samples are past the lower log-likelihood
                # bound, both runs are now "active" and should be used.
                nlive = nlive_s + nlive_n
            else:
                # If instead our collection of dead points are below
                # the bound, just use our collection of saved samples.
                nlive = nlive_s
            add_info = {}

            # Increment our position along depending on
            # which dead point (saved or new) is worse.
            if logl_s <= logl_n:
                add_info['batch'] = saved_d['batch'][idx_saved]
                add_source = saved_d
                add_idx = int(idx_saved)
                idx_saved += 1
            else:
                add_info['batch'] = self.batch + 1
                add_source = new_d
                add_idx = int(idx_new)
                idx_new += 1

            for k in [
                    'id', 'u', 'v', 'logl', 'nc', 'boundidx', 'it',
                    'bounditer', 'scale', 'blob'
            ]:
                add_info[k] = add_source[k][add_idx]
            self.saved_run.append(add_info)
            self.saved_run['n'].append(nlive)

            # Attempt to step along our samples. If we're out of samples,
            # set values to defaults.
            try:
                logl_s = saved_d['logl'][idx_saved]
                nlive_s = saved_d['n'][idx_saved]
            except IndexError:
                logl_s = np.inf
                nlive_s = 0
            try:
                logl_n = new_d['logl'][idx_new]
                nlive_n = new_d['n'][idx_new]
            except IndexError:
                logl_n = np.inf
                nlive_n = 0

        plateau_mode = False
        plateau_counter = 0
        plateau_logdvol = 0
        logvol = self.sampler.logvol_init
        logl_array = np.array(self.saved_run['logl'])
        nlive_array = np.array(self.saved_run['n'])

        for i, (cur_logl, nlive) in enumerate(zip(logl_array, nlive_array)):
            # Save the number of live points and expected ln(volume).
            if (not plateau_mode and i != len(nlive_array) - 1
                    and logl_array[i] == logl_array[i + 1]):
                plateau_mask = (logl_array[i:] == cur_logl)
                nplateau = plateau_mask.sum()
                if nplateau > 1:
                    # the number of live points should not change throughout
                    # the plateau unless we are also merging it with the run
                    # where the plateau is explored through final points,
                    # i.e. when the number of live-points decreases.
                    plateau_counter = nplateau
                    plateau_logdvol = logvol + np.log(1. / (nlive + 1))
                    plateau_mode = True
            if not plateau_mode:
                logvol -= math.log((nlive + 1.) / nlive)
            else:
                logvol = logvol + np.log1p(-np.exp(plateau_logdvol - logvol))
            self.saved_run['logvol'].append(logvol)
            if plateau_mode:
                plateau_counter -= 1
                if plateau_counter == 0:
                    plateau_mode = False
        # ensure that we correctly merged

        assert self.saved_run['logl'][0] == min(new_d['logl'][0],
                                                saved_d['logl'][0])
        assert self.saved_run['logl'][-1] == max(new_d['logl'][-1],
                                                 saved_d['logl'][-1])

        new_logwt, new_logz, new_logzvar, new_h = compute_integrals(
            logl=self.saved_run['logl'], logvol=self.saved_run['logvol'])
        self.saved_run['logwt'].extend(new_logwt.tolist())
        self.saved_run['logz'].extend(new_logz.tolist())
        self.saved_run['logzvar'].extend(new_logzvar.tolist())
        self.saved_run['h'].extend(new_h.tolist())

        # Reset results.
        self.new_run = None
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

        # Increment batch counter.
        self.batch += 1

        # Saved batch quantities.
        self.saved_run['batch_nlive'] = old_batch_nlive + [(max(new_d['n']))]
        self.saved_run['batch_bounds'] = old_batch_bounds + [((llmin, llmax))]

    def run_nested(self,
                   nlive_init=None,
                   maxiter_init=None,
                   maxcall_init=None,
                   dlogz_init=0.01,
                   logl_max_init=np.inf,
                   n_effective_init=np.inf,
                   nlive_batch=None,
                   wt_function=None,
                   wt_kwargs=None,
                   maxiter_batch=None,
                   maxcall_batch=None,
                   maxiter=None,
                   maxcall=None,
                   maxbatch=None,
                   n_effective=None,
                   stop_function=None,
                   stop_kwargs=None,
                   use_stop=True,
                   save_bounds=True,
                   print_progress=True,
                   print_func=None,
                   live_points=None,
                   resume=False,
                   checkpoint_file=None,
                   checkpoint_every=60):
        """
        **The main dynamic nested sampling loop.** After an initial "baseline"
        run using a constant number of live points, dynamically allocates
        additional (nested) samples to optimize a specified weight function
        until a specified stopping criterion is reached.

        Parameters
        ----------
        nlive_init : int, optional
            The number of live points used during the initial ("baseline")
            nested sampling run. Default is the number provided at
            initialization

        maxiter_init : int, optional
            Maximum number of iterations for the initial baseline nested
            sampling run. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall_init : int, optional
            Maximum number of likelihood evaluations for the initial
            baseline nested sampling run. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        dlogz_init : float, optional
            The baseline run will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            `0.01`.

        logl_max_init : float, optional
            The baseline run will stop when the sampled ln(likelihood) exceeds
            this threshold. Default is no bound (`np.inf`).

        n_effective_init: int, optional
            Minimum number of effective posterior samples needed during the
            baseline run. If the estimated "effective sample size" (ESS)
            exceeds this number, sampling will terminate.
            Default is no ESS (`np.inf`).
            This option is deprecated and will be removed in a future release.

        nlive_batch : int, optional
            The number of live points used when adding additional samples
            from a nested sampling run within each batch. Default is the
            number provided at init

        wt_function : func, optional
            A cost function that takes a :class:`Results` instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = pfrac * pweight + (1. - pfrac) * zweight

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter_batch : int, optional
            Maximum number of iterations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall_batch : int, optional
            Maximum number of likelihood evaluations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxiter : int, optional
            Maximum number of iterations allowed. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is `sys.maxsize` (no limit).

        maxbatch : int, optional
            Maximum number of batches allowed. Default is `sys.maxsize`
            (no limit).

        n_effective: int, optional
            Minimum number of effective posterior samples needed during the
            entire run. If the estimated "effective sample size" (ESS)
            exceeds this number, sampling will terminate.
            Default is max(10000, ndim^2)

        stop_function : func, optional
            A function that takes a :class:`Results` instance and
            returns a boolean indicating that we should terminate the run
            because we've collected enough samples.

        stop_kwargs : float, optional
            Extra arguments to be passed to the stopping function.

        use_stop : bool, optional
            Whether to evaluate our stopping function after each batch.
            Disabling this can improve performance if other stopping criteria
            such as :data:`maxcall` are already specified. Default is `True`.

        save_bounds : bool, optional
            Whether or not to save distributions used to bound
            the live points internally during dynamic live point allocation.
            Default is `True`.

        print_progress : bool, optional
            Whether to output a simple summary of the current run that
            updates each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        live_points: list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            and optionally list of blobs associated with these likelihood calls
            (if blob=True in the sampler)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods. By default, if these are not provided the initial
            set of live points will be drawn from the unit `ndim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points will lead to incorrect results.**

        resume: bool, optional
            If resume is set to true, we will try to resume a previously
            interrupted run
        checkpoint_file: string, optional
            if not None The state of the sampler will be saved into this
            file every checkpoint_every seconds
        checkpoint_every: float, optional
            The number of seconds between checkpoints that will save
            the internal state of the sampler
        """

        # Check for deprecated options
        if n_effective_init is not np.inf:
            with warnings.catch_warnings():
                warnings.filterwarnings("once")
                warnings.warn(
                    "The n_effective_init option to DynamicSampler.run_nested "
                    "is deprecated and will be removed in future releases",
                    DeprecationWarning)

        # Initialize values.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if maxiter_batch is None:
            maxiter_batch = sys.maxsize
        if maxcall_batch is None:
            maxcall_batch = sys.maxsize
        if maxbatch is None:
            maxbatch = sys.maxsize
        if maxiter_init is None:
            maxiter_init = sys.maxsize
        if maxcall_init is None:
            maxcall_init = sys.maxsize
        if wt_function is None:
            wt_function = weight_function
        if wt_kwargs is None:
            wt_kwargs = {}
        if stop_function is None:
            default_stop_function = True
            stop_function = stopping_function
        else:
            default_stop_function = False
        if stop_kwargs is None:
            stop_kwargs = {}
        if default_stop_function:
            if n_effective is None:
                # The reason to scale with square of number of
                # dimensions is because the number coefficients
                # defining covariance is roughly 0.5 * N^2
                n_effective = max(self.ndim * self.ndim, 10000)

            stop_kwargs['target_n_effective'] = n_effective
        nlive_init = nlive_init or self.nlive0
        nlive_batch = nlive_batch or self.nlive0

        # Run the main dynamic nested sampling loop.
        ncall = self.ncall

        niter = self.it - 1
        logl_bounds = (-np.inf, np.inf)
        maxcall_init = min(maxcall_init, maxcall)  # set max calls
        maxiter_init = min(maxiter_init, maxiter)  # set max iterations

        if resume and self.internal_state == DynamicSamplerStatesEnum.RUN_DONE:
            warnings.warn(
                """You tried to resume the run that has ended successfully.
This is not supported. No sampling was performed""", RuntimeWarning)
            return
        # Baseline run.
        pbar, print_func = get_print_func(print_func, print_progress)
        self.checkpoint_timer = DelayTimer(checkpoint_every)
        try:
            if not self.base:
                for results in self.sample_initial(
                        nlive=nlive_init,
                        dlogz=dlogz_init,
                        maxcall=maxcall_init,
                        maxiter=maxiter_init,
                        logl_max=logl_max_init,
                        live_points=live_points,
                        n_effective=n_effective_init,
                        resume=resume,
                        save_samples=True):
                    if resume:
                        resume = False
                    ncall += results.nc
                    niter += 1
                    if (checkpoint_file is not None and self.internal_state
                            != DynamicSamplerStatesEnum.INBASEADDLIVE
                            and self.checkpoint_timer.is_time()):
                        self.save(checkpoint_file)
                    # Print progress.
                    if print_progress:
                        print_func(results,
                                   niter,
                                   ncall,
                                   nbatch=0,
                                   dlogz=dlogz_init,
                                   logl_max=logl_max_init)
            for n in range(self.batch, maxbatch):
                # Update stopping criteria.
                res = self.results
                mcall = min(maxcall - ncall, maxcall_batch)
                miter = min(maxiter - niter, maxiter_batch)
                if mcall > 0 and miter > 0 and use_stop:
                    if self.use_pool_stopfn:
                        M = self.M
                    else:
                        M = map
                    stop, stop_vals = stop_function(res,
                                                    stop_kwargs,
                                                    rstate=self.rstate,
                                                    M=M,
                                                    return_vals=True)
                    stop_val = stop_vals[2]
                else:
                    stop = False
                    stop_val = np.nan

                # If we have likelihood calls remaining, iterations remaining,
                # and we have failed to hit the minimum ESS, run our batch.
                if mcall > 0 and miter > 0 and not stop:
                    # Compute our sampling bounds using the provided
                    # weight function.
                    passback = self.add_batch(nlive=nlive_batch,
                                              wt_function=wt_function,
                                              wt_kwargs=wt_kwargs,
                                              maxiter=miter,
                                              maxcall=mcall,
                                              save_bounds=save_bounds,
                                              print_progress=print_progress,
                                              print_func=print_func,
                                              stop_val=stop_val,
                                              resume=resume,
                                              checkpoint_file=checkpoint_file)
                    if resume:
                        # The assumption here is after the first resume
                        # iteration we will proceed as normal
                        resume = False
                    ncall, niter, logl_bounds, results = passback
                elif logl_bounds[1] != np.inf:
                    # We ran at least one batch and now we're done!
                    if print_progress:
                        print_func(results,
                                   niter,
                                   ncall,
                                   nbatch=n,
                                   stop_val=stop_val,
                                   logl_min=logl_bounds[0],
                                   logl_max=logl_bounds[1])
                    break
                else:
                    # We didn't run a single batch but now we're done!
                    break
            self.internal_state = DynamicSamplerStatesEnum.RUN_DONE
            if checkpoint_file is not None:
                # In the very end I save the checkpoint no matter
                # the timing
                self.save(checkpoint_file)
        finally:
            if pbar is not None:
                pbar.close()
            self.loglikelihood.history_save()

    def add_batch(self,
                  nlive=500,
                  dlogz=1e-2,
                  mode='weight',
                  wt_function=None,
                  wt_kwargs=None,
                  maxiter=None,
                  maxcall=None,
                  logl_bounds=None,
                  save_bounds=True,
                  print_progress=True,
                  print_func=None,
                  stop_val=None,
                  resume=False,
                  checkpoint_file=None,
                  checkpoint_every=None):
        """
        Allocate an additional batch of (nested) samples based on
        the combined set of previous samples using the specified
        weight function.

        Parameters
        ----------
        nlive : int, optional
            The number of live points used when adding additional samples
            in the batch. Default is `500`.

        mode: string, optional
            How to allocate a new batch.
            The possible values are 'auto', 'weight', 'full', 'manual'
            'weight' means to use the weight_function to decide the optimal
            logl range.
            'full' means sample the whole posterior again
            'auto' means choose automatically, which currently means using
            'weight'
            'manual' means that logl_bounds need to be explicitely specified

        wt_function : func, optional
            A cost function that takes a `Results` instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = pfrac * pweight + (1. - pfrac) * zweight

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter : int, optional
            Maximum number of iterations allowed. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is `sys.maxsize` (no limit).

        logl_bounds : tuple of size (2,), optional
            The ln(likelihood) bounds used to bracket the run. If `None`,
            the provided `wt_function` will be used to determine the bounds
            (this is the default behavior).

        save_bounds : bool, optional
            Whether or not to save distributions used to bound
            the live points internally during dynamic live point allocations.
            Default is `True`.

        print_progress : bool, optional
            Whether to output a simple summary of the current run that
            updates each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        stop_val : float, optional
            The value of the stopping criteria to be passed to
            :meth:`print_func`. Used internally within :meth:`run_nested` to
            keep track of progress.
        resume: bool, optional
            If resume is set to true, we will try to resume a previously
            interrupted run
        checkpoint_file: string, optional
            if not None The state of the sampler will be saved into this
            file every checkpoint_every seconds
        checkpoint_every: float, optional
            The number of seconds between checkpoints that will save
            the internal state of the sampler. If this is None, we
            we will use the timer created in run_nested()
        """

        # Initialize values.
        maxcall = maxcall or sys.maxsize
        maxiter = maxiter or sys.maxsize
        wt_function = wt_function or weight_function
        wt_kwargs = wt_kwargs or {}
        stop_val = stop_val or np.nan

        res = self.results

        if mode != 'manual' and logl_bounds is not None:
            raise RuntimeError(
                "specified logl_bounds are only allowed for manual mode")
        if mode == 'manual' and logl_bounds is None:
            raise RuntimeError(
                "logl_bounds need to be specified for manual mode")
        if mode == 'auto' or mode == 'weight':
            logl_bounds = wt_function(res, wt_kwargs)
        # this is just for printing
        if logl_bounds is None:
            logl_min, logl_max = -np.inf, np.inf
        else:
            logl_min, logl_max = logl_bounds
        # For printing as well, we just display old logz,logzerr here
        logz, logzvar = res['logz'][-1], res['logzerr'][-1]**2

        # If we have either likelihood calls or iterations remaining,
        # add our new batch of live points.
        ncall, niter, n = self.ncall, self.it - 1, self.batch
        if checkpoint_file is not None:
            # if checkpoint_every is provided we are assuming we are
            # running externally otherwise we are being run from run_nested
            # and in that case we use a global timer
            # We have to care about this because if our batches take
            # shorter than checkpoint_every we still would like to save
            if checkpoint_every is not None:
                timer = DelayTimer(checkpoint_every)
            else:
                timer = self.checkpoint_timer
        if maxcall > 0 and maxiter > 0:
            pbar, print_func = get_print_func(print_func, print_progress)
            try:
                results = None  # to silence pylint as
                # sample_batch() should return something given maxiter/maxcall
                for cur_results in self.sample_batch(nlive_new=nlive,
                                                     dlogz=dlogz,
                                                     logl_bounds=logl_bounds,
                                                     maxiter=maxiter,
                                                     maxcall=maxcall,
                                                     save_bounds=save_bounds,
                                                     resume=resume):
                    if resume:
                        # only one resume iteration, after that
                        # we switch to normal, although currently
                        # resume is not used anywhere after
                        resume = False
                    if cur_results.worst >= 0:
                        ncall += cur_results.nc
                        niter += 1

                    # Reorganize results.
                    results = IteratorResult(worst=cur_results.worst,
                                             ustar=cur_results.ustar,
                                             vstar=cur_results.vstar,
                                             loglstar=cur_results.loglstar,
                                             blob=None,
                                             logvol=np.nan,
                                             logwt=np.nan,
                                             logz=logz,
                                             logzvar=logzvar,
                                             h=np.nan,
                                             nc=cur_results.nc,
                                             worst_it=cur_results.worst_it,
                                             boundidx=cur_results.boundidx,
                                             bounditer=cur_results.bounditer,
                                             eff=cur_results.eff,
                                             delta_logz=np.nan)

                    # Print progress.
                    if print_progress:
                        print_func(results,
                                   niter,
                                   ncall,
                                   nbatch=n + 1,
                                   stop_val=stop_val,
                                   logl_min=logl_min,
                                   logl_max=logl_max)
                    if (checkpoint_file is not None and self.internal_state
                            != DynamicSamplerStatesEnum.INBATCHADDLIVE
                            and self.internal_state
                            != DynamicSamplerStatesEnum.BATCH_DONE
                            and timer.is_time()):
                        # we do not save the state if we are finishing the
                        # batch run and we are just adding live-points in
                        # the end
                        self.save(checkpoint_file)
            finally:
                if pbar is not None:
                    pbar.close()
                self.loglikelihood.history_save()

            # Combine batch with previous runs.
            self.combine_runs()
            # Pass back info.
            self.internal_state = DynamicSamplerStatesEnum.BATCH_DONE
            return ncall, niter, logl_bounds, results
        else:
            raise RuntimeError(
                'add_batch called with no leftover function calls'
                'or iterations')
