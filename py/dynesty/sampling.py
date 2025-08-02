#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for proposing new live points used by
:class:`~dynesty.sampler.Sampler`
:class:`~dynesty.dynamicsampler.DynamicSampler`.

"""

from collections import namedtuple
import warnings
import numpy as np
from numpy import linalg
import math
from .utils import unitcheck, apply_reflect, get_random_generator, SamplerHistoryItem
from .bounding import randsphere

__all__ = [
    'InternalSampler', 'RSliceSampler', 'SliceSampler', 'RWalkSampler',
    'UniformBoundSampler', 'UnitCubeSampler'
]
SamplerArgument = namedtuple('SamplerArgument', [
    'u', 'loglstar', 'axes', 'scale', 'prior_transform', 'loglikelihood',
    'rseed', 'kwargs'
])


SamplerReturn = namedtuple('SamplerReturn', [
    'u', 'v', 'logl', 'ncalls', 'sampling_history', 'tuning_info',
    'proposal_stats'
])

INTERNAL_SAMPLER_LIST = ['rwalk', 'unif', 'rslice', 'slice']


class InternalSampler:
    """Base class for all internal samplers.

    This class is not meant to be used directly.
    The basic interface of the class is to provide sampling that can
    be distributed over workers.

    The key methods are `prepare_sampler`, `sample` and `tune`.

    The `prepare_sampler` method constructs the list of SamplerArguments
    objects.

    The `sample` method is called by the workers to sample a new point.
    Importantly the `sample` method is a static method and does not have
    access to the class instance.

    The `tune` method is called by the parent process to adjust things,
    such as scale of the proposal distribution. The `tune` method is not
    a static method and has access to the class instance.
    """

    def __init__(self, **kwargs):
        """Initialize the internal sampler.

        Importantely this sets up the sampler_kwargs that is being passed
        to each .sample() call

        Parameters
        ----------
        kwargs : dict
            A dictionary of additional method-specific parameters.
            This common keywords:
        nonbounded : array
            Array of boolean values indicating which dimensions are
            non-bounded.
        periodic : array
            Array of boolean values indicating which dimensions are
            periodic.
        reflective : array
            Array of boolean values indicating which dimensions are
            reflective.
        ndim: int
            Number of dimensions.
        """
        self.scale = 1
        self.input_kwargs = kwargs
        self.sampler_kwargs = dict()
        self.ndim = kwargs.get('ndim')
        for k in ['nonbounded', 'periodic', 'reflective']:
            self.sampler_kwargs[k] = kwargs.get(k)

    @property
    def update_bound_interval_ratio(self):
        """ How often to force updating the bounds
        The value is in units of ncall per nlive.
        I.e. the value of 10 means for N live points,
        the bound will be updated every 10 * N calls
        """
        return 1  # default value

    def _new_from_template(self, template_kwargs):
        """
        Create a new sampler from a template.
        """
        template_kwargs1 = self.input_kwargs.copy()
        for k, v in template_kwargs.items():
            if k not in self.input_kwargs:
                template_kwargs1[k] = v
            else:
                if template_kwargs1[k] != self.input_kwargs[k]:
                    warnings.warn(
                        "Incompatible sampler parameters: "
                        f"{template_kwargs1[k]} vs {self.input_kwargs[k]}")
        return self.__class__(**template_kwargs1)

    def prepare_sampler(self,
                        loglstar=None,
                        points=None,
                        axes=None,
                        seeds=None,
                        prior_transform=None,
                        loglikelihood=None,
                        nested_sampler=None):
        """
        Prepare the list of arguments for sampling.

        Parameters
        ----------
        loglstar : float
            Ln(likelihood) bound.
        points : `~numpy.ndarray` with shape (n, ndim)
            Initial sample points.
        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points.
        seeds : `~numpy.ndarray` with shape (n,)
            Random number generator seeds.
        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.
        nested_sampler : `~dynesty.samplers.Sampler`
            The nested sampler object used to sample.

        Returns
        -------
        arglist:
            List of `SamplerArgument` objects containing the parameters
            needed for sampling.
        """
        arg_list = []
        kwargs = self.sampler_kwargs
        for curp, curax, curseed in zip(points, axes, seeds):
            curarg = SamplerArgument(u=curp,
                                     loglstar=loglstar,
                                     axes=curax,
                                     scale=self.scale,
                                     prior_transform=prior_transform,
                                     loglikelihood=loglikelihood,
                                     rseed=curseed,
                                     kwargs=kwargs)
            arg_list.append(curarg)
        return arg_list

    @staticmethod
    def sample(args):
        """
        Sample a new live point.

        Parameters
        ----------
        args : `SamplerArgument`
            The arguments needed for sampling.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.
        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.
        logl : float
            Ln(likelihood) of the final proposed point.
        nc : int
            Number of function calls used to generate the sample.
        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.
        """
        pass

    def tune(self, tuning_info, update=False):
        """

        Accumulate sampling info and optionally update the proposal scale and
        other tuning parameters.

        Parameters
        ----------
        tuning_info : dict
            Dictionary containing the sampling information.
        update : bool
            Whether to update the proposal scale or not (default: False).
        """
        pass

    @property
    def citations(self):
        return []


class UniformBoundSampler(InternalSampler):
    """
    Uniformly sample within a bounding proposal distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_sampler(self,
                        loglstar=None,
                        points=None,
                        axes=None,
                        seeds=None,
                        prior_transform=None,
                        loglikelihood=None,
                        nested_sampler=None):
        """
        Prepare the list of arguments for sampling.

        This is is overriding the base class method and providing the bound
        itself to the sampler through kwargs.

        """
        self.sampler_kwargs['bound'] = nested_sampler.bound
        self.sampler_kwargs['ndim'] = nested_sampler.ndim
        self.sampler_kwargs['n_cluster'] = nested_sampler.ncdim
        if nested_sampler.bound.need_centers:
            self.sampler_kwargs['bound'].ctrs = nested_sampler.live_u

        return super().prepare_sampler(loglstar=loglstar,
                                       points=points,
                                       axes=axes,
                                       seeds=seeds,
                                       prior_transform=prior_transform,
                                       loglikelihood=loglikelihood,
                                       nested_sampler=nested_sampler)

    @staticmethod
    def sample(args):
        """
        Return a new live point sampling uniformly within the
        boundary.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Initial sample (not used)

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. (not used)

        scale : float
            Value used to scale the provided axes. (not used)

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.
            This method requires keywords:
            bound (dynesty.bounding object)
            ndim (number of dimensions)
            n_cluster (number of dimensions that are clustered)
            nonbounded array

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """

        # Unzipping.
        rstate = get_random_generator(args.rseed)
        bound = args.kwargs['bound']
        nonbounded = args.kwargs.get('nonbounded')
        n_cluster = args.kwargs.get('n_cluster')
        ndim = args.kwargs['ndim']
        nc = 0
        tuning_info = None
        if nonbounded is not None:
            nonbounded = nonbounded[:n_cluster]
        ntries = 0
        threshold_warning = 10000
        threshold_warned = False
        sampling_history = []
        while True:
            u = bound.samples(1, rstate=rstate).flatten()
            if not unitcheck(u, nonbounded):
                ntries += 1
                if ntries > threshold_warning and not threshold_warned:
                    warnings.warn(
                        "Ellipsoid sampling is extremely inefficient",
                        category=RuntimeWarning)
                    threshold_warned = True

                continue
            else:
                ntries = 0
            if n_cluster != ndim:
                u = np.concatenate(
                    (u, rstate.uniform(size=(ndim - n_cluster))))
            v = args.prior_transform(np.asarray(u))
            logl = args.loglikelihood(np.asarray(v))
            sampling_history.append(SamplerHistoryItem(u=u, v=v, logl=logl))
            nc += 1
            if logl > args.loglstar:
                break
        return SamplerReturn(u=u,
                             v=v,
                             logl=logl,
                             ncalls=nc,
                             tuning_info=tuning_info,
                             sampling_history=sampling_history,
                             proposal_stats={'n_proposals': ntries})


class UnitCubeSampler(InternalSampler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ndim = kwargs['ndim']

    def prepare_sampler(self,
                        loglstar=None,
                        points=None,
                        axes=None,
                        seeds=None,
                        prior_transform=None,
                        loglikelihood=None,
                        nested_sampler=None):
        self.sampler_kwargs['ndim'] = self.ndim
        return super().prepare_sampler(loglstar=loglstar,
                                       points=points,
                                       axes=axes,
                                       seeds=seeds,
                                       prior_transform=prior_transform,
                                       loglikelihood=loglikelihood,
                                       nested_sampler=nested_sampler)

    @staticmethod
    def sample(args):
        """
        Return a new live point sampling uniformly within the
        boundary.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Initial sample (not used)

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. (not used)

        scale : float
            Value used to scale the provided axes. (not used)

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.
            This method requires keywords:
            bound (dynesty.bounding object)
            ndim (number of dimensions)
            n_cluster (number of dimensions that are clustered)
            nonbounded array

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """

        # Unzipping.
        rstate = get_random_generator(args.rseed)
        ndim = args.kwargs.get('ndim')
        nc = 0
        tuning_info = None
        sampling_history = []
        while True:
            u = rstate.uniform(size=ndim)
            v = args.prior_transform(np.asarray(u))
            logl = args.loglikelihood(np.asarray(v))
            sampling_history.append(SamplerHistoryItem(u=u, v=v, logl=logl))
            nc += 1
            if logl > args.loglstar:
                break
        return SamplerReturn(u=u,
                             v=v,
                             logl=logl,
                             ncalls=nc,
                             tuning_info=tuning_info,
                             sampling_history=sampling_history,
                             proposal_stats=dict(n_proposals=nc))


class RWalkSampler(InternalSampler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize random walk parameters.
        walks = max(2, kwargs.get('walks', 25))
        self.facc = kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / walks, self.facc))
        self.rwalk_history = {'n_accept': 0, 'n_reject': 0}
        self.ncdim = kwargs.get('ncdim')
        # Since the sample is a static method, it's crucial
        # to put relevant information into kwargs which is then passed to
        # the samplers
        self.sampler_kwargs['walks'] = walks
        self.sampler_kwargs['ncdim'] = self.ncdim

    def tune(self, tuning_info, update=True):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps.
        For rwalk the scale is important because it
        determines the speed of diffusion of points.
        I.e. if scale is too large, the proposal efficiency will be very low
        so it's likely that we'll only do one random walk step at the time,
        thus producing very correlated chain.
        The keyword update determines if we are just accumulating the number
        of steps or actually adjusting the scale
        """
        self.scale = tuning_info['scale']
        hist = self.rwalk_history
        hist['n_accept'] += tuning_info['accept']
        hist['n_reject'] += tuning_info['reject']
        if not update:
            return
        accept, reject = hist['n_accept'], hist['n_reject']
        facc = (1. * accept) / (accept + reject)
        # Here we are now trying to solve the Eqn
        # f0 = F(s) where F is the function
        # providing the acceptance rate given logscale
        # and f0 is our target acceptance rate
        # in this case a Newton like update to s
        # is s_{k+1} = s_k - 1/F'(s_k) * (F_k - F_0)
        # We can speculate that F(s)~ C*exp(-Ns)
        # i.e. it's inversely proportional to volume
        # Then F'(s) = -N * F \approx N * F_0
        # Therefore s_{k+1} = s_k + 1/(N*F_0) * (F_k-F0)
        # See also Robbins-Munro recursion which we don't follow
        # here because our coefficients a_k do not obey \sum a_k^2 = \infty
        self.scale *= math.exp((facc - self.facc) / self.ncdim / self.facc)
        hist['n_accept'] = 0
        hist['n_reject'] = 0

    @property
    def update_bound_interval_ratio(self):
        """ How often to force updating the bounds
        The value is in units of ncall per nlive.
        I.e. the value of 10 means for N live points,
        the bound will be updated every 10 * N calls
        """
        return self.sampler_kwargs['walks']  # default value

    @staticmethod
    def sample(args):
        """
        Return a new live point proposed by random walking away from an
        existing live point.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the initial sample. **This is a copy of an existing
            live point.**

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. For random walks new positions are
            proposed using the :class:`~dynesty.bounding.Ellipsoid` whose
            shape is defined by axes.

        scale : float
            Value used to scale the provided axes.

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """

        # Unzipping.
        rstate = get_random_generator(args.rseed)
        return generic_random_walk(args.u, args.loglstar, args.axes,
                                   args.scale, args.prior_transform,
                                   args.loglikelihood, rstate, args.kwargs)

    @property
    def citations(self):
        return [("Skilling (2006)", "projecteuclid.org/euclid.ba/1340370944")]


class SliceSampler(InternalSampler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize slice parameters.
        slices = kwargs.get('slices', 5)
        self.slice_history = {'n_contract': 0, 'n_expand': 0}

        self.sampler_kwargs['slices'] = slices
        # Since the sample is a static method, it's crucial
        # to put relevant information into kwargs which is then passed to
        # the samplers

    @property
    def update_bound_interval_ratio(self):
        """ How often to force updating the bounds
        The value is in units of ncall per nlive.
        I.e. the value of 10 means for N live points,
        the bound will be updated every 10 * N calls
        """
        return self.sampler_kwargs['slices'] * self.ndim

    def tune(self, sampler_info, update=True):
        tune_slice(self, sampler_info, update=update)

    @staticmethod
    def sample(args):
        """
        Return a new live point proposed by a series of random slices
        away from an existing live point. Standard "Gibbs-like" implementation
        where a single multivariate "slice" is a combination of `ndim`
        univariate slices
        through each axis.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the initial sample. **This is a copy of an existing
            live point.**

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. For slices new positions are
            proposed along the arthogonal basis defined by :data:`axes`.

        scale : float
            Value used to scale the provided axes.

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """

        # Unzipping.
        (u, loglstar, axes, scale, prior_transform, loglikelihood,
         kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                    args.prior_transform, args.loglikelihood, args.kwargs)
        rstate = get_random_generator(args.rseed)
        # Periodicity.
        nonperiodic = kwargs.get('nonperiodic', None)
        doubling = kwargs.get('slice_doubling', False)
        # Setup.
        n = len(u)
        assert axes.shape[0] == n
        slices = kwargs['slices']  # number of slices
        nc = 0
        n_expand = 0
        n_contract = 0
        sampling_history = []
        # Modifying axes and computing lengths.
        axes = scale * axes.T  # scale based on past tuning
        # Note we are transposing as axes[:,i] corresponds to i-th principal
        # axis of the ellipsoid
        expansion_warning_set = False
        # Slice sampling loop.
        for _ in range(slices):

            # Shuffle axis update order.
            idxs = np.arange(n)
            rstate.shuffle(idxs)

            # Slice sample along a random direction.
            for idx in idxs:

                # Select axis.
                axis = axes[idx]
                (u_prop, v_prop, logl_prop, nc1, n_expand1, n_contract1,
                 expansion_warning) = generic_slice_step(
                     u, axis, nonperiodic, loglstar, loglikelihood,
                     prior_transform, doubling, sampling_history, rstate)
                u = u_prop
                nc += nc1
                n_expand += n_expand1
                n_contract += n_contract1
                if expansion_warning and not doubling:
                    # if we expanded the interval by more than
                    # the threshold we set the warning and enable doubling
                    expansion_warning_set = True
                    doubling = True
                    warnings.warn('Enabling doubling strategy of slice '
                                  'sampling from Neal(2003)')
        tuning_info = {
            'n_expand': n_expand,
            'n_contract': n_contract,
            'expansion_warning_set': expansion_warning_set
        }

        return SamplerReturn(u=u_prop,
                             v=v_prop,
                             logl=logl_prop,
                             ncalls=nc,
                             tuning_info=tuning_info,
                             sampling_history=sampling_history,
                             proposal_stats=dict(n_expand=n_expand,
                                                 n_contract=n_contract))

    @property
    def citations(self):
        return [("Neal (2003)", "projecteuclid.org/euclid.aos/1056562461"),
                ("Handley, Hobson & Lasenby (2015a)",
                 "ui.adsabs.harvard.edu/abs/2015MNRAS.450L..61H"),
                ("Handley, Hobson & Lasenby (2015b)",
                 "ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H")]


class RSliceSampler(InternalSampler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize slice parameters.
        slices = kwargs.get('slices', 5)
        self.slice_history = {'n_contract': 0, 'n_expand': 0}

        self.sampler_kwargs['slices'] = slices
        # Since the sample is a static method, it's crucial
        # to put relevant information into kwargs which is then passed to
        # the samplers

    def tune(self, sampler_info, update=True):
        tune_slice(self, sampler_info, update=update)

    @property
    def update_bound_interval_ratio(self):
        """ How often to force updating the bounds
        The value is in units of ncall per nlive.
        I.e. the value of 10 means for N live points,
        the bound will be updated every 10 * N calls
        """
        return self.sampler_kwargs['slices']

    @staticmethod
    def sample(args):
        """
        Return a new live point proposed by a series of random slices
        away from an existing live point. Standard "random" implementation
        where
        each slice is along a random direction based on the provided axes.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the initial sample. **This is a copy of an existing
            live point.**

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new slice directions.

        scale : float
            Value used to scale the provided axes.

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        tuning_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """

        # Unzipping.
        (u, loglstar, axes, scale, prior_transform, loglikelihood,
         kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                    args.prior_transform, args.loglikelihood, args.kwargs)
        rstate = get_random_generator(args.rseed)
        # Periodicity.
        nonperiodic = kwargs.get('nonperiodic', None)
        doubling = kwargs.get('slice_doubling', False)
        sampling_history = []
        # Setup.
        n = len(u)
        assert axes.shape[0] == n
        slices = kwargs['slices']  # number of slices
        nc = 0
        n_expand = 0
        n_contract = 0
        expansion_warning_set = False

        # Slice sampling loop.
        for _ in range(slices):

            # Propose a direction on the unit n-sphere.
            drhat = rstate.standard_normal(size=n)
            drhat /= linalg.norm(drhat)

            # Transform and scale based on past tuning.
            direction = np.dot(axes, drhat) * scale

            (u_prop, v_prop, logl_prop, nc1, n_expand1, n_contract1,
             expansion_warning) = generic_slice_step(u, direction, nonperiodic,
                                                     loglstar, loglikelihood,
                                                     prior_transform, doubling,
                                                     sampling_history, rstate)
            u = u_prop
            nc += nc1
            n_expand += n_expand1
            n_contract += n_contract1
            if expansion_warning and not doubling:
                doubling = True
                expansion_warning_set = True
                warnings.warn('Enabling doubling strategy of slice '
                              'sampling from Neal(2003)')

        tuning_info = {
            'n_expand': n_expand,
            'n_contract': n_contract,
            'expansion_warning_set': expansion_warning_set
        }

        return SamplerReturn(u=u_prop,
                             v=v_prop,
                             logl=logl_prop,
                             ncalls=nc,
                             tuning_info=tuning_info,
                             sampling_history=sampling_history,
                             proposal_stats=dict(n_expand=n_expand,
                                                 n_contract=n_contract))

    @property
    def citations(self):
        return [("Neal (2003)", "projecteuclid.org/euclid.aos/1056562461"),
                ("Handley, Hobson & Lasenby (2015a)",
                 "ui.adsabs.harvard.edu/abs/2015MNRAS.450L..61H"),
                ("Handley, Hobson & Lasenby (2015b)",
                 "ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H")]


def generic_random_walk(u, loglstar, axes, scale, prior_transform,
                        loglikelihood, rstate, kwargs):
    """
    Generic random walk step

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
        Position of the initial sample.
        **This is a copy of an existing live point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. For random walks new positions are
        proposed using the :class:`~dynesty.bounding.Ellipsoid` whose
        shape is defined by axes.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    tuning_info : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Periodicity.
    nonbounded = kwargs.get('nonbounded')
    periodic = kwargs.get('periodic')
    reflective = kwargs.get('reflective')

    # Setup.
    n = len(u)
    n_cluster = axes.shape[0]
    walks = kwargs['walks']  # number of steps
    sampling_history = []
    n_accept = 0
    # Total number of accepted points with L>L*

    n_reject = 0
    # Total number of points proposed to within the ellipsoid and cube
    # but rejected due to L<=L* condition

    ncall = 0
    # Total number of Likelihood calls (proposals evaluated)

    # Here we loop for exactly walks iterations.
    while ncall < walks:

        # This proposes a new point within the ellipsoid
        u_prop, fail = propose_ball_point(u,
                                          scale,
                                          axes,
                                          n,
                                          n_cluster,
                                          rstate=rstate,
                                          periodic=periodic,
                                          reflective=reflective,
                                          nonbounded=nonbounded)
        if fail:
            n_reject += 1
            ncall += 1
            continue

        # Check proposed point.
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        ncall += 1
        sampling_history.append(
            SamplerHistoryItem(u=u_prop, v=v_prop, logl=logl_prop))

        if logl_prop > loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            n_accept += 1
        else:
            n_reject += 1
    if n_accept == 0:
        # Technically we can find out the likelihood value
        # stored somewhere
        # But I'm currently recomputing it
        v = prior_transform(u)
        logl = loglikelihood(v)

    tuning_info = {'accept': n_accept, 'reject': n_reject, 'scale': scale}

    return SamplerReturn(u=u,
                         v=v,
                         logl=logl,
                         ncalls=ncall,
                         tuning_info=tuning_info,
                         sampling_history=sampling_history,
                         proposal_stats=dict(n_accept=n_accept,
                                             n_reject=n_reject))


def propose_ball_point(u,
                       scale,
                       axes,
                       n,
                       n_cluster,
                       rstate=None,
                       periodic=None,
                       reflective=None,
                       nonbounded=None):
    """
    Here we are proposing points uniformly within an n-d ellipsoid.
    We are only trying once.
    We return the tuple with
    1) proposed point or None
    2) failure flag (if True, the generated point was outside bounds)
    """

    # starting point for clustered dimensions
    u_cluster = u[:n_cluster]

    # draw random point for non clustering parameters
    # we only need to generate them once
    u_non_cluster = rstate.random(n - n_cluster)
    u_prop = np.zeros(n)
    u_prop[n_cluster:] = u_non_cluster

    # Propose a direction on the unit n-sphere.
    dr = randsphere(n_cluster, rstate=rstate)
    # This generates uniform distribution within n-d ball

    # Transform to proposal distribution.
    du = np.dot(axes, dr)
    u_prop[:n_cluster] = u_cluster + scale * du

    # Wrap periodic parameters
    if periodic is not None:
        u_prop[periodic] = np.mod(u_prop[periodic], 1)

    # Reflect
    if reflective is not None:
        u_prop[reflective] = apply_reflect(u_prop[reflective])

    # Check unit cube constraints.
    if unitcheck(u_prop, nonbounded):
        return u_prop, False
    else:
        return None, True


def _slice_doubling_accept(x1, F, loglstar, L, R, fL, fR):
    """
    Acceptance test of slice sampling when doubling mode is used.
    This is an exact implementation of algorithm 6 of Neal 2003
    here w=1 and x0=0 as we are working in the
    coordinate system of F(A) = f(x0+A*w)

    Arguments are
    1) candidate location x1
    2) wrapped logl function (see generic_slice_step)
    3) threshold logl value
    4) left edge of the full interval
    5) right edge of the full interval
    6) value at left edge
    7) value at right edge
    """
    lhat, rhat = L, R
    f_lhat = fL
    f_rhat = fR
    D = False
    while rhat - lhat > 1.1:
        # Define slice and window.
        M = (lhat + rhat) / 2.
        # Propose new position.
        if (0 < M <= x1) or (x1 < M <= 0):
            D = True
        if x1 < M:
            rhat = M
            f_rhat = F(rhat)[1]
        else:
            lhat = M
            f_lhat = F(lhat)[1]
        if D and loglstar >= f_lhat and loglstar >= f_rhat:
            return False
    return True


def generic_slice_step(u, direction, nonperiodic, loglstar, loglikelihood,
                       prior_transform, doubling, sampling_history, rstate):
    """
    Do a slice generic slice sampling step along a specified dimension

    Parameters
    ----------
    u: ndarray (ndim sized)
        Starting point in unit cube coordinates
        It MUST satisfy the logl>loglstar criterion
    direction: ndarray (ndim sized)
        Step direction vector
    nonperiodic: ndarray(bool)
        mask for nonperiodic variables
    loglstar: float
        the critical value of logl, so that new logl must be >loglstar
    loglikelihood: function
    prior_transform: function
    rstate: random state

    """
    nc, n_expand, n_contract = 0, 0, 0
    n_expand_threshold = 1000  # Threshold for warning the user
    n = len(u)
    rand0 = rstate.random()  # initial scale/offset
    dirlen = linalg.norm(direction)
    maxlen = np.sqrt(n) / 2.
    # maximum initial interval length (the diagonal of the cube)
    if dirlen > maxlen:
        # I stopped giving warnings, as it was too noisy
        dirnorm = dirlen / maxlen
    else:
        dirnorm = 1
    direction = direction / dirnorm

    #  The function that evaluates the logl at the location of
    # u0 + x*direction0
    def F(x):
        nonlocal nc
        u_new = u + x * direction
        if unitcheck(u_new, nonperiodic):
            v_new = prior_transform(u_new)
            logl = loglikelihood(v_new)
            sampling_history.append(
                SamplerHistoryItem(u=u_new, v=v_new, logl=logl))
        else:
            logl = -np.inf
        nc += 1
        return u_new, logl

    # asymmetric step size on the left/right (see Neal 2003)
    nstep_l = -rand0
    nstep_r = (1 - rand0)

    logl_l = F(nstep_l)[1]
    logl_r = F(nstep_r)[1]
    expansion_warning = False
    if not doubling:
        # "Stepping out" the left and right bounds.
        while logl_l > loglstar:
            nstep_l -= 1
            logl_l = F(nstep_l)[1]
            n_expand += 1
        while logl_r > loglstar:
            nstep_r += 1
            logl_r = F(nstep_r)[1]
            n_expand += 1
        if n_expand > n_expand_threshold:
            expansion_warning = True
            warnings.warn('The slice sample interval was expanded more '
                          f'than {n_expand_threshold} times')

    else:
        # "Stepping out" the left and right bounds.
        K = 1
        while (logl_l > loglstar or logl_r > loglstar):
            V = rstate.random()
            if V < 0.5:
                nstep_l -= (nstep_r - nstep_l)
                logl_l = F(nstep_l)[1]
            else:
                nstep_r += (nstep_r - nstep_l)
                logl_r = F(nstep_r)[1]
            n_expand += K
            K *= 2
        L = nstep_l
        R = nstep_r
        fL = logl_l
        fR = logl_r

    # Sample within limits. If the sample is not valid, shrink
    # the limits until we hit the `loglstar` bound.

    while True:
        # Define slice and window.
        nstep_hat = nstep_r - nstep_l

        # Propose new position.
        nstep_prop = nstep_l + rstate.random() * nstep_hat  # scale from left
        u_prop, logl_prop = F(nstep_prop)
        n_contract += 1

        # If we succeed, move to the new position.
        # note that if we are using doubling mode we accept only
        # if _slice_doubling_accept() returns True
        if logl_prop > loglstar and (not doubling or _slice_doubling_accept(
                nstep_prop, F, loglstar, L, R, fL, fR)):
            break
        # If we fail, check if the new point is to the left/right of
        # our original point along our proposal axis and update
        # the bounds accordingly.
        else:
            if nstep_prop < 0:
                nstep_l = nstep_prop
            elif nstep_prop > 0:  # right
                nstep_r = nstep_prop
            else:
                # If `nstep_prop = 0` something has gone horribly wrong.
                raise RuntimeError("Slice sampler has failed to find "
                                   "a valid point. Some useful "
                                   "output quantities:\n"
                                   f"u: {u}\n"
                                   f"nstep_left: {nstep_l}\n"
                                   f"nstep_right: {nstep_r}\n"
                                   f"nstep_hat: {nstep_hat}\n"
                                   f"u_prop: {u_prop}\n"
                                   f"loglstar: {loglstar}\n"
                                   f"logl_prop: {logl_prop}\n"
                                   f"direction: {direction}\n")
    v_prop = prior_transform(u_prop)
    return (u_prop, v_prop, logl_prop, nc, n_expand, n_contract,
            expansion_warning)


def tune_slice(sampler, tuning_info, update=True):
    """Update the slice proposal scale based on the relative
    size of the slices compared to our initial guess.
    For slice sampling the scale is only 'advisory' in the sense that
    the right scale will just speed up sampling as we'll have to expand
    or contract less. It won't affect the quality of the samples much.
    The keyword update determines if we are just accumulating the number
    of steps or actually adjusting the scale
    """
    # see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4063214/
    # also 2002.06212
    # https://www.tandfonline.com/doi/full/10.1080/10618600.2013.791193
    # and https://github.com/joshspeagle/dynesty/issues/260
    hist = sampler.slice_history

    hist['n_expand'] += tuning_info['n_expand']
    hist['n_contract'] += tuning_info['n_contract']
    if tuning_info['expansion_warning_set']:
        sampler.sampler_kwargs['slice_doubling'] = True
    if not update:
        return
    n_expand, n_contract = max(hist['n_expand'], 1), hist['n_contract']
    mult = (n_expand * 2. / (n_expand + n_contract))
    # avoid drastic updates to the scale factor limiting to factor
    # of two
    mult = np.clip(mult, 0.5, 2)
    # Remember I can't apply the rule that scale < cube diagonal
    # because scale is multiplied by axes
    sampler.scale = sampler.scale * mult
    hist['n_expand'] = 0
    hist['n_contract'] = 0
