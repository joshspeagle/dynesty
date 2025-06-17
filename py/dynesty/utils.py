#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of useful functions.

"""

import sys
import warnings
import math
import copy
import time
import os
import shutil
from collections import namedtuple
from functools import partial
import pickle as pickle_module
# To allow replacing of the pickler
import numpy as np
from scipy.special import logsumexp
from ._version import __version__ as DYNESTY_VERSION
try:
    import tqdm
except ImportError:
    tqdm = None

try:
    import h5py
except ImportError:
    h5py = None

__all__ = [
    "unitcheck", "resample_equal", "mean_and_cov", "quantile", "jitter_run",
    "resample_run", "reweight_run", "unravel_run", "merge_runs", "kld_error",
    "get_enlarge_bootstrap", "LoglOutput", "LogLikelihood", "RunRecord",
    "DelayTimer"
]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

IteratorResult = namedtuple('IteratorResult', [
    'worst', 'ustar', 'vstar', 'loglstar', 'logvol', 'logwt', 'logz',
    'logzvar', 'h', 'nc', 'worst_it', 'boundidx', 'bounditer', 'eff',
    'delta_logz', 'blob'
])

IteratorResultShort = namedtuple('IteratorResultShort', [
    'worst', 'ustar', 'vstar', 'loglstar', 'nc', 'worst_it', 'boundidx',
    'bounditer', 'eff'
])

_LOWL_VAL = -1e300


class LoglOutput:
    """
    Class that encapsulates the output of the likelihood function.
    The reason we need this wrapper is to preserve the blob associated with
    the likelihood function.

    """

    def __init__(self, v, blob_flag):
        """
        Initialize the object

        Parameters
        ----------
        v: float or tuple
            if blob_flag is true v have to be a tuple of logl and blob
            if it is False v is just logl
        blob_flag: boolean
            flag to mark whether the v has a blob or not
        """
        if blob_flag:
            self.val = v[0]
            self.blob = v[1]
        else:
            self.val = v

    def __lt__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self) < float(v1)

    def __gt__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self) > float(v1)

    def __le__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self) <= float(v1)

    def __ge__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self) >= float(v1)

    def __eq__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self) == float(v1)

    def __float__(self):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return float(self.val)


class LogLikelihood:
    """
    Class that calls the likelihood function (using a pool if provided)
    Also if requested it saves the history of evaluations
    """

    def __init__(self,
                 loglikelihood,
                 ndim,
                 pool=None,
                 save=False,
                 history_filename=None,
                 blob=False):
        """ Initialize the object.

        Parameters
        ----------
        loglikelihood: function
        ndim: int
            Dimensionality
        pool: Pool (optional)
            Any kind of pool capable of performing map()
        save: bool
            if True the function evaluations will be saved in the hdf5 file
        history_filename: string
            The filename where the history will go
        blob: boolean
            if True we expect the logl output to be a tuple of logl value and
            a blob, otherwise it'll be logl value only
        """
        self.loglikelihood = loglikelihood
        self.pool = pool
        self.history_pars = []
        self.history_logl = []
        self.save_every = 10000
        self.save = save
        self.history_filename = history_filename
        self.ndim = ndim
        self.failed_save = False
        self.blob = blob
        if save:
            self.history_init()

    def map(self, pars):
        """
        Evaluate the likelihood function on the list of vectors
        The pool is used if it was provided when the object was created

        Returns
        -------
        ret: The list of LoglOutput objects
        """
        if self.pool is None:
            ret = list([
                LoglOutput(_, self.blob) for _ in map(self.loglikelihood, pars)
            ])
        else:
            ret = [
                LoglOutput(_, self.blob)
                for _ in self.pool.map(self.loglikelihood, pars)
            ]
        if self.save:
            self.history_append([_.val for _ in ret], pars)
        return ret

    def __call__(self, x):
        """
        Evaluate the likelihood f-n once
        """
        ret = LoglOutput(self.loglikelihood(x), self.blob)
        if self.save:
            self.history_append([ret.val], [x])
        return ret

    def history_append(self, logls, pars):
        """
        Append to the internal history the list of loglikelihood values
        And points
        """
        self.history_logl.extend(logls)
        self.history_pars.extend(pars)
        if len(self.history_logl) > self.save_every:
            self.history_save()

    def history_init(self):
        """ Initialize the hdf5 storage of evaluations """
        if h5py is None:
            raise RuntimeError(
                'h5py module is required for saving history of calls')
        self.history_counter = 0
        try:
            with h5py.File(self.history_filename, mode='w') as fp:
                fp.create_dataset('param', (self.save_every, self.ndim),
                                  maxshape=(None, self.ndim))
                fp.create_dataset('logl', (self.save_every, ),
                                  maxshape=(None, ))
        except OSError:
            print('Failed to initialize history file')
            raise

    def history_save(self):
        """
        Save the actual history from an internal buffer into the file
        """
        if self.failed_save or not self.save:
            # if failed to save before, do not try again
            # also quickly return if saving is not needed
            return
        try:
            with h5py.File(self.history_filename, mode='a') as fp:
                # pylint: disable=no-member
                nadd = len(self.history_logl)
                fp['param'].resize(self.history_counter + nadd, axis=0)
                fp['logl'].resize(self.history_counter + nadd, axis=0)
                fp['param'][-nadd:, :] = np.array(self.history_pars)
                fp['logl'][-nadd:] = np.array(self.history_logl)
                self.history_pars = []
                self.history_logl = []
                self.history_counter += nadd
        except OSError:
            warnings.warn(
                'Failed to save history of evaluations. Will not try again.')
            self.failed_save = True

    def __getstate__(self):
        """Get state information for pickling."""
        state = self.__dict__.copy()
        if 'pool' in state:
            del state['pool']
        return state


class RunRecord:
    """
    This is the dictionary like class that saves the results of the nested
    run so it is basically a collection of various lists of
    quantities
    """

    def __init__(self, dynamic=False):
        """
        If dynamic is true. We initialize the class for
        a dynamic nested run
        """
        D = {}
        keys = [
            'id',  # live point labels
            'u',  # unit cube samples
            'v',  # transformed variable samples
            'logl',  # loglikelihoods of samples
            'logvol',  # expected ln(volume)
            'logwt',  # ln(weights)
            'logz',  # cumulative ln(evidence)
            'logzvar',  # cumulative error on ln(evidence)
            'h',  # cumulative information
            'nc',  # number of calls at each iteration
            'boundidx',  # index of bound dead point was drawn from
            'it',  # iteration the live (now dead) point was proposed
            'n',  # number of live points interior to dead point
            'bounditer',  # active bound at a specific iteration
            'scale',  # scale factor at each iteration
            'blob'  # blobs output by the log-likelihood
        ]
        if dynamic:
            keys.extend([
                'batch',  # live point batch ID
                # these are special since their length
                # is == the number of batches
                'batch_nlive',  # number of live points added in batch
                'batch_bounds'  # loglikelihood bounds used in batch
            ])
        for k in keys:
            D[k] = []
        self.D = D

    def append(self, newD):
        """
        append new information to the RunRecord in the form a dictionary
        i.e. run.append(dict(batch=3, niter=44))
        """
        for k in newD.keys():
            self.D[k].append(newD[k])

    def __getitem__(self, k):
        return self.D[k]

    def __setitem__(self, k, v):
        self.D[k] = v

    def keys(self):
        return self.D.keys()


class DelayTimer:
    """ Utility class that allows us to detect a certain
    time has passed"""

    def __init__(self, delay):
        """ Initialise the time with delay of dt seconds

        Parameters
        ----------

        delay: float
            The number of seconds in the timer
        """
        self.delay = delay
        self.last_time = time.time()

    def is_time(self):
        """
        Returns true if more than self.dt seconds has passed
        since the initialization or last call of successful is_time()

        Returns
        -------
        ret: bool
             True if specified amout of time has passed since the
             initialization or last successful is_time() call
        """
        curt = time.time()
        if curt - self.last_time > self.delay:
            self.last_time = curt
            return True
        return False


PrintFnArgs = namedtuple('PrintFnArgs',
                         ['niter', 'short_str', 'mid_str', 'long_str'])


def print_fn(results,
             niter,
             ncall,
             add_live_it=None,
             dlogz=None,
             stop_val=None,
             nbatch=None,
             logl_min=-np.inf,
             logl_max=np.inf,
             pbar=None):
    """
    The default function used to print out results in real time.

    Parameters
    ----------

    results : tuple
        Collection of variables output from the current state of the sampler.
        Currently includes:
        (1) particle index,
        (2) unit cube position,
        (3) parameter position,
        (4) ln(likelihood),
        (5) ln(volume),
        (6) ln(weight),
        (7) ln(evidence),
        (8) Var[ln(evidence)],
        (9) information,
        (10) number of (current) function calls,
        (11) iteration when the point was originally proposed,
        (12) index of the bounding object originally proposed from,
        (13) index of the bounding object active at a given iteration,
        (14) cumulative efficiency, and
        (15) estimated remaining ln(evidence).

    niter : int
        The current iteration of the sampler.

    ncall : int
        The total number of function calls at the current iteration.

    add_live_it : int, optional
        If the last set of live points are being added explicitly, this
        quantity tracks the sorted index of the current live point being added.

    dlogz : float, optional
        The evidence stopping criterion. If not provided, the provided
        stopping value will be used instead.

    stop_val : float, optional
        The current stopping criterion (for dynamic nested sampling). Used if
        the `dlogz` value is not specified.

    nbatch : int, optional
        The current batch (for dynamic nested sampling).

    logl_min : float, optional
        The minimum log-likelihood used when starting sampling. Default is
        `-np.inf`.

    logl_max : float, optional
        The maximum log-likelihood used when stopping sampling. Default is
        `np.inf`.

    """
    if pbar is None:
        print_fn_fallback(results,
                          niter,
                          ncall,
                          add_live_it=add_live_it,
                          dlogz=dlogz,
                          stop_val=stop_val,
                          nbatch=nbatch,
                          logl_min=logl_min,
                          logl_max=logl_max)
    else:
        print_fn_tqdm(pbar,
                      results,
                      niter,
                      ncall,
                      add_live_it=add_live_it,
                      dlogz=dlogz,
                      stop_val=stop_val,
                      nbatch=nbatch,
                      logl_min=logl_min,
                      logl_max=logl_max)


def get_print_fn_args(results,
                      niter,
                      ncall,
                      add_live_it=None,
                      dlogz=None,
                      stop_val=None,
                      nbatch=None,
                      logl_min=-np.inf,
                      logl_max=np.inf):
    # Extract results at the current iteration.
    loglstar = results.loglstar
    logz = results.logz
    logzvar = results.logzvar
    delta_logz = results.delta_logz
    bounditer = results.bounditer
    nc = results.nc
    eff = results.eff

    # Adjusting outputs for printing.
    if delta_logz > 1e6:
        delta_logz = np.inf
    if logzvar >= 0. and logzvar <= 1e6:
        logzerr = np.sqrt(logzvar)
    else:
        logzerr = np.nan
    if logz <= -1e6:
        logz = -np.inf
    if loglstar <= -1e6:
        loglstar = -np.inf

    # Constructing output.
    long_str = []
    # long_str.append("iter: {:d}".format(niter))
    if add_live_it is not None:
        long_str.append("+{:d}".format(add_live_it))
    short_str = list(long_str)
    if nbatch is not None:
        long_str.append("batch: {:d}".format(nbatch))
    long_str.append("bound: {:d}".format(bounditer))
    long_str.append("nc: {:d}".format(nc))
    long_str.append("ncall: {:d}".format(ncall))
    long_str.append("eff(%): {:6.3f}".format(eff))
    short_str.append(long_str[-1])
    long_str.append("loglstar: {:6.3f} < {:6.3f} < {:6.3f}".format(
        logl_min, loglstar, logl_max))
    short_str.append("logl*: {:6.1f}<{:6.1f}<{:6.1f}".format(
        logl_min, loglstar, logl_max))
    long_str.append("logz: {:6.3f} +/- {:6.3f}".format(logz, logzerr))
    short_str.append("logz: {:6.1f}+/-{:.1f}".format(logz, logzerr))
    mid_str = list(short_str)
    if dlogz is not None:
        long_str.append("dlogz: {:6.3f} > {:6.3f}".format(delta_logz, dlogz))
        mid_str.append("dlogz: {:6.1f}>{:6.1f}".format(delta_logz, dlogz))
    else:
        long_str.append("stop: {:6.3f}".format(stop_val))
        mid_str.append("stop: {:6.3f}".format(stop_val))

    return PrintFnArgs(niter=niter,
                       short_str=short_str,
                       mid_str=mid_str,
                       long_str=long_str)


def print_fn_tqdm(pbar,
                  results,
                  niter,
                  ncall,
                  add_live_it=None,
                  dlogz=None,
                  stop_val=None,
                  nbatch=None,
                  logl_min=-np.inf,
                  logl_max=np.inf):
    """
    This is a function that does the status printing using tqdm module
    """
    fn_args = get_print_fn_args(results,
                                niter,
                                ncall,
                                add_live_it=add_live_it,
                                dlogz=dlogz,
                                stop_val=stop_val,
                                nbatch=nbatch,
                                logl_min=logl_min,
                                logl_max=logl_max)

    pbar.set_postfix_str(" | ".join(fn_args.long_str), refresh=False)
    pbar.update(fn_args.niter - pbar.n)


def print_fn_fallback(results,
                      niter,
                      ncall,
                      add_live_it=None,
                      dlogz=None,
                      stop_val=None,
                      nbatch=None,
                      logl_min=-np.inf,
                      logl_max=np.inf):
    """
    This is a function that does the status printing using just
    standard printing into the console
    """
    fn_args = get_print_fn_args(results,
                                niter,
                                ncall,
                                add_live_it=add_live_it,
                                dlogz=dlogz,
                                stop_val=stop_val,
                                nbatch=nbatch,
                                logl_min=logl_min,
                                logl_max=logl_max)
    niter, short_str, mid_str, long_str = (fn_args.niter, fn_args.short_str,
                                           fn_args.mid_str, fn_args.long_str)

    long_str = ["iter: {:d}".format(niter)] + long_str

    # Printing.
    long_str = ' | '.join(long_str)
    mid_str = ' | '.join(mid_str)
    short_str = '|'.join(short_str)
    if sys.stderr.isatty() and hasattr(shutil, 'get_terminal_size'):
        columns = shutil.get_terminal_size(fallback=(80, 25))[0]
    else:
        columns = 200
    if columns > len(long_str):
        sys.stderr.write("\r" + long_str + ' ' * (columns - len(long_str) - 2))
    elif columns > len(mid_str):
        sys.stderr.write("\r" + mid_str + ' ' * (columns - len(mid_str) - 2))
    else:
        sys.stderr.write("\r" + short_str + ' ' *
                         (columns - len(short_str) - 2))
    sys.stderr.flush()


# List of results attributes as
# Name, type, description, shape (if array)
_RESULTS_STRUCTURE = [
    ('logl', 'array[float]', 'Log likelihood', 'niter'),
    ('samples_it', 'array[int]',
     "the sampling iteration when the sample was proposed "
     "(e.g., iteration 570)", 'niter'),
    ('samples_id', 'array[int]',
     'The unique ID of the sample XXX (within nlive points)', None),
    ('samples_n', 'array[int]',
     'The number of live points at the point when the sample was proposed',
     'niter'),
    ('samples_u', 'array[float]', '''The coordinates of live points in the
    unit cube coordinate system''', 'niter,ndim'),
    ('samples_v', 'array[float]', '''The coordinates of live points''',
     'niter,ndim'),
    ('samples', 'array',
     '''the location (in original coordinates). Identical to samples_v''',
     'niter,ndim'), ('niter', 'int', 'number of iterations', None),
    ('ncall', 'int', 'Total number likelihood calls', None),
    ('logz', 'array', 'Array of cumulative log(Z) integrals', 'niter'),
    ('logzerr', 'array', 'Array of uncertainty of log(Z)', 'niter'),
    ('logwt', 'array', 'Array of log-posterior weights', 'niter'),
    ('eff', 'float', 'Sampling efficiency', None),
    ('nlive', 'int', 'Number of live points for a static run', None),
    ('logvol', 'array[float]', 'Logvolumes of dead points', 'niter'),
    ('information', 'array[float]', 'Information Integral H', 'niter'),
    ('bound', 'array[object]',
     "the set of bounding objects used to condition proposals for the "
     "base run", 'nbound'),
    ('bound_iter', 'array[int]',
     "index of the bound being used for an iteration that generated the point",
     'niter'),
    ('samples_bound', 'array[int]',
     "The index of the bound that the corresponding sample was drawn from",
     'niter'),
    ('samples_batch', 'array[int]',
     "Tracks the batch during which the samples were proposed", 'niter'),
    ('batch_bounds', 'array[tuple]',
     "The log-likelihood bounds used to run a batch.", 'nbatch'),
    ('batch_nlive', 'array[int]',
     "The number of live points used for  given batch", 'nbatch'),
    ('scale', 'array[float]', "Scalar scale applied for proposals", 'niter'),
    ('blob', 'array[]',
     'The auxiliary blobs computed by the log-likelihood function', 'niter')
]


class Results:
    """
    Contains the full output of a run along with a set of helper
    functions for summarizing the output.
    The object is meant to be unchangeable record of the static or
    dynamic nested run.

    Results attributes (name, type, description, array size):
    """

    _ALLOWED = set([_[0] for _ in _RESULTS_STRUCTURE])

    def __init__(self, key_values):
        """
        Initialize the results using the list of key value pairs
        or a dictionary
        Results([('logl', [1, 2, 3]), ('samples_it',[1,2,3])])
        Results(dict(logl=[1, 2, 3], samples_it=[1,2,3]))
        """
        self._keys = []
        self._initialized = False
        if isinstance(key_values, dict):
            key_values_list = key_values.items()
        else:
            key_values_list = key_values
        for k, v in key_values_list:
            assert k not in self._keys  # ensure no duplicates
            assert k in Results._ALLOWED, k
            self._keys.append(k)
            setattr(self, k, copy.copy(v))
        required_keys = ['samples_u', 'samples_id', 'logl', 'samples']
        # TODO I need to add here logz, logzerr
        # but that requires ensuring that merge_runs always computes logz
        for k in required_keys:
            if k not in self._keys:
                raise ValueError('Key %s must be provided' % k)
        if 'nlive' in self._keys:
            self._dynamic = False
        elif 'samples_n' in self._keys:
            self._dynamic = True
        else:
            raise ValueError(
                'Trying to construct results object without nlive '
                'or samples_n information')
        self._initialized = True

    def __copy__(self):
        # this will be a deep copy
        return Results(self.asdict().items())

    def copy(self):
        '''
        return a copy of the object
        all numpy arrays will be copied too
        '''
        return self.__copy__()

    def __setattr__(self, name, value):
        if name[0] != '_' and self._initialized:
            raise RuntimeError("Cannot set attributes directly")
        super().__setattr__(name, value)

    def __getitem__(self, name):
        if name in self._keys:
            return getattr(self, name)
        else:
            raise KeyError(name)

    def __repr__(self):
        m = max(list(map(len, list(self._keys)))) + 1
        return '\n'.join(
            [k.rjust(m) + ': ' + repr(getattr(self, k)) for k in self._keys])

    def __contains__(self, key):
        return key in self._keys

    def keys(self):
        """ Return the list of attributes/keys stored in Results """
        return self._keys

    def items(self):
        """
Return the list of items in the results object as list of key,value pairs
        """
        return ((k, getattr(self, k)) for k in self._keys)

    def asdict(self):
        """
        Return contents of the Results object as dictionary
        """
        # importantly here we copy attribute values
        return dict((k, copy.copy(getattr(self, k))) for k in self._keys)

    def isdynamic(self):
        """ Return true if the results was constructed using dynamic
        nested sampling run with (potentially) variable number of
        live-points"""
        return self._dynamic

    def importance_weights(self):
        """
        Return the importance weights for the each sample.
        """
        logwt = self['logwt'] - self['logz'][-1]
        wt = np.exp(logwt)
        wt = wt / wt.sum()
        return wt

    def samples_equal(self, rstate=None):
        """
        Return the equally weighted samples in random order.
        """
        if rstate is None:
            rstate = get_random_generator()
        return resample_equal(self['samples'],
                              self.importance_weights(),
                              rstate=rstate)

    def summary(self):
        """Return a formatted string giving a quick summary
        of the results."""

        if self._dynamic:
            res = ("niter: {:d}\n"
                   "ncall: {:d}\n"
                   "eff(%): {:6.3f}\n"
                   "logz: {:6.3f} +/- {:6.3f}".format(self['niter'],
                                                      np.sum(self['ncall']),
                                                      self['eff'],
                                                      self['logz'][-1],
                                                      self['logzerr'][-1]))
        else:
            res = ("nlive: {:d}\n"
                   "niter: {:d}\n"
                   "ncall: {:d}\n"
                   "eff(%): {:6.3f}\n"
                   "logz: {:6.3f} +/- {:6.3f}".format(self['nlive'],
                                                      self['niter'],
                                                      np.sum(self['ncall']),
                                                      self['eff'],
                                                      self['logz'][-1],
                                                      self['logzerr'][-1]))

        print('Summary\n=======\n' + res)


Results.__doc__ += '\n\n' + str('\n'.join(
    ['| ' + str(_) for _ in _RESULTS_STRUCTURE])) + '\n'


def results_substitute(results, kw_dict):
    """ This is an utility method that takes a Result object and
substituted certain keys in it. It returns a copy object!
    """
    new_list = []
    for k, w in results.items():
        if k not in kw_dict:
            new_list.append((k, w))
        else:
            new_list.append((k, kw_dict[k]))
    return Results(new_list)


def get_enlarge_bootstrap(sample, enlarge, bootstrap):
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
        if sample == 'unif':
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


def get_nonbounded(ndim, periodic, reflective):
    """
    Return a boolean mask for dimensions that are either
    periodic or reflective. It will be true for normal dimension
    and false for periodic/reflective
    """
    if periodic is not None and reflective is not None:
        if np.intersect1d(periodic, reflective).size > 0:
            raise ValueError("You have specified a parameter as both "
                             "periodic and reflective.")

    if periodic is not None or reflective is not None:
        nonbounded = np.ones(ndim, dtype=bool)
        if periodic is not None:
            if np.max(periodic) >= ndim:
                raise ValueError(
                    'Incorrect periodic variable index (larger than ndim')
            nonbounded[periodic] = False
        if reflective is not None:
            if np.max(reflective) >= ndim:
                raise ValueError(
                    'Incorrect periodic variable index (larger than ndim')
            nonbounded[reflective] = False
    else:
        nonbounded = None

    return nonbounded


def get_print_func(print_func, print_progress):
    pbar = None
    if print_func is None:
        if tqdm is None or not print_progress:
            print_func = print_fn
        else:
            pbar = tqdm.tqdm()
            print_func = partial(print_fn, pbar=pbar)
    return pbar, print_func


def get_random_generator(seed=None):
    """
    Return a random generator (using the seed provided if available)
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.Generator(np.random.PCG64(seed))


def get_seed_sequence(rstate, nitems):
    """
    Return the list of seeds to initialize random generators
    This is useful when distributing work across a pool
    """
    seeds = np.random.SeedSequence(rstate.integers(0, 2**63 - 1,
                                                   size=4)).spawn(nitems)
    return seeds


def get_neff_from_logwt(logwt):
    """
    Compute the number of effective samples from an array of unnormalized
    log-weights. We use Kish Effective Sample Size (ESS)  formula.

    Parameters
    ----------
    logwt: numpy array
        Array of unnormalized weights

    Returns
    -------
    int
        The effective number of samples
    """

    # If weights are normalized to the sum of 1,
    # the estimate is  N = 1/\sum(w_i^2)
    # if the weights are not normalized
    # N = (\sum w_i)^2 / \sum(w_i^2)
    W = np.exp(logwt - logwt.max())
    return W.sum()**2 / (W**2).sum()


def unitcheck(u, nonbounded=None):
    """Check whether `u` is inside the unit cube. Given a masked array
    `nonbounded`, also allows periodic boundaries conditions to exceed
    the unit cube."""

    if nonbounded is None:
        # No periodic boundary conditions provided.
        return u.min() > 0 and u.max() < 1
    else:
        # Alternating periodic and non-periodic boundary conditions.
        unb = u[nonbounded]
        # pylint: disable=invalid-unary-operand-type
        ub = u[~nonbounded]
        return (unb.min() > 0 and unb.max() < 1 and ub.min() > -0.5
                and ub.max() < 1.5)


def apply_reflect(u):
    """
    Iteratively reflect a number until it is contained in [0, 1].

    This is for priors with a reflective boundary condition, all numbers in the
    set `u = 2n +/- x` should be mapped to x.

    For the `+` case we just take `u % 1`.
    For the `-` case we take `1 - (u % 1)`.

    E.g., -0.9, 1.1, and 2.9 should all map to 0.9.

    Parameters
    ----------
    u: array-like
        The array of points to map to the unit cube

    Returns
    -------
    u: array-like
       The input array, modified in place.
    """
    idxs_even = np.mod(u, 2) < 1
    u[idxs_even] = np.mod(u[idxs_even], 1)
    u[~idxs_even] = 1 - np.mod(u[~idxs_even], 1)
    return u


def mean_and_cov(samples, weights):
    """
    Compute the weighted mean and covariance of the samples.

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples, ndim)
        2-D array containing data samples. This ordering is equivalent to
        using `rowvar=False` in `~numpy.cov`.

    weights : `~numpy.ndarray` with shape (nsamples,)
        1-D array of sample weights.

    Returns
    -------
    mean : `~numpy.ndarray` with shape (ndim,)
        Weighted sample mean vector.

    cov : `~numpy.ndarray` with shape (ndim, ndim)
        Weighted sample covariance matrix.

    Notes
    -----
    Implements the formulae found `here <https://goo.gl/emWFLR>`_.

    """

    # Compute the weighted mean.
    mean = np.average(samples, weights=weights, axis=0)

    # Compute the weighted covariance.
    dx = samples - mean
    wsum = np.sum(weights)
    w2sum = np.sum(weights**2)
    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', weights, dx, dx)

    return mean, cov


def resample_equal(samples, weights, rstate=None):
    """
    Resample a set of points from the weighted set of inputs
    such that they all have equal weight. The points are also
    randomly shuffled.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
    with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights in random order.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> utils.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.
   """

    if rstate is None:
        rstate = get_random_generator()

    cumulative_sum = np.cumsum(weights)
    if abs(cumulative_sum[-1] - 1.) > SQRTEPS:
        # same tol as in numpy's random.choice.
        # Guarantee that the weights will sum to 1.
        warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]
    # this ensures that the last element is strictly == 1

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return rstate.permutation(samples[idx])


def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def _get_nsamps_samples_n(res):
    """ Helper function for calculating the number of samples

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    Returns
    -------
    nsamps: int
        The total number of samples/iterations
    samples_n: array
        Number of live points at a given iteration

    """
    if res.isdynamic():
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        nsamps = len(samples_n)
    else:
        # If the number of live points is constant, compute `samples_n`.
        niter = res.niter
        nlive = res.nlive
        nsamps = len(res.logvol)
        if nsamps == niter:
            samples_n = np.ones(niter, dtype=int) * nlive
        elif nsamps == (niter + nlive):
            samples_n = np.minimum(np.arange(nsamps, 0, -1), nlive)
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
    return nsamps, samples_n


def _find_decrease(samples_n):
    """
    Find all instances where the number of live points is either constant
    or increasing.
    Return the mask,
    the values of nlive when nlives starts to decrease
    The ranges of decreasing nlives
    v=[3,2,1,13,13,12,23,22];
    > print(dynesty.utils._find_decrease(v))
    (array([ True, False, False,  True,  True, False,  True, False]),
    [3, 13, 23],
    [[0, 3], [4, 6], (6, 8)])

    """
    nsamps = len(samples_n)
    nlive_flag = np.zeros(nsamps, dtype=bool)
    nlive_start, bounds = [], []
    nlive_flag[1:] = np.diff(samples_n) < 0

    # For all the portions that are decreasing, find out where they start,
    # where they end, and how many live points are present at that given
    # iteration.
    ids = np.nonzero(nlive_flag)[0]
    if len(ids) > 0:
        boundl = ids[0] - 1
        last = ids[0]
        nlive_start.append(samples_n[boundl])
        for curi in ids[1:]:
            if curi == last + 1:
                last += 1
                # we are in the interval of continuisly decreasing values
                continue
            else:
                # we need to close the last interval
                bounds.append([boundl, last + 1])
                nlive_start.append(samples_n[curi - 1])
                last = curi
                boundl = curi - 1
        # we need to close the last interval
        bounds.append((boundl, last + 1))
        nlive_start = np.array(nlive_start)
    return ~nlive_flag, nlive_start, bounds


def jitter_run(res, rstate=None, approx=False):
    """
    Probes **statistical uncertainties** on a nested sampling run by
    explicitly generating a *realization* of the prior volume associated
    with each sample (dead point). Companion function to :meth:`resample_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our "jittered" prior volume realizations.

    """

    if rstate is None:
        rstate = get_random_generator()

    # Initialize evolution of live points over the course of the run.
    nsamps, samples_n = _get_nsamps_samples_n(res)
    logl = res.logl

    # Simulate the prior volume shrinkage associated with our set of "dead"
    # points. At each iteration, if the number of live points is constant or
    # increasing, our prior volume compresses by the maximum value of a set
    # of `K_i` uniformly distributed random numbers (i.e. as `Beta(K_i, 1)`).
    # If instead the number of live points is decreasing, that means we're
    # instead  sampling down a set of uniform random variables
    # (i.e. uniform order statistics).

    if approx:
        nlive_flag = np.ones(nsamps, dtype=bool)
        nlive_start, bounds = [], []
    else:
        nlive_flag, nlive_start, bounds = _find_decrease(samples_n)

    # The maximum out of a set of `K_i` uniformly distributed random variables
    # has a marginal distribution of `Beta(K_i, 1)`.
    t_arr = np.zeros(nsamps)
    t_arr[nlive_flag] = rstate.beta(a=samples_n[nlive_flag], b=1)

    # If we instead are sampling the set of uniform order statistics,
    # we note that the jth largest value is marginally distributed as
    # `Beta(j, K_i-j+1)`. The full joint distribution is::
    #
    #     X_(j) / X_N = (Y_1 + ... + Y_j) / (Y_1 + ... + Y_{K+1})
    #
    # where X_(j) is the prior volume of the live point with the `j`-th
    # *best* likelihood (i.e. prior volume shrinks as likelihood increases)
    # and the `Y_i`'s are i.i.d. exponentially distributed random variables.
    nunif = len(nlive_start)
    for i in range(nunif):
        nstart = nlive_start[i]
        bound = bounds[i]
        sn = samples_n[bound[0]:bound[1]]
        y_arr = rstate.exponential(scale=1.0, size=nstart + 1)
        ycsum = y_arr.cumsum()
        ycsum /= ycsum[-1]
        uorder = ycsum[np.append(nstart, sn - 1)]
        rorder = uorder[1:] / uorder[:-1]
        t_arr[bound[0]:bound[1]] = rorder

    # These are the "compression factors" at each iteration. Let's turn
    # these into associated ln(volumes).
    logvol = np.log(t_arr).cumsum()

    (saved_logwt, saved_logz, saved_logzvar,
     saved_h) = compute_integrals(logl=logl, logvol=logvol)

    # Overwrite items with our new estimates.
    substitute = {
        'logvol': logvol,
        'logwt': saved_logwt,
        'logz': saved_logz,
        'logzerr': np.sqrt(np.maximum(saved_logzvar, 0)),
        'h': saved_h
    }

    new_res = results_substitute(res, substitute)
    return new_res


def compute_integrals(logl=None, logvol=None, reweight=None):
    """
    Compute weights, logzs and variances using quadratic estimator.
    Returns logwt, logz, logzvar, h

    Parameters:
    -----------
    logl: array
        array of log likelihoods
    logvol: array
        array of log volumes
    reweight: array (or None)
        (optional) reweighting array to reweight posterior
    """
    # pylint: disable=invalid-unary-operand-type
    # Unfortunately pylint doesn't get the asserts
    assert logl is not None
    assert logvol is not None

    loglstar_pad = np.concatenate([[-1.e300], logl])

    # we want log(exp(logvol_i)-exp(logvol_(i+1)))
    # assuming that logvol0 = 0
    # log(exp(LV_{i})-exp(LV_{i+1})) =
    # = LV{i} + log(1-exp(LV_{i+1}-LV{i}))
    # = LV_{i+1} - (LV_{i+1} -LV_i) + log(1-exp(LV_{i+1}-LV{i}))
    dlogvol = np.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
    # logdvol is log(delta(volumes)) i.e. log (X_i-X_{i-1})
    logdvol2 = logdvol + math.log(0.5)
    # These are log(1/2(X_(i+1)-X_i))

    dlogvol = -np.diff(logvol, prepend=0)
    # this are delta(log(volumes)) of the run

    # These are log((L_i+L_{i_1})*(X_i+1-X_i)/2)
    saved_logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        saved_logwt = saved_logwt + reweight
    saved_logz = np.logaddexp.accumulate(saved_logwt)
    # This implements eqn 16 of Speagle2020

    logzmax = saved_logz[-1]
    # we'll need that to just normalize likelihoods to avoid overflows

    # H is defined as
    # H = 1/z int( L * ln(L) dX,X=0..1) - ln(z)
    # incomplete H can be defined as
    # H = int( L/Z * ln(L) dX,X=0..x) - z_x/Z * ln(Z)
    h_part1 = np.cumsum(
        (np.exp(loglstar_pad[1:] - logzmax + logdvol2) * loglstar_pad[1:] +
         np.exp(loglstar_pad[:-1] - logzmax + logdvol2) * loglstar_pad[:-1]))
    # here we divide the likelihood by zmax to avoid to overflow
    saved_h = h_part1 - logzmax * np.exp(saved_logz - logzmax)
    # changes in h in each step
    dh = np.diff(saved_h, prepend=0)

    # I'm applying abs() here to avoid nans down the line
    # because partial H integrals could be negative
    saved_logzvar = np.abs(np.cumsum(dh * dlogvol))
    return saved_logwt, saved_logz, saved_logzvar, saved_h


def progress_integration(loglstar, loglstar_new, logz, logzvar, logvol,
                         dlogvol, h):
    """
    This is the calculation of weights and logz/var estimates one step at the
    time.
    Importantly the calculation of H is somewhat different from
    compute_integrals as incomplete integrals of H() of require knowing Z

    Return logwt, logz, logzvar, h
    """
    # Compute relative contribution to results.
    logdvol = logsumexp(a=[logvol + dlogvol, logvol], b=[0.5, -0.5])
    logwt = np.logaddexp(loglstar_new, loglstar) + logdvol  # weight
    logz_new = np.logaddexp(logz, logwt)  # ln(evidence)
    lzterm = (math.exp(loglstar - logz_new + logdvol) * loglstar +
              math.exp(loglstar_new - logz_new + logdvol) * loglstar_new)
    h_new = (lzterm + math.exp(logz - logz_new) * (h + logz) - logz_new
             )  # information
    dh = h_new - h

    logzvar_new = logzvar + dh * dlogvol
    # var[ln(evidence)] estimate
    return logwt, logz_new, logzvar_new, h_new


def resample_run(res, rstate=None, return_idx=False):
    """
    Probes **sampling uncertainties** on a nested sampling run using bootstrap
    resampling techniques to generate a *realization* of the (expected) prior
    volume(s) associated with each sample (dead point). This effectively
    splits a nested sampling run with `K` particles (live points) into a
    series of `K` "strands" (i.e. runs with a single live point) which are then
    bootstrapped to construct a new "resampled" run. Companion function to
    :meth:`jitter_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    return_idx : bool, optional
        Whether to return the list of resampled indices used to construct
        the new run. Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        samples and weights based on our "bootstrapped" samples and
        (expected) prior volumes.

    """

    if rstate is None:
        rstate = get_random_generator()

    # Check whether the final set of live points were added to the
    # run.
    nsamps = len(res.ncall)
    if res.isdynamic():
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        samples_batch = res.samples_batch
        batch_bounds = res.batch_bounds
        added_final_live = True
    else:
        # If the number of live points is constant, compute `samples_n` and
        # set up the `added_final_live` flag.
        nlive = res.nlive
        niter = res.niter
        if nsamps == niter:
            samples_n = np.ones(niter, dtype=int) * nlive
            added_final_live = False
        elif nsamps == (niter + nlive):
            samples_n = np.minimum(np.arange(nsamps, 0, -1), nlive)
            added_final_live = True
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
        samples_batch = np.zeros(len(samples_n), dtype=int)
        batch_bounds = np.array([(-np.inf, np.inf)])
    batch_llmin = batch_bounds[:, 0]
    # Identify unique particles that make up each strand.
    ids = np.unique(res.samples_id)

    # Split the set of strands into two groups: a "baseline" group that
    # contains points initially sampled from the prior, which gives information
    # on the evidence, and an "add-on" group, which gives additional
    # information conditioned on our baseline strands.
    base_ids = []
    addon_ids = []
    for i in ids:
        sbatch = samples_batch[res.samples_id == i]
        if np.any(batch_llmin[sbatch] == -np.inf):
            base_ids.append(i)
        else:
            addon_ids.append(i)
    nbase, nadd = len(base_ids), len(addon_ids)
    base_ids, addon_ids = np.array(base_ids), np.array(addon_ids)

    # Resample strands.
    if nbase > 0 and nadd > 0:
        live_idx = np.append(base_ids[rstate.integers(0, nbase, size=nbase)],
                             addon_ids[rstate.integers(0, nadd, size=nadd)])
    elif nbase > 0:
        live_idx = base_ids[rstate.integers(0, nbase, size=nbase)]
    elif nadd > 0:
        raise ValueError("The provided `Results` does not include any points "
                         "initially sampled from the prior!")
    else:
        raise ValueError("The provided `Results` does not appear to have "
                         "any particles!")

    # Find corresponding indices within the original run.
    samp_idx = np.arange(len(res.ncall))
    samp_idx = np.concatenate(
        [samp_idx[res.samples_id == idx] for idx in live_idx])

    # Derive new sample size.
    nsamps = len(samp_idx)

    # Sort the loglikelihoods (there will be duplicates).
    logls = res.logl[samp_idx]
    idx_sort = np.argsort(logls)
    samp_idx = samp_idx[idx_sort]
    logl = res.logl[samp_idx]

    if added_final_live:
        # Compute the effective number of live points for each sample.
        samp_n = np.zeros(nsamps, dtype=int)
        uidxs, uidxs_n = np.unique(live_idx, return_counts=True)
        for uidx, uidx_n in zip(uidxs, uidxs_n):
            sel = (res.samples_id == uidx)  # selection flag
            sbatch = samples_batch[sel][0]  # corresponding batch ID
            lower = batch_llmin[sbatch]  # lower bound
            upper = max(res.logl[sel])  # upper bound

            # Add number of live points between endpoints equal to number of
            # times the strand has been resampled.
            samp_n[(logl > lower) & (logl < upper)] += uidx_n

            # At the endpoint, divide up the final set of points into `uidx_n`
            # (roughly) equal chunks and have live points decrease across them.
            endsel = (logl == upper)
            endsel_n = np.count_nonzero(endsel)
            chunk = endsel_n / uidx_n  # define our chunk
            counters = np.array(np.arange(endsel_n) / chunk, dtype=int)
            nlive_end = counters[::-1] + 1  # decreasing number of live points
            samp_n[endsel] += nlive_end  # add live point sequence
    else:
        # If we didn't add the final set of live points, the run has a constant
        # number of live points and can simply be re-ordered.
        samp_n = samples_n[samp_idx]

    # Assign log(volume) to samples.
    logvol = np.cumsum(np.log(samp_n / (samp_n + 1.)))

    saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
        logl=logl, logvol=logvol)

    # Compute sampling efficiency.
    eff = 100. * len(res.ncall[samp_idx]) / sum(res.ncall[samp_idx])

    # Copy results.
    # Overwrite items with our new estimates.
    new_res_dict = dict(niter=len(res.ncall[samp_idx]),
                        ncall=res.ncall[samp_idx],
                        eff=eff,
                        blob=res.blob[samp_idx],
                        samples=res.samples[samp_idx],
                        samples_id=res.samples_id[samp_idx],
                        samples_it=res.samples_it[samp_idx],
                        samples_u=res.samples_u[samp_idx],
                        samples_n=samp_n,
                        logwt=np.asarray(saved_logwt),
                        logl=logl,
                        logvol=logvol,
                        logz=np.asarray(saved_logz),
                        logzerr=np.sqrt(
                            np.maximum(np.asarray(saved_logzvar), 0)),
                        information=np.asarray(saved_h))
    new_res = Results(new_res_dict)

    if return_idx:
        return new_res, samp_idx
    else:
        return new_res


def reweight_run(res, logp_new, logp_old=None):
    """
    Reweight a given run based on a new target distribution.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    logp_new : `~numpy.ndarray` with shape (nsamps,)
        New target distribution evaluated at the location of the samples.

    logp_old : `~numpy.ndarray` with shape (nsamps,)
        Old target distribution evaluated at the location of the samples.
        If not provided, the `logl` values from `res` will be used.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our reweighted samples.

    """

    # Extract info.
    if logp_old is None:
        logp_old = res['logl']
    logrwt = logp_new - logp_old  # ln(reweight)
    logvol = res['logvol']
    logl = res['logl']

    saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
        logl=logl, logvol=logvol, reweight=logrwt)

    # Overwrite items with our new estimates.
    substitute = {
        'logvol': logvol,
        'logwt': saved_logwt,
        'logz': saved_logz,
        'logzerr': np.sqrt(np.maximum(saved_logzvar, 0)),
        'h': saved_h
    }

    new_res = results_substitute(res, substitute)
    return new_res


def unravel_run(res, print_progress=True):
    """
    Unravels a run with `K` live points into `K` "strands" (a nested sampling
    run with only 1 live point). **WARNING: the anciliary quantities provided
    with each unraveled "strand" are only valid if the point was initialized
    from the prior.**

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    new_res : list of :class:`~dynesty.results.Results` instances
        A list of new :class:`~dynesty.results.Results` instances
        for each individual strand.

    """

    idxs = res.samples_id  # label for each live/dead point

    # Check if we added in the last set of dead points.
    added_live = True
    try:
        if len(idxs) != (res.niter + res.nlive):
            added_live = False
    except AttributeError:
        pass

    if (np.diff(res.logl) == 0).sum() == 0:
        warnings.warn('The likelihood seem to have plateaus. '
                      'The unraveling such runs may be inaccurate')

    # Recreate the nested sampling run for each strand.
    new_res = []
    nstrands = len(np.unique(idxs))
    for counter, idx in enumerate(np.unique(idxs)):
        # Select strand `idx`.
        strand = (idxs == idx)
        nsamps = sum(strand)
        logl = res.logl[strand]

        # Assign log(volume) to samples. With K=1 live point, the expected
        # shrinking in `logvol` at each iteration is `-log(2)` (i.e.
        # shrinking by 1/2). If the final set of live points were added,
        # the expected value of the final live point is a uniform
        # sample and so has an expected value of half the volume
        # of the final dead point.
        if added_live:
            niter = nsamps - 1
            logvol_dead = -math.log(2) * (1. + np.arange(niter))
            if niter > 0:
                logvol_live = logvol_dead[-1] + math.log(0.5)
                logvol = np.append(logvol_dead, logvol_live)
            else:  # point always live
                logvol = np.array([math.log(0.5)])
        else:
            niter = nsamps
            logvol = -math.log(2) * (1. + np.arange(niter))

        saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
            logl=logl, logvol=logvol)

        # Compute sampling efficiency.
        eff = 100. * nsamps / sum(res.ncall[strand])

        # Save results.
        rdict = dict(nlive=1,
                     niter=niter,
                     ncall=res.ncall[strand],
                     eff=eff,
                     samples=res.samples[strand],
                     samples_id=res.samples_id[strand],
                     samples_it=res.samples_it[strand],
                     samples_u=res.samples_u[strand],
                     blob=res.blob[strand],
                     logwt=saved_logwt,
                     logl=logl,
                     logvol=logvol,
                     logz=saved_logz,
                     logzerr=np.sqrt(saved_logzvar),
                     information=saved_h)

        # Add on batch information (if available).
        try:
            rdict['samples_batch'] = res.samples_batch[strand]
            rdict['batch_bounds'] = res.batch_bounds
        except AttributeError:
            pass

        # Append to list of strands.
        new_res.append(Results(rdict))

        # Print progress.
        if print_progress:
            sys.stderr.write(f'\rStrand: {counter+1}/{nstrands}     ')

    return new_res


def merge_runs(res_list, print_progress=True):
    """
    Merges a set of runs with differing (possibly variable) numbers of
    live points into one run.

    Parameters
    ----------
    res_list : list of :class:`~dynesty.results.Results` instances
        A list of :class:`~dynesty.results.Results` instances returned from
        previous runs.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    combined_res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance for the combined run.

    """

    ntot = len(res_list)
    counter = 0

    # Establish our set of baseline runs and "add-on" runs.
    rlist_base = []
    rlist_add = []
    for r in res_list:
        try:
            if np.any(r.samples_batch == 0):
                rlist_base.append(r)
            else:
                rlist_add.append(r)
        except AttributeError:
            rlist_base.append(r)
    nbase, nadd = len(rlist_base), len(rlist_add)
    if nbase == 1 and nadd == 1:
        rlist_base = res_list
        rlist_add = []

    # Merge baseline runs while there are > 2 remaining results.
    if len(rlist_base) > 1:
        while len(rlist_base) > 2:
            rlist_new = []
            nruns = len(rlist_base)
            i = 0
            while i < nruns:
                try:
                    # Ignore posterior quantities while merging the runs.
                    r1, r2 = rlist_base[i], rlist_base[i + 1]
                    res = _merge_two(r1, r2, compute_aux=False)
                    rlist_new.append(res)
                except IndexError:
                    # Append the odd run to the new list.
                    rlist_new.append(rlist_base[i])
                i += 2
                counter += 1
                # Print progress.
                if print_progress:
                    sys.stderr.write(f'\rMerge: {counter}/{ntot}     ')
            # Overwrite baseline set of results with merged results.
            rlist_base = copy.copy(rlist_new)

        # Compute posterior quantities after merging the final baseline runs.
        res = _merge_two(rlist_base[0], rlist_base[1], compute_aux=True)
    else:
        res = rlist_base[0]

    # Iteratively merge any remaining "add-on" results.
    nruns = len(rlist_add)
    for i, r in enumerate(rlist_add):
        if i < nruns - 1:
            res = _merge_two(res, r, compute_aux=False)
        else:
            res = _merge_two(res, r, compute_aux=True)
        counter += 1
        # Print progress.
        if print_progress:
            sys.stderr.write(f'\rMerge: {counter}/{ntot}     ')

    res = check_result_static(res)

    return res


def check_result_static(res):
    """ If the run was from a dynamic run but had constant
    number of live points, return a new Results object with
    nlive parameter, so we could use it as static run
    """
    samples_n = _get_nsamps_samples_n(res)[1]
    nlive = max(samples_n)
    niter = res.niter
    standard_run = False

    # Check if we have a constant number of live points.
    if samples_n.size == niter and np.all(samples_n == nlive):
        standard_run = True

    # Check if we have a constant number of live points where we have
    # recycled the final set of live points.
    nlive_test = np.minimum(np.arange(niter, 0, -1), nlive)
    if samples_n.size == niter and np.all(samples_n == nlive_test):
        standard_run = True
    # If the number of live points is consistent with a standard nested
    # sampling run, slightly modify the format to keep with previous usage.
    if standard_run:
        resdict = res.asdict()
        resdict['nlive'] = nlive
        resdict['niter'] = niter - nlive
        res = Results(resdict)
    return res


def kld_error(res,
              error='jitter',
              rstate=None,
              return_new=False,
              approx=False):
    """
    Computes the `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`_ *from* the
    discrete probability distribution defined by `res` *to* the discrete
    probability distribution defined by a **realization** of `res`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        :class:`~dynesty.results.Results` instance for the distribution we
        are computing the KL divergence *from*.

    error : {`'jitter'`, `'resample'`}, optional
        The error method employed, corresponding to :meth:`jitter_run` or
        :meth:`resample_run`. Default is `'jitter'`.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    return_new : bool, optional
        Whether to return the realization of the run used to compute the
        KL divergence. Default is `False`.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    kld : `~numpy.ndarray` with shape (nsamps,)
        The cumulative KL divergence defined *from* `res` *to* a
        random realization of `res`.

    new_res : :class:`~dynesty.results.Results` instance, optional
        The :class:`~dynesty.results.Results` instance corresponding to
        the random realization we computed the KL divergence *to*.

    """

    # Define our original importance weights.
    logp2 = res.logwt - res.logz[-1]

    # Compute a random realization of our run.
    if error == 'jitter':
        new_res = jitter_run(res, rstate=rstate, approx=approx)
    elif error == 'resample':
        new_res, samp_idx = resample_run(res, rstate=rstate, return_idx=True)
        logp2 = logp2[samp_idx]  # re-order our original results to match
    else:
        raise ValueError("Input `'error'` option '{error}' is not valid.")

    # Define our new importance weights.
    logp1 = new_res['logwt'] - new_res['logz'][-1]

    # Compute the KL divergence.
    kld = np.cumsum(np.exp(logp1) * (logp1 - logp2))

    if return_new:
        return kld, new_res
    else:
        return kld


def _prepare_for_merge(res):
    """
    Internal method used to prepare a run for merging with another run.
    It takes the results object and it returns the dictionary with basic run
    info and the number of live points at each iteration.
    """
    # Initialize the first ("base") run.
    run_info = dict(id=res.samples_id,
                    u=res.samples_u,
                    v=res.samples,
                    logl=res.logl,
                    nc=res.ncall,
                    it=res.samples_it,
                    blob=res.blob)
    nrun = len(run_info['id'])

    # Number of live points throughout the run.
    if res.isdynamic():
        run_nlive = res.samples_n
    else:
        niter, nlive = res.niter, res.nlive
        if nrun == niter:
            run_nlive = np.ones(niter, dtype=int) * nlive
        elif nrun == (niter + nlive):
            # this is the case where the last live points are added
            # one by one in the end of the run
            run_nlive = np.minimum(np.arange(nrun, 0, -1), nlive)
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points in `res1`.")

    # Batch information (if available).
    # note we also check for existance of batch_bounds
    # because unravel_run makes 'static' runs of 1 livepoint
    # but some will have bounds
    if res.isdynamic() or 'batch_bounds' in res.keys():
        run_info['batch'] = res.samples_batch
        run_info['bounds'] = res.batch_bounds
    else:
        run_info['batch'] = np.zeros(nrun, dtype=int)
        run_info['bounds'] = np.array([(-np.inf, np.inf)])
    return run_nlive, run_info


def _merge_two(res1, res2, compute_aux=False):
    """
    Internal method used to merges two runs with differing (possibly variable)
    numbers of live points into one run.

    Parameters
    ----------
    res1 : :class:`~dynesty.results.Results` instance
        The "base" nested sampling run.

    res2 : :class:`~dynesty.results.Results` instance
        The "new" nested sampling run.

    compute_aux : bool, optional
        Whether to compute auxiliary quantities (evidences, etc.) associated
        with a given run. **WARNING: these are only valid if `res1` or `res2`
        was initialized from the prior *and* their sampling bounds overlap.**
        Default is `False`.

    Returns
    -------
    res : :class:`~dynesty.results.Results` instances
        :class:`~dynesty.results.Results` instance from the newly combined
        nested sampling run.

    """
    base_nlive, base_info = _prepare_for_merge(res1)
    new_nlive, new_info = _prepare_for_merge(res2)
    base_nsamples = len(base_info['id'])
    new_nsamples = len(new_info['id'])
    # Initialize our new combined run.
    combined_info = dict()
    for curk in [
            'id', 'u', 'v', 'logl', 'logvol', 'logwt', 'logz', 'logzvar', 'h',
            'nc', 'it', 'n', 'batch', 'blob'
    ]:
        combined_info[curk] = []

    # These are merged batch bounds
    combined_bounds = np.unique(np.concatenate(
        (base_info['bounds'], new_info['bounds'])),
                                axis=0)
    # Here we try to find where the new bounds are in the combined bounds
    new_bound_map = {}
    base_bound_map = {}
    for i in range(len(new_info['bounds'])):
        new_bound_map[i] = np.where(
            np.all(new_info['bounds'][i] == combined_bounds, axis=1))[0][0]
    for i in range(len(base_info['bounds'])):
        base_bound_map[i] = np.where(
            np.all(base_info['bounds'][i] == combined_bounds, axis=1))[0][0]

    base_lowedge = np.min(base_info['bounds'][base_info['batch']])
    new_lowedge = np.min(new_info['bounds'][new_info['batch']])

    # Iteratively walk through both set of samples to simulate
    # a combined run.
    combined_nsamples = base_nsamples + new_nsamples
    # Start our counters at the beginning of each set of dead points.
    base_idx, new_idx = 0, 0
    for i in range(combined_nsamples):
        # Attempt to step along our samples. If we're out of samples,
        # set values to defaults.
        if base_idx < base_nsamples:
            base_cur_logl = base_info['logl'][base_idx]
            base_cur_nlive = base_nlive[base_idx]
        else:
            base_cur_logl = np.inf
            base_cur_nlive = 0
            # TODO this is potentially incorrect
            # It is not clear what nlive should be when we
            # are past the end of one of the run
        if new_idx < new_nsamples:
            new_cur_logl = new_info['logl'][new_idx]
            new_cur_nlive = new_nlive[new_idx]
        else:
            new_cur_logl = np.inf
            new_cur_nlive = 0

        if base_cur_logl > new_lowedge and new_cur_logl > base_lowedge:
            # If our samples from the both runs are past the each others'
            # lower log-likelihood bound, both runs are now "active".
            cur_nlive = base_cur_nlive + new_cur_nlive
        elif base_cur_logl <= new_lowedge:
            # If instead our collection of dead points from the "base" run
            # are below the bound, just use those.
            cur_nlive = base_cur_nlive
        else:
            # Our collection of dead points from the "new" run
            # are below the bound, so just use those.
            cur_nlive = new_cur_nlive

        # Increment our position along depending on
        # which dead point (saved or new) is worse.

        if base_cur_logl <= new_cur_logl:
            add_idx = base_idx
            from_run = base_info
            from_map = base_bound_map
            base_idx += 1
        else:
            add_idx = new_idx
            from_run = new_info
            from_map = new_bound_map
            new_idx += 1
        combined_info['batch'].append(from_map[from_run['batch'][add_idx]])

        for curk in ['id', 'u', 'v', 'logl', 'nc', 'it', 'blob']:
            combined_info[curk].append(from_run[curk][add_idx])

        combined_info['n'].append(cur_nlive)

    plateau_mode = False
    plateau_counter = 0
    plateau_logdvol = 0
    logvol = 0.
    logl_array = np.array(combined_info['logl'])
    nlive_array = np.array(combined_info['n'])
    for i, (curl, nlive) in enumerate(zip(logl_array, nlive_array)):
        # Save the number of live points and expected ln(volume).
        if not plateau_mode:
            plateau_mask = (logl_array[i:] == curl)
            nplateau = plateau_mask.sum()
            if nplateau > 1:
                # the number of live points should not change throughout
                # the plateau
                # assert np.ptp(nlive_array[i:][plateau_mask]) == 0
                # TODO currently I disabled this check

                plateau_counter = nplateau
                plateau_logdvol = logvol + np.log(1. / (nlive + 1))
                plateau_mode = True
        if not plateau_mode:
            logvol -= math.log((nlive + 1.) / nlive)
        else:
            logvol = logvol + np.log1p(-np.exp(plateau_logdvol - logvol))
        combined_info['logvol'].append(logvol)
        if plateau_mode:
            plateau_counter -= 1
            if plateau_counter == 0:
                plateau_mode = False
    # Compute sampling efficiency.
    eff = 100. * combined_nsamples / sum(combined_info['nc'])

    # Save results.
    r = dict(niter=combined_nsamples,
             ncall=np.asarray(combined_info['nc']),
             eff=eff,
             samples=np.asarray(combined_info['v']),
             logl=np.asarray(combined_info['logl']),
             logvol=np.asarray(combined_info['logvol']),
             batch_bounds=np.asarray(combined_bounds),
             blob=np.asarray(combined_info['blob']))

    for curk in ['id', 'it', 'n', 'u', 'batch']:
        r['samples_' + curk] = np.asarray(combined_info[curk])

    # Compute the posterior quantities of interest if desired.
    if compute_aux:

        (r['logwt'], r['logz'], combined_logzvar,
         r['information']) = compute_integrals(logvol=r['logvol'],
                                               logl=r['logl'])
        r['logzerr'] = np.sqrt(np.maximum(combined_logzvar, 0))

        # Compute batch information.
        combined_id = np.asarray(combined_info['id'])
        batch_nlive = [
            len(np.unique(combined_id[combined_info['batch'] == i]))
            for i in np.unique(combined_info['batch'])
        ]

        # Add to our results.
        r['batch_nlive'] = np.array(batch_nlive, dtype=int)

    # Combine to form final results object.
    res = Results(r)

    return res


def _kld_error(args):
    """ Internal `pool.map`-friendly wrapper for :meth:`kld_error`
    used by :meth:`stopping_function`."""

    # Extract arguments.
    results, error, approx, rseed = args
    rstate = get_random_generator(rseed)
    return kld_error(results,
                     error,
                     rstate=rstate,
                     return_new=True,
                     approx=approx)


def old_stopping_function(results,
                          args=None,
                          rstate=None,
                          M=None,
                          return_vals=False):
    """
    The old stopping function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns the run a stopping value based on a weighted average of the
    stopping values for the posterior and evidence::
        stop = pfrac * stop_post + (1.- pfrac) * stop_evid
    The evidence stopping value is based on the estimated evidence error
    (i.e. standard deviation) relative to a given threshold::
        stop_evid = evid_std / evid_thresh
    The posterior stopping value is based on the fractional error (i.e.
    standard deviation / mean) in the Kullback-Leibler (KL) divergence
    relative to a given threshold::
        stop_post = (kld_std / kld_mean) / post_thresh
    Estimates of the mean and standard deviation are computed using `n_mc`
    realizations of the input using a provided `'error'` keyword (either
    `'jitter'` or `'resample'`).
    Returns the boolean `stop <= 1`. If `True`, the :class:`DynamicSampler`
    will stop adding new samples to our results.
    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.
    args : dictionary of keyword arguments, optional
        Arguments used to set the stopping values. Default values are
        `pfrac = 1.0`, `evid_thresh = 0.1`, `post_thresh = 0.02`,
        `n_mc = 128`, `error = 'jitter'`, and `approx = True`.
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

    with warnings.catch_warnings():
        warnings.filterwarnings("once")
        warnings.warn(
            "This an old stopping function that will "
            "be removed in future releases", DeprecationWarning)
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
    post_thresh = args.get('post_thresh', 0.02)
    if pfrac > 0. and post_thresh < 0.:
        raise ValueError(
            f"The provided `post_thresh` {post_thresh} is not non-negative "
            f"even though `pfrac` is {pfrac}.")
    n_mc = args.get('n_mc', 128)
    if n_mc <= 1:
        raise ValueError(f"The number of realizations {n_mc} must be greater "
                         "than 1.")
    if n_mc < 20:
        warnings.warn("Using a small number of realizations might result in "
                      "excessively noisy stopping value estimates.")
    error = args.get('error', 'jitter')
    if error not in {'jitter', 'resample'}:
        raise ValueError(f"The chosen `'error'` option {error} is not valid.")
    approx = args.get('approx', True)

    # Compute realizations of ln(evidence) and the KL divergence.
    rlist = [results for i in range(n_mc)]
    error_list = [error for i in range(n_mc)]
    approx_list = [approx for i in range(n_mc)]
    seeds = get_seed_sequence(rstate, n_mc)
    args = zip(rlist, error_list, approx_list, seeds)
    outputs = list(M(_kld_error, args))
    kld_arr, lnz_arr = np.array([(kld[-1], res.logz[-1])
                                 for kld, res in outputs]).T

    # Evidence stopping value.
    lnz_std = np.std(lnz_arr)
    stop_evid = lnz_std / evid_thresh

    # Posterior stopping value.
    kld_mean, kld_std = np.mean(kld_arr), np.std(kld_arr)
    stop_post = (kld_std / kld_mean) / post_thresh

    # Effective stopping value.
    stop = pfrac * stop_post + (1. - pfrac) * stop_evid

    if return_vals:
        return stop <= 1., (stop_post, stop_evid, stop)
    else:
        return stop <= 1.


def restore_sampler(fname, pool=None):
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

    Returns
    -------
    Static or dynamic nested sampling object

    """
    with open(fname, 'rb') as fp:
        res = pickle_module.load(fp)
    sampler = res['sampler']
    save_ver = res['version']
    dynesty_format_version = 1
    file_format_version = res['format_version']
    if file_format_version != dynesty_format_version:
        raise RuntimeError('Incorrect format version')
    if save_ver != DYNESTY_VERSION:
        warnings.warn(
            f'The dynesty version in the checkpoint file ({save_ver})'
            f'does not match the current dynesty version'
            f'({DYNESTY_VERSION}). That is *NOT* guaranteed to work')
    if pool is not None:
        mapper = pool.map
    else:
        mapper = map
    if hasattr(sampler, 'sampler'):
        # This is the case of the dynamic sampler
        # this is better be written as isinstanceof()
        # but I couldn't do it due to circular imports
        # TODO

        # Here we are dealing with the special case of dynamic sampler
        # where it has internal samplers that also need their pool configured
        # this is the initial sampler
        samplers = [sampler, sampler.sampler]
        if sampler.batch_sampler is not None:
            samplers.append(sampler.batch_sampler)
    else:
        samplers = [sampler]

    for cursamp in samplers:
        cursamp.M = mapper
        cursamp.pool = pool
        cursamp.loglikelihood.pool = pool
    return sampler


def save_sampler(sampler, fname):
    """
    Save the state of the dynamic sampler in a file

    Parameters
    ----------
    sampler: object
        Dynamic or Static nested sampler
    fname: string
        Filename of the save file.

    """
    format_version = 1
    # this is an internal version of the format we are
    # using. Increase this if incompatible changes are being made
    D = {
        'sampler': sampler,
        'version': DYNESTY_VERSION,
        'format_version': format_version
    }
    tmp_fname = fname + '.tmp'
    try:
        with open(tmp_fname, 'wb') as fp:
            pickle_module.dump(D, fp)
        try:
            os.rename(tmp_fname, fname)
        except FileExistsError:
            # this can happen in Windows, See #450
            shutil.move(tmp_fname, fname)
    except:  # noqa
        try:
            os.unlink(tmp_fname)
        except:  # noqa
            pass
        raise
