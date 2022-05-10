#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for handling results.

"""

import sys
import copy
import numpy as np
import shutil
from collections import namedtuple

__all__ = ["Results", "print_fn"]

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
    ('eff', 'float', 'Sampling efficiency XXX', None),
    ('nlive', 'int', 'Number of live points for a static run', None),
    ('logvol', 'array[float]', 'Logvolumes of dead points', 'niter'),
    ('information', 'array[float]', 'Information Integral H', 'niter'),
    ('bound', 'array[object]',
     "the set of bounding objects used to condition proposals", 'XXX'),
    ('bound_iter', 'array[XXX]',
     "the iteration when the corresponding bound was created to propose "
     "new live points (e.g., iteration 520)", 'XXX'),
    ('samples_bound', 'array[XXX]',
     "The index of the bound that the corresponding sample was drawn from",
     'niter'),
    ('samples_batch', 'array[XXX]',
     "Tracks the batch during which the samples were proposed", 'nbatch???'),
    ('batch_bounds', 'array[XXX]',
     "The log-likelihood bounds used to run a batch.", 'nbatch???'),
    ('batch_nlive', 'array[int]',
     "The number of live points added in a given batch ???"
     "How is it different from samples_n", 'nbatch???'),
    ('scale', 'array[float]', "Scalar scale applied for proposals", 'niter')
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
            assert (k not in self._keys)  # ensure no duplicates
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

    def summary(self):
        """Return a formatted string giving a quick summary
        of the results."""

        if self._dynamic:
            res = ("niter: {:d}\n"
                   "ncall: {:d}\n"
                   "eff(%): {:6.3f}\n"
                   "logz: {:6.3f} +/- {:6.3f}".format(self.niter,
                                                      sum(self.ncall),
                                                      self.eff, self.logz[-1],
                                                      self.logzerr[-1]))
        else:
            res = ("nlive: {:d}\n"
                   "niter: {:d}\n"
                   "ncall: {:d}\n"
                   "eff(%): {:6.3f}\n"
                   "logz: {:6.3f} +/- {:6.3f}".format(self.nlive, self.niter,
                                                      sum(self.ncall),
                                                      self.eff, self.logz[-1],
                                                      self.logzerr[-1]))

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
