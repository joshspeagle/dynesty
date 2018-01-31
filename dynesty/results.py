#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for handling results.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

__all__ = ["Results", "print_fn"]


def print_fn(results, niter, ncall, add_live_it=None,
             dlogz=None, stop_val=None, nbatch=None,
             logl_min=-np.inf, logl_max=np.inf):
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

    # Extract results at the current iteration.
    (worst, ustar, vstar, loglstar, logvol, logwt,
     logz, logzvar, h, nc, worst_it, boundidx, bounditer,
     eff, delta_logz) = results

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
    print_str = "\r"  # overwrite previous output
    print_str += "iter: {:d}".format(niter)
    if add_live_it is not None:
        print_str += "+{:d}".format(add_live_it)
    print_str += " | "
    if nbatch is not None:
        print_str += "batch: {:d} | ".format(nbatch)
    print_str += "bound: {:d} | ".format(bounditer)
    print_str += "nc: {:d} | ".format(nc)
    print_str += "ncall: {:d} | ".format(ncall)
    print_str += "eff(%): {:6.3f} | ".format(eff)
    print_str += "loglstar: {:6.3f} < {:6.3f} < {:6.3f} | ".format(logl_min,
                                                                   loglstar,
                                                                   logl_max)
    print_str += "logz: {:6.3f} +/- {:6.3f} | ".format(logz, logzerr)
    if dlogz is not None:
        print_str += "dlogz: {:6.3f} > {:6.3f}".format(delta_logz, dlogz)
    else:
        print_str += "stop: {:6.3f}".format(stop_val)
    print_str += "            "  # clear previous output

    # Printing.
    sys.stderr.write(print_str)
    sys.stderr.flush()


class Results(dict):
    """Contains the full output of a run along with a set of helper
    functions for summarizing the output."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(list(map(len, list(self.keys())))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def summary(self):
        """Return a formatted string giving a quick summary
        of the results."""

        res = ("nlive: {:d}\n"
               "niter: {:d}\n"
               "ncall: {:d}\n"
               "eff(%): {:6.3f}\n"
               "logz: {:6.3f} +/- {:6.3f}"
               .format(self.nlive, self.niter, sum(self.ncall),
                       self.eff, self.logz[-1], self.logzerr[-1]))

        print('Summary\n=======\n'+res)
