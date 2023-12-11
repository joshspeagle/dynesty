#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The wrapper around multiprocessing pool that can be helpful
with dynesty since it avoids some overhead that one would get
with standard pool
"""

import multiprocessing as mp

__all__ = ['Pool']


class FunctionCache:
    """
    Singleton class to cache the functions and optional arguments between calls
    """


def initializer(loglike, prior_transform, logl_args, logl_kwargs, ptform_args,
                ptform_kwargs):
    """
    Initialized function used to initialize the
    singleton object inside each worker of the pool
    """
    FunctionCache.loglike = loglike
    FunctionCache.prior_transform = prior_transform
    FunctionCache.logl_args = logl_args
    FunctionCache.logl_kwargs = logl_kwargs
    FunctionCache.ptform_args = ptform_args
    FunctionCache.ptform_kwargs = ptform_kwargs


def loglike_cache(x, *args, **kwargs):
    """
    Likelihood function call
    """
    return FunctionCache.loglike(x, *FunctionCache.logl_args, *args,
                                 **FunctionCache.logl_kwargs, **kwargs)


def prior_transform_cache(x, *args, **kwargs):
    """
    Prior transform call
    """
    return FunctionCache.prior_transform(x, *FunctionCache.ptform_args, *args,
                                         **FunctionCache.ptform_kwargs,
                                         **kwargs)


class Pool:
    """
    The multiprocessing pool wrapper class
    It is intended to be used as a context manager for dynesty sampler only.

    Parameters
    ----------
    njobs: int
        The number of multiprocessing jobs/processes
    loglike: function
        ln(likelihood) function
    prior_transform: function
        Function transforming from a unit cube to the parameter
        space of interest according to the prior
    logl_args: tuple(optional)
        The optional arguments to be added to the likelihood
        function call. Note that if you specify the additional
        arguments here, you do not need to provide them again
        to the sampler.
    logl_kwargs: tuple(optional)
        The optional keywords to be added to the likelihood
        function call
    ptform_args: tuple(optional)
        The optional arguments to be added to the prior transform
        function call
    ptform_kwargs: tuple(optional)
        The optional keywords to be added to the prior transform
        function call

    Attributes
    ----------
    loglike: function
        ln(likelihood) function
    prior_transform: function
        Function transforming from a unit cube to the parameter
        space of interest according to the prior

    Examples
    --------
    To use the dynesty pool you have to use it with the context manager::

        with dynesty.pool.Pool(16, loglike, prior_transform) as pool:
            dns = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim,
                                     pool=pool)

    Also note that you have to provide the .loglike/.prior_transform attributes
    from the pool object to the Nested samper rather than your original
    functions!

    If your likelihood function takes additional arguments, it is better to
    pass them when creating the pool, rather then to nested sampler::

        with dynesty.pool.Pool(16, loglike, prior_transform, 
                                            logl_args=(...) ) as pool:
            dns = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim,
                                     pool=pool)

    as this way they will not need to be pickled and unpickled every function
    call.
    
    Note though that if you specify logl_args, and ptform_args when  creating
    the Pool *AND* in the sampler those will be concatenated
    """

    def __init__(self,
                 njobs,
                 loglike,
                 prior_transform,
                 logl_args=None,
                 logl_kwargs=None,
                 ptform_args=None,
                 ptform_kwargs=None):
        self.logl_args = logl_args
        self.logl_kwargs = logl_kwargs
        self.ptform_args = ptform_args
        self.ptform_kwargs = ptform_kwargs
        self.njobs = njobs
        self.loglike_0 = loglike
        self.prior_transform_0 = prior_transform
        self.loglike = loglike_cache
        self.prior_transform = prior_transform_cache
        self.pool = None

    def __enter__(self):
        """
        Activate the pool
        """
        initargs = (self.loglike_0, self.prior_transform_0, self.logl_args
                    or (), self.logl_kwargs or {}, self.ptform_args
                    or (), self.ptform_kwargs or {})
        self.pool = mp.Pool(self.njobs, initializer, initargs)
        initializer(*initargs)
        # running this in the master process seems to help with
        # restoration of the sampler ( #403)
        return self

    def map(self, F, x):
        """ Apply the function F to the list x

        Parameters
        ==========

        F: function
        x: iterable
        """
        return self.pool.map(F, x)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.pool.terminate()
        except:  # noqa
            pass
        try:
            del (FunctionCache.loglike, FunctionCache.prior_transform,
                 FunctionCache.logl_args, FunctionCache.logl_kwargs,
                 FunctionCache.ptform_args, FunctionCache.ptform_kwargs)
        except:  # noqa
            pass

    @property
    def size(self):
        """
        Return the number of processes in the pool
        """
        return self.njobs

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()
