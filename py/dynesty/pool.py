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


def loglike_cache(x):
    """
    Likelihood function call
    """
    return FunctionCache.loglike(x, *FunctionCache.logl_args,
                                 **FunctionCache.logl_kwargs)


def prior_transform_cache(x):
    """
    Prior transform call
    """
    return FunctionCache.prior_transform(x, *FunctionCache.ptform_args,
                                         **FunctionCache.ptform_kwargs)


class Pool:
    """ The multiprocessing Pool wrapper class
    It is intended to be used as a context manager for dynesty sampler only.

    with dynesty.pool.Pool(16, like, prior_transform) as pool:
        dns = DynamicNestedSampler(pool.like, pool.prior_transform, ndim
                                     pool =pool)
    """

    def __init__(self,
                 njobs,
                 loglike,
                 prior_transform,
                 logl_args=None,
                 logl_kwargs=None,
                 ptform_args=None,
                 ptform_kwargs=None):
        """
        Initialized the Pool
        """
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
        """ Activate the pool """
        initargs = (self.loglike_0, self.prior_transform_0, self.logl_args
                    or (), self.logl_kwargs or {}, self.ptform_args
                    or (), self.ptform_kwargs or {})
        self.pool = mp.Pool(self.njobs, initializer, initargs)
        return self

    def map(self, F, x):
        """ Apply the mapping operation """
        return self.pool.map(F, x)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.pool.close()
        except:  # noqa
            pass
        try:
            self.pool.join()
        except:  # noqa
            pass

    @property
    def size(self):
        return self.njobs
