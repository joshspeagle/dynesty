# Module containg various objects to be used as priors.
# These return the ln-prior-probability
import numpy as np
import scipy.stats

__all__ = ["prior_transform",
           "Prior", "TopHat", "Normal", "ClippedNormal",
           "LogNormal", "LogUniform", "Beta"]


def prior_transform(self, unit_coords, priors, prior_args=[]):
    """An example of one way to use the `Prior` objects below to go from unit
    cube to parameter space, for nested sampling.  This takes and returns a
    list instead of an array, to accomodate possible vector parameters.  Thus
    one will need something like ``theta_array=np.concatenate(*theta)``

    :param unit_coords:
        Coordinates on the unit prior hyper-cube.  Iterable.

    :param priors:
        A list of `Prior` objects, iterable of same length as `unit_coords`.

    :param prior_args: (optional)
        A list of dictionaries of prior function keyword arguments.

    :returns theta:
        A list of parameter values corresponding to the given coordinates on
        the prior unit hypercube.
    """
    theta = []
    for i, (u, p) in enumerate(zip(unit_coords, priors)):
        func = p.unit_transform
        try:
            kwargs = prior_args[i]
        except(IndexError):
            kwargs = {}
        theta.append(func(u, **kwargs))
    return theta


class Prior(object):
    """Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased. When called, the
    argument should be a variable and it should return the ln-prior-probability
    of that value.

    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """Constructor.

        :param parnames:
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names in case they are being fit for.
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update `params` values using alias.
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass

    def __len__(self):
        """The length is set by the maximum size of any of the prior_params.
        Note that the prior params must therefore be scalar of same length as
        the maximum size of any of the parameters.  This is not checked.
        """
        return max([np.size(self.params.get(k, 1)) for k in self.prior_params])

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        :param x:
            Value of the parameter, scalar or iterable of same length as the
            Prior object.

        :param kwargs: optional
            All extra keyword arguments are sued to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = self.distribution.pdf(x, *self.args,
                                  loc=self.loc, scale=self.scale)
        return np.log(p)

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(*self.args, size=len(self),
                                     loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def loc(self):
        """This should be overridden.
        """
        return 0

    @property
    def scale(self):
        """This should be overridden.
        """
        return 1

    @property
    def args(self):
        return []

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)


class TopHat(Prior):

    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.uniform

    @property
    def scale(self):
        return self.params['maxi'] - self.params['mini']

    @property
    def loc(self):
        return self.params['mini']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Normal(Prior):
    """A simple gaussian prior.
    """

    prior_params = ['mean', 'sigma']
    distribution = scipy.stats.norm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['sigma'],
                self.params['mean'] + self.params['sigma'])

    def bounds(self, **kwargs):
        # if len(kwargs) > 0:
        #    self.update(**kwargs)
        return (-np.inf, np.inf)


class ClippedNormal(Prior):
    """A Gaussian prior clipped to some range.
    """

    prior_params = ['mean', 'sigma', 'mini', 'maxi']
    distribution = scipy.stats.truncnorm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    @property
    def args(self):
        a = (self.params['mini'] - self.params['mean']) / self.params['sigma']
        b = (self.params['maxi'] - self.params['mean']) / self.params['sigma']
        return [a, b]

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogUniform(Prior):
    """Like log-normal, but the distribution of ln of the variable is
    distributed uniformly instead of normally.
    """

    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.reciprocal

    @property
    def args(self):
        a = self.params['mini']
        b = self.params['maxi']
        return [a, b]

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Beta(Prior):
    """A Beta distribution.
    """

    prior_params = ['mini', 'maxi', 'alpha', 'beta']
    distribution = scipy.stats.beta

    @property
    def scale(self):
        return self.params.get('maxi', 1) - self.params.get('mini', 0)

    @property
    def loc(self):
        return self.params.get('mini', 0)

    @property
    def args(self):
        a = self.params['alpha']
        b = self.params['beta']
        return [a, b]

    @property
    def range(self):
        return (self.params.get('mini', 0), self.params.get('maxi', 1))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogNormal(Prior):

    prior_params = ['mode', 'sigma']
    distribution = scipy.stats.lognorm

    @property
    def args(self):
        pass

    @property
    def scale(self):
        pass

    @property
    def loc(self):
        pass
