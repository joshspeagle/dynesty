The Prior Transform
===================

The `nestle.sample` function takes two user-defined functions as
inputs. The first function should give the natural logarithm of the
likelihood function of interest. The second function is a "prior
transform." *What's up with that?*

The prior transform function is used to specify the Bayesian prior for
the problem, in a round-about way. It is a transformation from a space
where variables are independently and uniformly distributed between 0
and 1 to the parameter space of interest. For independent parameters,
this would be the product of the `inverse cumulative distribution
function <https://en.wikipedia.org/wiki/Quantile_function>`_ (also
known as the *percent point function* or *quantile function*) for each
parameter.

Uniform priors
--------------

This is easier to illustrate with a concrete example.  Suppose we
wanted a uniform prior in the range [-5, 5.) in all variables:

.. math::

   p(x) \propto \left\{
                \begin{array}{ll}
                  1 \quad -5 \le x < 5\\
                  0 \quad {\rm otherwise}
                \end{array}
              \right.

The prior transform would be::

    def prior_transform(x):
        return 10. * x - 5.

because if ``x`` is an array of numbers between 0 and 1, the result
will be an array of numbers between -5 and 5.

For a slightly more complex example, suppose we have problem with two
variables where the prior on the first variable is a uniform
distribution between -1 and 1 and the prior on the second variable is
a uniform distribution between -10 and 10. The prior transform would
be::

    def prior_transform(x):
        return np.array([2. * x[0] - 1., 20. * x[1] - 10.])

or equivalently::

    def prior_transform(x):
        return np.array([2., 20.]) * x + np.array([-1., 10.])

Incidentally, the simplest possible prior to define would be a uniform
distribution between 0 and 1 in all parameters, for which the
transform would be ``lambda x: x``.

Non-uniform priors
------------------

Suppose we wish to specify a normal distribution for our prior, in two
dimensions. We can use the inverse cumulative distribution function
for the normal distribution, `scipy.special.ndtri`::

    from scipy.special import ndtri

    prior_transform = ndtri

This specifies a normal distribution with mean 0 and standard
deviation 1 in all dimensions. To specify a different mean and
standard deviation, simply transform the output::

   def prior_transform(x):
       return mu + sigma * ndtri(x)

where ``mu`` is an array giving the mean in each dimension and
``sigma`` is an array giving the standard deviation in each dimension.

The distributions in `scipy.stats` (particularly the ``ppf`` method)
might be particuarly useful in constructing more complex priors.


Why?
----

**Why not combine the prior and likelihood in a single function (giving
the posterior) as in, e.g., emcee?**

Unlike traditional MCMC, nested sampling starts by randomly sampling
from the entire parameter space. This is not possible without
specifying some sort of contraints on the parameters. Thus the user
would still have to specify bounds on all the parameters. You could
use `nestle.sample` this way: pass a "loglikelihood" function that in
fact returns the log of the posterior, and then specify a prior
transform that is uniform in some parameter range (as illustrated in
the first example). This will give valid samples. However, the
evidence will be affected by the range of the prior. A wider uniform
prior decreases the weight of the high-likelihood regions in the
evidence integral, leading to a lower evidence.

There are a couple advantages to specifying the prior as a
transform rather than simply as bounds. First, it naturally allows
priors that extend to infinity but have finite integrals, such as the
normal distribution illustrated above. Second, assuming the prior
transform is computationally cheap, this lets us cheaply draw samples
according to the prior. We will (proportionally) avoid evaluating the
likelihood in regions we know to have small prior values. (This
preferential sampling is accounted for in the results.)
