============
Crash Course
============

``dynesty`` requires three basic ingredients to sample from a given
distribution:

* the likelihood (via a :func:`loglikelihood` function),

* the prior (via a :func:`prior_transform` function that transforms samples
  from the unit cube to the target prior), and

* the dimensionality of the parameter space.

As an example, let's define our likelihood to be a 3-D correlated multivariate
Normal (Gaussian) distribution and our prior to be uniform in each dimension
from [-10, 10)::

    import numpy as np

    # Define the dimensionality of our problem.
    ndim = 3

    # Define our 3-D correlated multivariate normal likelihood.
    C = np.identity(ndim)  # set covariance to identity matrix
    C[C==0] = 0.95  # set off-diagonal terms
    Cinv = linalg.inv(C)  # define the inverse (i.e. the precision matrix)
    lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
                    np.log(np.linalg.det(C)))  # ln(normalization)

    def loglike(x):
        """The log-likelihood function."""

        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

    # Define our uniform prior.
    def ptform(u):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior within [-10., 10.) for each variable."""

        return 10. * (2. * u - 1.)

Estimating the evidence and posterior is as simple as::

    import dynesty

    # "Standard" nested sampling.
    sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    sampler.run_nested()
    results = sampler.results

    # "Dynamic" nested sampling.
    dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim)
    dsampler.run_nested()
    dresults = dsampler.results

We can visualize our results using several of the built-in plotting utilities.
For instance::

    from dynesty import plotting as dyplot

    # Plot a summary of the run.
    rfig, raxes = dyplot.runplot(results)

    # Plot traces and 1-D marginalized posteriors.
    tfig, taxes = dyplot.traceplot(results)

    # Plot the 2-D marginalized posteriors.
    cfig, caxes = dyplot.cornerplot(results)
