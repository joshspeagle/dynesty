======
Nestle
======

/ˈnesəl/ *(rhymes with "wrestle")*

.. image:: _images/sphx_glr_plot_ellipsoids_001.png
   :width: 280px
   :align: right
   :target: examples/plot_ellipsoids.html

Pure Python, MIT-licensed implementation of nested sampling algorithms.

`Nested Sampling
<https://en.wikipedia.org/wiki/Nested_sampling_algorithm>`_ is a
computational approach for integrating posterior probability in order
to compare models in Bayesian statistics. It is similar to Markov
Chain Monte Carlo (MCMC) in that it generates samples that can be used
to estimate the posterior probability distribution. Unlike MCMC, the
nature of the sampling also allows one to calculate the integral of
the distribution. It also happens to be a pretty good method for robustly
finding global maxima.

Install
=======

Nestle is compatible with both Python and Legacy Python (formerly known as Python 2). It requires numpy 1.6+ and, optionally, scipy. Install with pip::

    pip install nestle

For the latest development version, see http://github.com/kbarbary/nestle.


Getting started
===============

The following is a simple but complete example of using Nestle to
sample the parameters of a line given three data points with
uncertainties::

    import numpy as np
    import nestle

    data_x = np.array([1., 2., 3.])
    data_y = np.array([1.4, 1.7, 4.1])
    data_yerr = np.array([0.2, 0.15, 0.2])

    # Define a likelihood function
    def loglike(theta):
        y = theta[1] * data_x + theta[0]
        chisq = np.sum(((data_y - y) / data_yerr)**2)
        return -chisq / 2.

    # Define a function mapping the unit cube to the prior space.
    # This function defines a flat prior in [-5., 5.) in both dimensions.
    def prior_transform(x):
        return 10.0 * x - 5.0

    # Run nested sampling.
    result = nestle.sample(loglike, prior_trasform, 2)

    result.logz     # log evidence
    result.logzerr  # numerical (sampling) error on logz
    result.samples  # array of sample parameters
    result.weights  # array of weights associated with each sample


- See :doc:`examples/index` for more detailed examples.
- See :doc:`prior` for more information on the ``prior_tranform`` function.
- See :doc:`stopping` for discussion on when the algorithm terminates.
- See the :doc:`api` page for detailed API documentation.


Available methods
=================

The trick in nested sampling is to, at each step in the algorithm,
*efficiently* choose a new point in parameter space drawn with
*uniform probability* from the parameter space with likelihood greater
than the current likelihood constraint. The different methods all use
the current set of active points as an indicator of where the target
parameter space lies, but differ in how they select new points from
it.

MCMC exploration (`method='classic'`)
-------------------------------------

This is close to the method described in Skilling (2004). 

Single ellipsoid (`method='single'`)
------------------------------------

This is the method described by Mukherjee, Parkinson & Liddle (2006).
Determines a single ellipsoid that bounds all active points, enlarges
the ellipsoid by a user-settable factor, and selects a new
point at random from within the ellipsoid. The enlargement factor is designed to
ensure that the high-likelihood region is completely enclosed in the ellipsoid.

Multiple ellipsoids (`method='multi'`)
--------------------------------------

This is the method first described in Shaw, Bridges & Hobson (2007) and
implemented in the MultiNest software (Feroz, Hobson & Bridges 2009).

In cases where the posterior is multi-modal, the single-ellipsoid
method can be extremely inefficient: In such situations, there are
clusters of active points on separate high-likelihood regions
separated by regions of lower likelihood.  Bounding all points in a
single ellipsoid means that the ellipsoid includes the
lower-likelihood regions we wish to avoid sampling from.

The solution is to detect these clusters and bound them in separate
ellipsoids.  For this, we use a recursive process where we perform
K-means clustering with K=2. If the resulting two ellipsoids have a
significantly lower total volume than the parent ellipsoid (less than
half), we accept the split and repeat the clustering and volume test
on each of the two subset of points. This process continues
recursively. Alternatively, if the total ellipse volume is
significantly greater than expected (based on the expected density of
points) this indicates that there may be more than two clusters and
that K=2 was not an appropriate cluster division. We therefore still
try to subdivide the clusters recursively. However, we still only
accept the final split into N clusters if the total volume decrease is
significant.


References
==========

Feroz, Hobson & Bridges 2009. MultiNest: an efficient and robust
Bayesian inference tool for cosmology and particle physics. *MNRAS*,
**398**, `1601 <http://adsabs.harvard.edu/abs/2009MNRAS.398.1601F>`_.

Mukherjee, Parkinson & Liddle 2006. A Nested sampling algorithm for
cosmological model selection. *ApJ*, **638**, `L51
<http://adsabs.harvard.edu/abs/2006ApJ...638L..51M>`_.

Shaw, Bridges & Hobson 2007. Efficient Bayesian inference for
multimodal problems in cosmology. *MNRAS*, **378**, `1365
<http://adsabs.harvard.edu/abs/2007MNRAS.378.1365S>`_.

Silvia & Skilling 2006. Data Analysis: A Bayesian Tutorial, 2nd
Edition. Oxford University Press.

Skilling, J. 2004. Nested Sampling. In *Maximum entropy and Bayesian
methods in science and engineering* (ed. G. Erickson, J.T. Rychert,
C.R. Smith). *AIP Conf. Proc.*, **735**,
`395 <http://adsabs.harvard.edu/abs/2004AIPC..735..395S>`_.

See also http://www.inference.phy.cam.ac.uk/bayesys/.


Citation
========

If you use Nestle in your work, please cite the github repository and the
relevant references listed above.


.. toctree::
   :hidden:
   :maxdepth: 2

   examples/index
   prior
   stopping
   api
   devdocs
