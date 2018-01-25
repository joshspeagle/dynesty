===============
Getting Started
===============

Nested Sampling with dynesty
============================

To give a concrete example, let's return to the simple 3-D multivariate normal
likelihood and uniform prior from [-10, 10) used in :ref:`Crash Course` to
define the :meth:`loglikelihood` and :meth:`prior_transform` functions::

    import numpy as np

    # Define the dimensionality of our problem.
    ndim = 3

    # Define our 3-D correlated multivariate normal log-likelihood.
    C = np.identity(ndim)
    C[C==0] = 0.95
    Cinv = linalg.inv(C)
    lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
                    np.log(np.linalg.det(C)))

    def loglike(x):
        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

    # Define our uniform prior via the prior transform.
    def ptform(u):
        return 20. * u - 10.

Initialization
--------------

Nested Sampling in `dynesty` is done via a particular `sampler`
object that is initialized from the :ref:`Top-Level Interface`. To start,
let's use :meth:`~dynesty.dynesty.NestedSampler` to initialize a particular
sampler from `~dynesty.nestedsamplers`. There are only 3 required arguments: 
a log-likelihood function (`loglike`), a prior transform function (`ptform`),
and the number of dimensions taken by the loglikelihood (`ndim`). 

Using the functions above, we can initialize our sampler using::

    from dynesty import NestedSampler

    # initialize our nested sampler
    sampler = NestedSampler(loglike, ptform, ndim)

See :ref:`Top-Level Interface` for more details on the API, :ref:`Examples` 
for more examples of usage, and :ref:`FAQ` for
some additional advice. Here we'll go over just the basics.

Live Points
-----------

Similar to ensemble sampling methods such as 
`emcee <http://dan.iel.fm/emcee/current/>`_, the behavior of Nested Sampling
can also be sensitive to the number of live points used. Increasing the number
of live points leads to smaller changes in the prior volume :math:`\ln X` over
time. This improves the effective resolution while simultaneously increasing
the runtime.

In addition, the number of live points can also affect the stability of our
:ref:`Bounding Options`. By default, ``dynesty`` inflates the size of the
chosen bounds by an enlargement factor to ensure they effectively bound the
iso-likelihood contours. These bounds become more robust the more live points
are used, leading to more efficient proposals.

The number of live points can be specified upon initialization via the 
`nlive` argument. For example, if we want to run with 1000 live points rather
than the default 250, we would use::

    NestedSampler(loglike, ptform, ndim, nlive=1000)

Bounding Options
----------------

``dynesty`` supports a number of options for bounding the target distribution:

* **no bound** (`'none'`), i.e. sampling from the entire unit cube,

* a **single bounding ellipsoid** (`'single'`),

* **multiple** (possibly overlapping) **bounding ellipsoids** (`'multi'`),

* **overlapping balls** centered on each live point (`'balls'`), and

* **overlapping cubes** centered on each live point (`'cubes'`).

By default, ``dynesty`` uses multi-ellipsoidal decomposition (`'multi'`),
which often is flexible enough to capture the complexity of many likelihood
distributions while simple enough to quickly and efficiently generate new
samples. For more complex distributions, overlapping balls (`'balls'`)
or cubes (`'cubes'`) can generate more flexible bounding distributions but
come with significantly more overhead that can be less efficient at generating
samples. For simpler distributions, a single ellipsoid (`'single'`) is often
sufficient. Sampling directly from the unit cube (`'none'`) is extremely
inefficient but is a useful option to verify your results and
look for possible biases. It otherwise should only be used if the
log-likelihood is trivial to compute.

Specifying the particular bounding distribution can be done upon initialization
via the `bound` argument. If we wanted to sample using overlapping balls rather
than multiple bounding ellipsoids, for instance, we would use::

    NestedSampler(loglike, ptform, ndim, nlive=1000, bound='balls')

As mentioned in :ref:`Live Points`, bounding distributions in ``dynesty`` are
enlarged in an attempt to conservatively encompass the iso-likelihood contour
associated with each dead point. By default, this is done in real-time using
bootstrapping methods. This procedure can lead to some instability in the size
of the bounds if only a few number of live points are used 
(see the :ref:`FAQ`). In addition, the *volumes* of all bounding objects can
also be enlarged by a constant factor (applied after any bootstrapping). The
number of bootstrap realizations used and the volume enlargement factor can be
specified using the `bootstrap` and `enlarge` arguments. 

For instance, if we want to use 50 bootstraps to determine expansion factors
with an additional fixed volume enlargement factor of 1.10, we would specify::
    
    NestedSampler(loglike, ptform, ndim, nlive=1000, bound='balls',
                  bootstrap=50, enlarge=1.10)

Additional information on the bounding objects can be found under
:ref:`Bounding` and in :ref:`Examples`.

To avoid excessive overhead spent constructing bounding
distributions, ``dynesty`` only updates bounding distributions after a fixed
number of likelihood calls specified by the `update_interval` argument. Larger
values generally decrease the sampling efficiency but can improve overall
performance. This value by default is set to be `round(0.6 * nlive)`, but if
we wanted to instead use a larger value we could just specify that via::

    NestedSampler(loglike, ptform, ndim, nlive=1000, bound='balls',
                  bootstrap=50, enlarge=1.10, update_interval=1.2)

We could also pass an integer if we'd like to specify the number of function
calls directly::

    NestedSampler(loglike, ptform, ndim, nlive=1000, bound='balls',
                  bootstrap=50, enlarge=1.10, update_interval=600)

Finally, ``dynesty`` tries to avoid constructing bounding distributions too
early in the run to avoid issues where the bounds can significantly exceed the
unit cube. For instance, the bounding distribution of the initial set of points
*by construction* exceeds the bounds of the unit cube. This can lead to a 
variety of problems associated with each method, especially in higher
dimensions (since volume scales as :math:`\propto r^D`).

To avoid this behavior, ``dynesty`` deliberately delays the first bounding
update until some heuristics are satisfied. If we wanted to adjust this
behavior, such as disabling the delay altogether,
we could do so by passing some parameters using the `first_update`
argument::

    NestedSampler(loglike, ptform, ndim, nlive=1000, bound='balls',
                  bootstrap=50, enlarge=1.10, update_interval=600,
                  first_update={'min_ncall': 0, 'min_eff': 100.})

See :ref:`Top-Level Interface` for more information.

Sampling Options
----------------

`dynesty` also supports several different sampling methods *conditioned on*
the provided bounds which can be passed via the `sample` argument:

* **uniform** sampling (`'unif'`),

* **random walks** away from a current live point (`'rwalk'`), and

* **slice sampling** away from a current live point (`'slice'`).

By default, `dynesty` samples uniformly within the bounding distribution
(`'unif'`). In low dimensions (:math:`D \lesssim 10`), uniform sampling is in
general quite efficient at generating new live point proposals **assuming the
bounding distributions are relatively stable**. In moderate dimensions
(:math:`D \sim 5-20`), random walks (`'rwalk'`) often can become similarly
efficient. In moderate-to-high dimensions (:math:`D \gtrsim 20`), sampling
techniques that do not rely on rejecting new proposed points such as
multivariate slice sampling (`'slice'`) become progressively more efficient.

One benefit to using random walks or slice sampling is that they require many
fewer live points to adapt to structure in higher dimensions (since they only
sample *conditioned* on the bounds). They also do not require bootstrap-style
corrections since they contain built-in methods to tune their step sizes.

Following the example above, let's say we wanted to combine the flexibility of
multiple bounding ellipsoids and slice sampling. Since slice sampling is
less efficient than uniform sampling, we would also like to increase the
relevant `update_interval` as well to avoid constructing new bounding
distributions too frequently. This might look something like:: 

    NestedSampler(loglike, ptform, ndim, bound='multi', sample='slice',
                  update_interval=float(ndim))

Running  Internally
-------------------

Sampling from our target distribution can be done using the
:meth:`~dynesty.sampler.Sampler.run_nested` function in the provided
`sampler`:: 

    sampler.run_nested()

Sampling will continue until specified stopping criteria are reached, and
the current state of the sampler is by default output to `~sys.stderr` in
real time. The stopping criteria can be any combination of:

* a fixed number of iterations (`maxiter`),

* a fixed number of likelihood calls (`maxcall`), and

* a specified :math:`\Delta \ln \hat{\mathcal{Z}}_i` tolerance (`dlogz`).

For instance, the code above would produce output like:

.. rst-class:: sphx-glr-script-out

Out::

    iter: 6718+1000 | bound: 9 | nc: 1 | ncall: 39582 | eff(%): 19.499 | 
    logz: -8.832 +/-  0.132 | dlogz:  0.006 <  5.005    

From left to right, this records: the current iteration (plus the number of
live points added after stopping), the current bound being used, the number
of likelihood calls made before accepting the last sample, the total number
of likelihood calls, the overall sampling efficiency, the current estimated
evidence, and the remaining `dlogz` (plus the stopping criterion).

By default, the stopping criteria are optimized for evidence estimation, with
posteriors treated as a nice byproduct. This works by scaling `dlogz` based on
the provided number of live points to try and avoid spending excessive amounts
of time sampling over the bulk of the posterior mass. If we were much more
interested in having more robust posterior estimates, or in
setting bounds for the amount of samples and function calls allows, we could
override this behavior. This would look something like::

    sampler.run_nested(dlogz=0.01, maxiter=15000, maxcall=50000)

Since sampling is done through the `sampler` objects, users can also continue
to add new samples based on where they left off. This is as easy as::

    sampler = NestedSampler(loglike, ptform, ndim, nlive=1000)

    sampler.run_nested()  # first run
    res1 = sampler.results

    sampler.run_nested(maxcall=10000)  # (possibly) adding more samples
    sampler.run_nested(dlogz=0.01)  # (possibly) adding more samples
    res2 = sampler.results

Running Externally
------------------

Similar to `emcee <http://dan.iel.fm/emcee/current/>`_, `sampler` objects in
``dynesty`` can also be run externally as a **generator** via the
:meth:`~dynesty.sampler.Sampler.sample` function. This might look something
like::

    # The main nested sampling loop.
    for it, results in enumerate(sampler.sample(dlogz=0.5)):
        pass

    # Adding the final set of live points.
    for it_final, results in enumerate(sampler.add_live_points()):
        pass

Results
=======

Sampling results can be accessed through the `~dynesty.sampler.Sampler.results`
property and are returned as a (modified) dictionary::

    results = sampler.results

We can print a quick summary of the run using
:meth:`~dynesty.results.Results.summary`, which provides basic information
about the evidence estimates and overall sampling efficiency::

    # Print out a summary of the results.
    res1.summary()
    res2.summary()

.. rst-class:: sphx-glr-script-out

Out::

    Summary
    =======
    nlive: 1000
    niter: 6718
    ncall: 39582
    eff(%): 19.499
    logz: -8.832 +/-  0.132

    Summary
    =======
    nlive: 1000
    niter: 13139
    ncall: 49499
    eff(%): 28.564
    logz: -8.818 +/-  0.084

Quick Rundown
-------------

While a number of quantities are contained in the `Results` instance,
the relevant quantities for most users will be the collection
of samples from the run (`samples`), their corresponding (unnormalized) 
log-weights (`logwt`), the cumulative log-evidence (`logz`), and the
approximate error on the evidence (`logzerr`). The remaining quantities are
used to help visualize the output (see :ref:`Visualizing Results`) and might
also be useful for more advanced users who want additional information about
the nested sampling run.

Full Summary
------------

As a dictionary, the full set of quantities provided in `Results` can be
accessed using :func:`keys`. A description of the full set of quantities
included in `Results` are listed below:

* `nlive`: the number of live points used in the run,

* `niter`: the number of iterations (samples),

* `ncall`: the total number of function calls,

* `eff`: the overall sampling efficiency,

* `samples`: the set of samples in the *native parameter space*,

* `samples_u`: the set of samples in the *unit cube*,

* `samples_id`: the unique particle index associated with each sample,

* `samples_it`: the iteration the sample was *originally* proposed,

* `logwt`: the log-weight (unnormalized) associated with each sample,

* `logl`: the log-likelihood associated with each sample,

* `logvol`: the (expected) ln(prior volume) associated with each sample,

* `logz`: the cumulative evidence at each iteration (sample),

* `logzerr`: the estimated error (standard deviation) on `logz`, and

* `information`: the estimated "information" (see :ref:`Role of Priors in
  Nested Sampling`) at each iteration (sample).

If the bounding distributions are also saved (the default behavior), then the
following quantities are also provided:

* `bound`: a (deep) copy of the set of bounding objects,

* `bound_iter`: the index of the bounding object active at a given iteration,

* `samples_bound`: the index of the bounding object the sample was *originally
  proposed from*, and

* `scale`: the scale-factor used at a given iteration (used to scale the bounds
  for non-uniform proposals).

Note that some of these quantities change when using :ref:`Dynamic Nested
Sampling`.

Visualizing Results
===================

Assuming we've completed a run and stored the resulting `res1` and `res2`
`~dynesty.results.Results` dictionaries as defined above, we can compare what
their relative weights by comparing them directly using some simple code like::

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # initialize 3-D plots of position and likelihood, colored by weight
    fig = plt.figure(figsize=(30, 10))

    # plotting the initial run
    ax = fig.add_subplot(121, projection='3d')
    p = ax.scatter(res1.samples[:, 0], res1.samples[:, 1], res1.samples[:, 2],
                   marker='o', c=np.exp(res1.logwt) * 1e7, linewidths=(0.,),
                   cmap='coolwarm')

    # plotting the extended run
    ax = fig.add_subplot(122, projection='3d')
    p = ax.scatter(res2.samples[:, 0], res2.samples[:, 1], res2.samples[:, 2],
                   marker='o', c=np.exp(res2.logwt) * 1e8, linewidths=(0.,),
                   cmap='coolwarm')

.. image:: ../images/quickstart_001.png
    :align: center

In the initial run (`res1`), we see that the majority of the importance weight
:math:`\hat{p}_i` is concentrated near the mode; in the extended run, however,
it is instead concentrated in a ring around the mode. This behavior represents
the fundamental compromise between the likelihood :math:`\mathcal{L}_i` and the
change in prior volume :math:`\Delta X_i`. The stark difference in the
distribution of weights between the two samples is driven entirely by
differences in :math:`\Delta X_i`. In the extended run (`res2`), the
distribution of weights directly follows the shape expected from the "typical
set" (see :ref:`Typical Sets` for additional discussion).

By contrast, since the final set of live points after :math:`N` samples are
uniformly sampled within :math:`X_{i=N}`, the expected change in the prior volume
is *constant*. This leads to *linear* (rather than exponential) compression of
the remaining prior volume, where the weight assigned to the 
live point with the :math:`k`-th lowest likelihood is then
:math:`\propto f(\mathcal{L}_{N+k}) \, X_N`. In the case where there is a
significant portion of prior volume remaining (as with `res1`), this leads to
extremely rapid traversal of the remaining prior volume and hence large 
importance weights.

dyplot
------

To avoid introducing an excessive burden on typical users, ``dynesty`` comes
with a variety of built-in plotting utilities in the :mod:`~dynesty.plotting`
module. These include a variety of generic summary plots as well as ways of
visualizing bounding distributions throughout the course of a run. We can
import them using::

    from dynesty import plotting as dyplot

The `dyplot` alias will be used for convenient shorthand throughout the
remainded of the documentation. While some basic usage will be demonstrated
below, please see the :ref:`API` for additional details.

Summary Plots
-------------

One of the most direct ways of visualizing how Nested Sampling computes
the *evidence* is by examining the relationship between the prior volume
:math:`\ln X_i` and:

#. the (effective) iteration :math:`i`, which illustrates how quickly/slowly
   our samples are compressing the prior volume,

#. the likelihood :math:`\mathcal{L}_i`, to see how smoothly we sample "up" the
   likelihood distribution to the `maximum likelihood (ML) estimate
   <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>`_,

#. the importance weight :math:`\hat{p}_i`, showcasing where the bulk of the
   **posterior mass** is located (i.e. the typical set), and 

#. the evidence :math:`\hat{\mathcal{Z}}_i`, to see where most of the contribution
   to the evidence (and its respective errors) are coming from.

A **summary plot** showcasing these features can be generated using
:func:`~dynesty.plotting.runplot`. As an example, a summary plot for `res2`
comparing it to the actual analytic :math:`\ln \mathcal{Z}` evidence solution
can be generated using::

    lnz_truth = ndim * -np.log(2 * 10.)  # analytic evidence solution
    fig, axes = dyplot.runplot(res2, lnz_truth=lnz_truth)  # summary (run) plot

.. image:: ../images/quickstart_002.png
    :align: center

We see that up until we recycle our final set of live points (see 
:ref:`Basic Algorithm`), as indicated by the dashed lines, the relationship
between :math:`\ln X_i` and :math:`i` is linear (i.e. prior volume compression
is exponential). Afterwards, however, it flattens out, rapidly traversing the
remaining prior volume in linear fashion. Comparing the general shape of the
likelihood and importance weights subplots also highlight how the typical set
is as much a function of :math:`\Delta X_i` as :math:`\mathcal{L}_i`: although
contributions initially increase as the likelihood increases, they quickly fall
as the ML region encompasses increasingly smaller effective volumes.

Trace Plots
-----------

Another common way to visualize the results of many sampling algorithms is to
generate a **trace plot** showing the evolution of particles (and their
marginal posterior distributions) in 1-D projections. This can be done using
the :meth:`~dynesty.plotting.traceplot` function, which plots a combination
of particle positions as a function of :math:`\ln X` (colored by importance
weight) and the corresponding 1-D marginalized posterior::

    fig, axes = dyplot.traceplot(res2, truths=np.zeros(ndim), 
                                 truth_color='black', show_titles=True,
                                 trace_cmap='viridis', connect=True,
                                 connect_highlight=range(5))

.. image:: ../images/quickstart_003.png
    :align: center

By default, :meth:`~dynesty.plotting.traceplot` returns the samples color-coded
by their relative weight and the 1-D marginalized posteriors smoothed by a
Normal (Gaussian) kernel with a standard deviation set to ~2% of the provided
range (which defaults to the 5-sigma bounds computed from the set of weighted
samples). It also can overplot input truth vectors as well as highlight
specific particle paths (shown above) to inspect the behavior of individual
particles. These can be useful to qualitatively identify problematic behavior.
For instance, while the particle paths shown above support the assumption
that our samples are i.i.d. within the likelihood constraints 
at a particular iteration.

Corner Plots
------------

In addition to trace plots, another common way to visualize (weighted) samples
is using **corner plots** (also called "triangle plots"), which show a
combination of 1-D and 2-D marginalized posteriors. ``dynesty`` supports
several corner plotting functions. The most straightforward is
:meth:`~dynesty.plotting.cornerpoints`, which simply plots the sample positions
colored according to their importance weights::

    # initialize figure
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.reshape((2, 5))  # reshape axes

    # add white space
    [a.set_frame_on(False) for a in axes[:, 2]]
    [a.set_xticks([]) for a in axes[:, 2]]
    [a.set_yticks([]) for a in axes[:, 2]]

    # plot initial run (res1; left)
    fg, ax = dyplot.cornerpoints(res1, cmap='plasma', truths=np.zeros(ndim),
                                 fig=(fig, axes[:, :2]))

    # plot extended run (res2; right)
    fg, ax = dyplot.cornerpoints(res2, cmap='viridis', truths=np.zeros(ndim),
                                 fig=(fig, axes[:, 3:]))

.. image:: ../images/quickstart_004.png
    :align: center

Just by looking at our projected samples, it is apparent that the results from
the extended run `res2` does a much better job of localizing the overall
distribution compared to `res1`. We can get a better qualitative and
quantitative handle on this by plotting the marginal 1-D and 2-D posterior
density estimates using :meth:`~dynesty.plotting.cornerplot` as::


    # initialize figure
    fig, axes = plt.subplots(3, 7, figsize=(35, 15))
    axes = axes.reshape((3, 7))  # reshape axes

    # add white space
    [a.set_frame_on(False) for a in axes[:, 3]]
    [a.set_xticks([]) for a in axes[:, 3]]
    [a.set_yticks([]) for a in axes[:, 3]]

    # plot initial run (res1; left)
    fg, ax = dyplot.cornerplot(res1, color='blue', truths=np.zeros(ndim),
                               truth_color='black', show_titles=True,
                               max_n_ticks=3, quantiles=None,
                               fig=(fig, axes[:, :3]))

    # plot extended run (res2; right)
    fg, ax = dyplot.cornerplot(res2, color='red', truths=np.zeros(ndim), 
                               truth_color='black', show_titles=True,
                               quantiles=None, max_n_ticks=3,
                               fig=(fig, axes[:, 4:]))

.. image:: ../images/quickstart_005.png
    :align: center

Similar to :meth:`~dynesty.plotting.runplot`, the marginal distributions shown
are by default smoothed by 2% in the specified range using a Normal (Gaussian)
kernel. Notice that even though our original run `res1` gave 
similar evidence estimates to the extended run `res2`, it gives significantly
more "noisy" estimates of the posterior.

Bounding Distribution Plots
---------------------------

To visualize how we're sampling in nested "shells", we can look at the
evolution of our bounding distributions in a given 2-D projection over the
course of a run. The :meth:`~dynesty.plotting.boundplot` function allows us to
look at the bounding distributions from two different perspectives: the
bounding distribution used when proposing new live points at a specific
iteration (specified using `it`), or the bounding distribution that a given
dead point originated from (specified using `idx`). While
:meth:`~dynesty.plotting.boundplot` natively plots in the space of the unit
cube, if a specified :meth:`prior_transform` is passed all samples are instead
converted to the original (native) model space.

Using :meth:`~dynesty.plotting.boundplot`, we can examine the evolution of the
bounding distributions over a given run via::

    # initialize figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # plot several snapshots over the course of the run
    for i, a in enumerate(axes.flatten()):
        it = int((i+1) * res2.niter / 8.)
        # overplot the result onto each subplot
        temp = dyplot.bountplot(res2, dims=(0, 1), it=it, 
                                prior_transform=prior_transform,
                                show_live=True, max_n_ticks=3,
                                span=[(-10, 10), (-10, 10)], fig=(fig, a))
        a.set_title('Iteration {0}'.format(it))

.. image:: ../images/quickstart_006.png
    :align: center

The figure illustrates that we first begin sampling directly from the unit
cube. After the conditions in `first_update` are satisfied, we then switch over
to the default multi-ellipsoidal bounding distributions. We see that these are
able to adapt well to the target distribution over time, ensuring we continue
to sample efficiently. We can also see the impact of bootstrapping
on the bounding ellipsoids since they always remain slightly larger than the
set of live points. While it slightly decreases the overall sampling
efficiency, this shows how the procedure helps to ensure no likelihood is
"left out" during the course of the Nested Sampling run.

Alternately, we can generate a corner plot of the bounding distribution using
:meth:`~dynesty.plotting.cornerbound` via::

    fig, axes = dyplot.cornerprop(res2, it=5000, 
                                  prior_transform=prior_transform,
                                  show_live=True, 
                                  span=[(-10, 10), (-10, 10)])

.. image:: ../images/quickstart_007.png
    :align: center

Basic Post-Processing
=====================

In addition to plotting, ``dynesty`` also contains some post-processing
utilities in the :mod:`~dynesty.utils` module. In many cases, a rough 
approximation of the posterior using the first two moments (mean and
covariance) can be useful. These can be computed from the set of (weighted) 
samples using the :meth:`~dynesty.utils.mean_and_cov` function::

    from dynesty import utils as dyfunc

    samples, weights = res2.samples, np.exp(res2.logwt - res2.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)

Runs can also be resampled to give a mew set of points with equal
weights, similar to MCMC methods, using the 
:meth:`~dynesty.utils.resample_equal` function::

    new_samples = dyfunc.resample_equal(samples, weights)

See :ref:`Nested Sampling Errors` for some additional discussion and 
demonstration of more functions.
