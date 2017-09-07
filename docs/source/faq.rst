===
FAQ
===

Bounding Questions
------------------

**Sometimes during a run a bound will be excessively large, requiring a large
number of log-likelihood evaluation before moving onwards. Why is this
happening? Is this a bug?**

This isn't a bug! It's Monte Carlo noise associated with the
bootstrapping process used when constructing new bounding distributions.
Depending on the chosen method, sometimes bounds can be unstable, leading
to large variations between bootstraps and subsequently large expansions
factors. If you'd instead like to take your chances by enlarging bounding 
objects manually, you can do so using the `enlarge` argument (and setting
`bootstrap=0`). Some of this is also discussed in the
:ref:`Gaussian Shells` and :ref:`Hyper-Pyramid` examples.

**During a run I sometimes see the bound index jump forward several places.
Is this normal?**

To avoid getting stuck sampling from bad bounding distributions (see above),
``dynesty`` automatically triggers a bounding update whenever the number of 
likelihood calls exceeds `update_interval` while sampling from a particular
bound. This can lead to multiple bounds being constructed before the sample
is accepted.

**How many bootstrap realizations do I need?**

Sec. 6.1 of `Buchner (2014) <https://arxiv.org/abs/1407.5459>`_ discusses
the basic behavior of bootstrapping and how many iterations are needed to
ensure that realizations do not include the same live point over some number
of realizations. By default, ``dynesty`` uses `bootstrap = 20`. This is more
aggressive than the `bootstrap = 50` recommended by Buchner (2014) but in 
general works reasonably well in practice.

**No matter what bounds, options, etc. I pick, the initial samples all
come from `bound = 0` and continue until the overall efficiency is quite low.
What's going on here?**

By default, ``dynesty`` opts to wait until some time has passed until
constructing the first bounding distribution and sampling conditioned on it.
This behavior is designed to avoid constructing overly large bounds that
significantly exceed the confines of the unit cube, which can lead to excessive
time spent generating random numbers. Prior to constructing the initial bound,
samples are proposed from the unit cube. The options that control these
heuristics can be modified using the `first_update` argument.

**What are the differences between** `'multi'` **and MultiNest?**

The multi-ellipsoid decomposition/bounding method implemented in ``dynesty``
is entirely based on the algorithm implemented in `nestle 
<http://kylebarbary.com/nestle/>`_ which itself is based on the algorithm
*described* in `Feroz, Hobson & Bridges (2009) 
<https://arxiv.org/abs/0809.3437>`_. As such, it doesn't include any
improvements, changes, etc. that may or may not be included in 
`MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_.

In addition, there are a few differences in the portion of the algorithm that
decides when to split an ellipsoid into multiple ellipsoids. As with
``nestle``, the implementation in ``dynesty`` is more conservative about
splitting ellipsoids to avoid over-constraining the remaining prior volume and
also enlarges all the resulting ellipsoids by a constant volume prefactor.
In general this results in a slightly lower sampling efficiency but greater
overall robustness.

These defaults can be changed through the :ref:`Top-Level Interface` via the
`enlarge`, `vol_dec` and `vol_check` keywords if you would like to experiment
with more conservative/aggressive behavior.

Live Point Questions
--------------------

**How many live points should I use?**

It depends. Increasing the number of live points helps establish more
flexible and robust bounds, improving the overall sampling efficiency and
prior volume resolution. However, it simultaneously increases the runtime.
These competing behaviors mean that compromises need to be made which are
problem-dependent.

In general, for ellipsoid-based bounds an absolute minimum of `ndim + 1`
live points is required, with `2 * ndim` being a (roughly) "safe" threshold.
If bootstraps are used to establish bounds while sampling uniformly, however,
many more live poits should be used. Around `25 * ndim` points are recommended
*for each expected mode* (when using `'multi'`).

Methods that do not depend on the absolute size of the bounds (but instead rely
on their shape) can use less live points. Their main restriction is
that new live point proposals (which "evolve" a copy of an existing live point
to a new position) must be independent of their starting point. Using too
few points can require excessive thinning, negating the benefit of using fewer
points. `5 * ndim` per mode seems to work reasonably well (see, e.g., the
:ref:`LogGamma` and :ref:`50-D Multivariate Normal` examples).

Sampling Questions
------------------

**Sampling is taking a long time. What should I do?!**

Unfortunately, there's no catch-all solution to this. The most important
first step is to make sure you're examining real-time outputs using the
`print_progress=True` option (enabled by default) if you're sampling internally
using :meth:`~dynesty.sampler.Sampler.run_nested` and printing out progress
if sampling externally using, e.g., :meth:`~dynesty.sampler.Sampler.sample`.

If the bounding distribution is updating frequently and you're using more
computationally intensive methods such as `'multi'`, some of this might be
due to excessive overhead associated with constructing the bounds. This can
be reduced by increasing `update_interval`.

If the overall sampling efficiency is low (*relative to what is expected*), it
might indicate that the distribution used (e.g., `'single'`) isn't effective
and more complex ones such as `'multi'` should be used instead. If you're
already using those but still getting inefficient proposals, that might
indicate that the bootstrapping updates (and/or enlargement factors) are 
unstable and giving excessively large bounds. You could either
use more live points or switch to an alternate sampling method less sensitive
to the size of the bounding distributions.

If sampling progresses efficiently after the first bounding update (i.e. when
`bound > 0`) for the majority of the run but becomes substantially less
efficient near the final `dlogz` stopping criterion, that could be a sign that
the the current set of live points are unable to give rise to bounding
distributions that are detailed enough to track the shape of the remaining
prior volume. As above, this behavior could be remedied by using more live
points or alternate sampling methods. Depending on the goal, the `dlogz` 
tolerance could also be adjusted.

Finally, if sampling seems to be progressing efficiently but is just
taking a long time, it might be because the high-likelihood regions of
parameter space are small compared to the prior volume. As discussed in 
:ref:`Role of Priors in Nested Sampling`, the time it takes to sample to a
given `dlogz` tolerance scales as the "information" gained by updating from
the prior to the posterior. Since Nested Sampling starts by sampling from the
entire prior volume, having overly-broad priors will increase the runtime.

**When using** `'balls'` **and/or** `'cubes'` **function calls take 
noticeably longer. What gives?**

Because those two methods model the bounding distribution as a *union* of
balls/cubes centered on each live point, there often are a huge number :math:`q`
of overlapping balls/cubes at any given point. Points proposed from an
individual ball/cube need to be accepted with probability :math:`1/q`, proposed
points both require frequent nearest neighbor searches and are rarely 
accepted. Although the implementation in ``dynesty`` already uses K-D trees via
`scipy.spatial.KDTree` to make this process quite efficient the overhead
associated with this process still remains substantial.

**I noticed that the number of iterations and/or function calls during a run
don't exactly match up with the limits I specify using, 
e.g.,** `maxiter` **or** `maxcall` **. Is this a bug?**

No, this is not a bug. When proposing a new point, ``dynesty`` currently only
checks the stopping criterion specified (whether iterations or function calls)
after that point has been accepted. This can also happen when using the 
`~dynesty.dynamicsampler.DynamicSampler` to propose a new batch of points,
since the first batch of points need to be allocated before checking the
stopping criterion.

**Why are** `'rwalk'` **and** `'slice'` **so inefficient?**

The main issue is that sampling in moderate and high-dimensional spaces is
inherently challenging due to the behavior of :ref:`Typical Sets`. Broadly
speaking, `'rwalk'` and `'slice'` are actually reasonably efficient when
compared to other (non-gradient) sampling methods on similar problems. 

In addition, it is also important to keep in mind that samples from ``dynesty``
are nominally independent (i.e. "pre-thinned"). For instance, for an MCMC
method with a sampling efficiency of 20% but requires thinning the resulting
chain by a factor of 10, the sampling efficiency is actually 2%.

**How many walks (steps) do you need to use for** `'rwalk'` **?**

In general, random walk behavior leads to excursions from the mean at a rate
that scales as (roughly) :math:`\sqrt{n} \sigma` where :math:`n` is the number
of walks and :math:`\sigma` is the typical length scale. The number of steps
needed then roughly scales as :math:`d^2`. In general this behavior doesn't
dominate unless sampling in high (:math:`d \gtrsim 20`) dimensions. In lower
dimensions (:math:`d \lesssim 10`), `walks=25` is often sufficient, while in
moderate dimensions (:math:`d \sim 10-20`) `walks=50` or greater are often
necessary to maintain independent samples.

**What are the differences between** `'slice'` **and PolyChord?**

Our implementation of multivariate slice sampling more closely follows the
prescription in `Neal (2003)
<https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461>`_ than the
algorithm outlined in the
`PolyChord <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_
paper. We conservatively enforce a strict Gibbs updating scheme that requires
sampling from *all* 1-D conditional distributions (in random order); we term
this entire update a "slice". This enables us to rigorously satisfy detailed
balance at the cost of being less efficient.

We also treat mode identification and sampling a little differently than
PolyChord. In ``dynesty`` our bounding objects are used to track modes as well
as a set of orthogonal basis vectors characterizing that mode. Slicing then 
takes place along that specific basis, allowing us to sample efficiently even in
a multi-modal context. For PolyChord, mode identification works using a
slightly different clustering algorithm and sampling takes place in a 
"pre-whitened" space based on the derived orthogonal basis.

**How many slices ("repeats") do you need to use for** `'slice'` **?**

Since slice sampling is a non-rejection for of sampling, the number of "slices"
requires for Nested Sampling in theory is independent of dimensionality
and can remain relatively constant. This is especially true if there are a set
of local principle axes that can be effectively captured by the bounding
distributions (e.g., `'multi'`). There are more pathological cases, however,
where the number of slices can weakly scale with dimensionality. In general
we find that the default `slices=3` is robust under a wide variety of
circumstances.

**The stopping criterion for Dynamic Nested Sampling is taking a long
time to evaluate. Is that normal?**

For large numbers of samples with a large number of varying live points, yes
this is normal. Every new particle increases the complexity of
simulating the errors used in the stopping criterion (see :ref:`Nested
Sampling Errors`), so the time required tends to scale with the number of
batches added.

Pool Questions
--------------

**My provided** `pool` **is crashing. What do I do?**

First, check that all relevant variables, functions, etc. are properly
accessible and that the `pool.map` function is working as intended. Second,
check if your pool has issues pickling some types of functions or evaluating
some of the functions in :mod:`~dynesty.sampling`. If those quick fixes don't
work, feel free to raise an issue. Multi-threading and multi-processing are
notoriously difficult to debug, however, especially on a problem I'm not
familiar with, so I might not be able to help all that much.
