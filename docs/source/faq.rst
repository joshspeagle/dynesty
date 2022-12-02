===
FAQ
===

This page contains a collection of frequently asked questions 
from ``dynesty`` users, along with some answers that hopefully are helpful to
you. If you don't see your particular issue addressed here, feel free to 
`open an issue <https://github.com/joshspeagle/dynesty/issues>`_.

**For citation information, see the :ref:`Citations` section on the homepage.**

Sampling Questions
------------------

**What sampling method should I be using?**

This is always problem-dependent, but some general advice is
to select one based on the dimensionality of your problem. In low dimensions,
uniform sampling is often quite efficient since the bounding distributions
can often encompass the majority of the prior volume. In moderate dimensions,
random walks often can serve as an effective way to propose new points
without relying on the exact shape/size of the bounds being correct. In
higher dimensions, we generally need non-rejection methods such as slice
sampling to generate samples efficiently since the prior volume is so large.
Using gradients can also help generate efficient proposals in this regime.
``dynesty`` uses these rules-of-thumb by default to choose a sampling option
with `'auto'`.

**Sampling seems to freeze around an efficiency of 10%. Is this a bug?**

This isn't a bug, but probably just a consequence of the first bounding update.
By default, `dynesty` waits to actually start sampling using the proposed
sampling/bounding methods passed to the sampler until a set of conditions
specified in `first_update` are satisfied. This lets the live points somewhat
move away from the edges of the prior and begin to adapt to the shape of the
target distribution, which helps to avoid problems such as the bounds
"shredding" the live points into lots of tiny islands. The basic heuristic
used is to wait until uniform proposals from the prior hit a cumulative
efficiency of 10%, but that threshold can be adjusted using the
`first_update` argument.

**Is there an easy way to add more samples to an existing set of results?**

Yes! There are actually a bunch of ways to do this. If you have the static
`NestedSampler` currently initialized, just executing `run_nested()` will start
adding samples where you left off.If you're instead interested in adding
more samples to a previous part of the run, the best strategy is to just
start a new independent run and then "combine" the old and new runs together
into a single (improved) run using the :meth:`~dynesty.utils.merge_runs`
function.

If you're using the `DynamicNestedSampler`, executing `run_nested` will
automatically add more dynamically-allocated samples based on your
target weight function as long as the stopping criteria hasn't been met.
If you would like to add a new batch of samples manually,
running `add_batch` will assign a new set of samples.
You can also specifically add new batch corresponding to a certain likelihood
range (i.e. corresponding to where your posterior is concentrated).
Also, if you are primarily interested in the posterior, you can use larger
values of n_effective parameter of `run_nested` as that will ensure your posterior
is less noisy.
Finally, :meth:`~dynesty.utils.merge_runs` also works with results generated
from Dynamic Nested Sampling, so it is just as easy to set off a new run and
combine it with your original result.

**There are inf values in my lower/upper log-likelihood bounds!
Should I be concerned?**

In most cases no. As mentioned in :ref:`Running Internally`, these values
are just the lower and upper limits of the log-likelihood used to limit
your sampling. If you're sampling starting from the prior, 
you're starting out from a likelihood of 0 and therefore a 
log-likelihood of `-inf`. If you haven't specified a particular `logl_max`
to terminate sampling, the default value is set to be `+inf` so it will
never prematurely terminate sampling. These values can change during
Dynamic Nested Sampling, at which point they serve as the endpoints between
which a new batch of live points is allocated.

In rare cases, errors in these bounds can be signs of Bad Things that may
have happened while sampling. This is often the case if the 
log-likelihood values being sampled (and displayed) are also 
are nonsensical (e.g., involve `nan` or `inf` values, etc.).
In that case, it is often useful to terminate the run early 
and examine the set of samples to see if there are any possible issues.

**Sometimes while sampling my estimated evidence
errors become undefined! Should I be concerned?**

Most often this is *not* a cause for concern. As mentioned in
:ref:`Approximate Evidence Errors`, `dynesty` uses an approximate method to
estimate evidence errors in real time based on the KL divergence
("information gain") and the current number of live points.
Sometimes this approximation can lead to
improper results (i.e. negative variances), which can often occur
early in the run when there is a lot of uncertainty in the prior volume.
While this often "corrects" itself later in the run, 
sometimes the effect can persist. Regardless of
whether the approximation converges, however, errors can still be computed
using the functions described in :ref:`Nested Sampling Errors` as normal.
I am currently working on developing a more robust approximation that
avoids some of these issues.

In rare cases, issues with the evidence error approximation can be a sign
that something has gone Terribly Wrong during the sampling phase. This
is often the case if the log-likelihood values being output also
are nonsensical (e.g., involve `nan` or `inf` values).
In that case, it is often useful to terminate the run early 
and examine the set of samples to see if there are any possible issues.

**When adding batches of live points sometimes the log-likelihoods being
displayed don't monotonically increase as I expect. What's going on?**

When points are added in each batch, they are allocated randomly between
the lower and upper log-likelihood bounds (since they are being sampled
randomly). These values are the ones being output to the terminal.
Once all the points have been allocated, then nested sampling
can begin by replacing each of the lowest log-likelihood values with a better
one.

**Sampling is taking much longer than I'd like. What should I do?!**

Unfortunately, there's no catch-all solution to this. The most important
first step is to make sure you're examining real-time outputs using the
`print_progress=True` option (enabled by default) if you're sampling internally
using :meth:`~dynesty.sampler.Sampler.run_nested` and printing out progress
if sampling externally using, e.g., :meth:`~dynesty.sampler.Sampler.sample`.

If the bounding distribution is updating frequently and you're using more
computationally intensive methods such as `'multi'`, some of this might be
due to excessive overhead associated with constructing the bounds. This can
be reduced by increasing `update_interval`.

If the overall sampling efficiency is low (*relative to what you'd expect*), it
might indicate that the distribution used (e.g., `'single'`) isn't effective
and more complex ones such as `'multi'` should be used instead. If you're
already using those but still getting inefficient proposals, that might
indicate that the bounding distribution are struggling to capture the
target distribution. This can happen if, e.g., the posterior occupies a thin,
strongly-curved manifold in several dimensions, which is hard to model with
a series of overlapping ellipsoids or other similar distributions.

Another possible culprit might be the enlargement factors. While the default
25% value usually doesn't significantly decrease the efficiency, there some
exceptions. If you are instead deriving expansion factors from bootstrapping,
it's possible you're experiencing severe Monte Carlo noise (see 
:ref:`Bounding Questions`). You could try to resolve this by either using
more live points or switching to an alternate sampling method less sensitive
to the size of the bounding distributions such as `'rwalk'` or `'rslice'`.

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
:ref:`Priors in Nested Sampling`, the time it takes to sample to a
given `dlogz` tolerance scales as the "information" gained by updating from
the prior to the posterior. Since Nested Sampling starts by sampling from the
entire prior volume, having overly-broad priors will increase the runtime.

**I noticed that the number of iterations and/or function calls during a run
don't exactly match up with the limits I specify using,
e.g.,** `maxiter` **or** `maxcall` **. Is this a bug?**

No, this is not a bug (i.e. this behavior is not unintended). 
When proposing a new point, ``dynesty`` currently only
checks the stopping criterion specified (whether iterations or function calls)
*after* that point has been accepted. This can also happen when using the 
`~dynesty.dynamicsampler.DynamicSampler` to propose a new batch of points,
since the first batch of points need to be allocated before checking the
stopping criterion.

**I find other sampling are inefficient relative to `'unif'`.**
**Why would I ever want to use them?**

The main reason these methods are more inefficient than uniform sampling
is that they are designed to sample from higher-dimensional (and somewhat
more "difficult") distributions, which
is inherently challenging due to the behavior of :ref:`Typical Sets`.
Broadly speaking, these methods are actually reasonably efficient
when compared to other (non-gradient) sampling methods on similar problems
(see, e.g., `here <https://arxiv.org/pdf/1502.01856.pdf>`_).

In addition, it is also important to keep in mind that samples from ``dynesty``
are nominally *independent* (i.e. already "thinned"). As a reference point,
consider an MCMC algorithm with a sampling efficiency of 20%. While this
might seem more efficient than the 4% default target efficiency of `'rwalk'`
in ``dynesty``, the output samples from MCMC are (by design) correlated.
If the resulting MCMC chain needs to be thinned by more than a factor of 5
to ensure independent samples, its "real" sampling efficiency is actually
then below the 4% nominally achieved by ``dynesty``. This is discussed
further in the `release paper
<https://github.com/joshspeagle/dynesty/tree/master/paper/dynesty.pdf>`_.

**How many walks (steps) do you need to use for** `'rwalk'` **?**

In general, random walk behavior leads to excursions from the mean at a rate
that scales as (roughly) :math:`\sqrt{n} \sigma` where :math:`n` is the number
of walks and :math:`\sigma` is the typical length scale. The number of steps
needed then roughly scales as :math:`d^2`. In general this behavior doesn't
dominate unless sampling in high (:math:`d \gtrsim 20`) dimensions. In lower
dimensions (:math:`d \lesssim 15`), `walks=25` is often sufficient, while in
moderate dimensions (:math:`d \sim 15-25`) `walks=50` or greater are often
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

Our implementation of `'rslice'` more closely follows the method
employed in PolyChord.

**How many slices ("repeats") do you need to use for** `'slice'` **?**

Since slice sampling is a form of non-rejection sampling,
the number of "slices" requires for Nested Sampling is
(in theory) independent of dimensionality and can remain relatively constant.
This is especially true if there are a set of local principle axes
that can be effectively captured by the bounding distributions
(e.g., `'multi'`). There are more pathological cases, however,
where the number of slices can weakly scale with dimensionality. In general
we find that the default (and conservative) `slices=3`
is robust under a wide variety of circumstances. Note that for the
`'slice'` sampler slices=3 means that slice steps will be done 3 times over
each of the dimension of the problem (N). I.e. the total number of the moves
will be 3*N. Also note that for the `'rslice'` sampler the default
is `slices=3+N` steps as `'rslice'` does not loop over each of the dimension,
as it chooses the move directions randomly.

**The stopping criterion for Dynamic Nested Sampling is taking a long
time to evaluate. Is that normal?**

This might mean you are using a version of ``dynesty`` below v1.2 or
you are using a large number of simulations to estimate the errors.
In earlier versions, the stopping criteria was much more computationally
intensive to evaluate. However, in both earlier and current versions, using
(1) large numbers of simulations with (2) large numbers of samples 
with (3) a large number of varying live points can make the stopping criteria
difficult to evaluate quickly. See 
:ref:`Nested Sampling Errors` for additional details.

**I'm trying to sample using gradients but getting extremely poor performance.
I thought gradients were supposed to make sampling more efficient!
What gives?**

While gradients are extremely useful in terms of substantially improving
the scaling of most sampling methods with dimensionality (gradient-based
methods have better polynomial scaling than non-gradient slice sampling, both
of which are *substantially* better over the runaway exponential scaling
of random walks), it can take a while for these benefits to really kick in.
These scaling arguments generally ignore the constant prefactor, which
can be quite large for many gradient-based approaches that require
integrating along some trajectory, often resulting in (at least) dozens of
function calls per sample. This often makes it more efficient to run simpler
sampling techniques on lower-dimensional problems. In general, Nested Sampling
methods are also unable to exploit gradient-based information to the same
degree as Hamiltonian Monte Carlo approaches, which further degrades
performance and scaling relative to what you might naively expect.

If you feel like your performance is poorer than expected even given these
caveats, or if you notice other results that make you highly suspicious of the
resulting samples, please double-check the :ref:`Sampling with Gradients`
page to make sure you've passed in the correct log-likelihood gradient and are
dealing with the unit cube Jacobian properly. Failing
to apply this (or applying it twice) violates conservation of energy and
momentum and leads to the integration timesteps along the trajectories
changing in undesirable ways. 
It's also possible the numerical errors in the Jacobian (if you've set
`compute_jac=True`) might be propagating through to the computed trajectories.
If so, consider trying to compute the analytic Jacobian by hand to reduce
the impact of numerical errors.

If you still find subpar performance, please feel free to 
`open an issue <https://github.com/joshspeagle/dynesty/issues>`_.


Live Point Questions
--------------------

**How many live points should I use?**

Short answer: **it depends**.

Longer answer: Unfortunately, there's no easy answer here.
Increasing the number of live points helps establish more
flexible and robust bounds, improving the overall sampling efficiency and
prior volume resolution. However, it simultaneously increases the runtime.
These competing behaviors mean that compromises need to be made which are
problem-dependent.

In general, for ellipsoid-based bounds an absolute minimum of `ndim + 1`
live points is "required", with `2 * ndim` being a (roughly) "safe" threshold.
If bootstraps are used to establish bounds while sampling uniformly, however,
many (many) more live points should be used. 
Around `50 * ndim` points are recommended *for each expected mode*.

Methods that do not depend on the absolute size of the bounds (but instead rely
on their shape) can use fewer live points. Their main restriction is
that new live point proposals (which "evolve" a copy of an existing live point
to a new position) must be independent of their starting point. Using too
few points can require excessive thinning, which quickly negates
the benefit of using fewer points if speed is an issue.
`10 * ndim` per mode seems to work reasonably well, although
this depends sensitively on the amount of prior volume that has to be
traversed: if the likelihood is a set of tiny islands in an ocean of
prior volume, then you'll need to use more live points to avoid missing them.
See :ref:`LogGamma`, :ref:`Eggbox`, or :ref:`Exponential Wave` for
some examples of this in practice.

Bounding Questions
------------------

**What bounds should I be using?**

Generally, `'multi'` (multiple ellipsoid decomposition) is the most
adaptive, being able to model a wide variety of behaviors and complex
distributions. It is enabled in ``dynesty`` by default.

For simple unimodal problems, `'single'` (a single bounding ellipsoid) 
can often do quite well. It also helps to guard against cases where
methods like `'multi'` can accidentally "shred" the posterior into many pieces
if the ellipsoid decompositions are too aggressive.

For low-dimensional problems, ensemble methods like `'balls'` and `'cubes'` 
can be quite effective by allowing live points themselves 
to create "emergent" structure. These can create more flexible shapes than
`'multi'`, although they have trouble modeling separate structures with
wildly different shapes.

In almost all cases, using no bound (`'none'`) should be seen as a fallback
option. It is mostly useful for systematics checks or in cases where the
number of live points is small relative to the number of dimensions.

**What are the differences between** `'multi'` **and MultiNest, nestle, etc.?**

The multi-ellipsoid decomposition/bounding method implemented in ``dynesty``
is entirely based on the algorithm implemented in `nestle 
<http://kylebarbary.com/nestle/>`_ which itself is based on the algorithm
*described* in `Feroz, Hobson & Bridges (2009) 
<https://arxiv.org/abs/0809.3437>`_. As such, it doesn't include any
improvements, changes, etc. that may or may not be included in 
`MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest/>`_.
Specifically, it uses a simple scheme based on iterative k-means
clustering than some of the more robust methods based on `agglomerative
clustering <https://en.wikipedia.org/wiki/Hierarchical_clustering>`_
implemented by some other codes such as
`UltraNest <https://github.com/JohannesBuchner/UltraNest/>`_.

In addition, there are a few differences in the portion of the algorithm that
decides when to split an ellipsoid into multiple ellipsoids. As with
``nestle``, the implementation in ``dynesty`` is more conservative about
splitting ellipsoids to avoid over-constraining the remaining prior volume and
also enlarges all the resulting ellipsoids by a constant volume prefactor.
It also recomputes the ellipsoids from scratch each time there is a
bounding update, rather than using ellipsoids from previous iterations.
In general this results in a slightly lower sampling efficiency but greater
overall robustness.

``dynesty`` also uses different heuristics than ``MultiNest`` or ``MultiNest``
when deciding, e.g., when to first construct bounds. By default, ``dynesty``
waits until the efficiency hits 10% and a certain number of iterations have
passed before deciding to try split up live points into any sort of
ellipsoid decomposition. This helps to avoid problems with "shredding" the
early set of live points (which tend to be quite dispersed) into an enormous
set of ellipsoids but can substantially affect the runtime for simple problems
with tight priors. See :ref:`Bounding Options` for additional details as well
as the answer below.

Finally, ``dynesty`` regularizes the ellipsoids based on their
`condition number <https://blogs.mathworks.com/cleve/2017/07/17/what-is-the-condition-number-of-a-matrix/>`_
to avoid issues involving numerical instability. This can reduce the sampling
efficiency for problems with very skewed distributions (i.e. large axis ratios)
but helps to ensure stable performance.

**No matter what bounds, options, etc. I pick, the initial samples all
come from `bound = 0` and continue until the overall efficiency is quite low.
What's going on here?**

By default, ``dynesty`` opts to wait until some time has passed until
constructing the first bounding distribution.
This behavior is designed to avoid constructing overly large bounds that often
significantly exceed the confines of the unit cube, which can lead to excessive
time spent generating random numbers early in a given run. 
Prior to constructing the initial bound,
samples are proposed from the unit cube, which is taken to be `bound = 0`. 
The options that control these
heuristics can be modified using the `first_update` argument.

**During a run I sometimes see the bound index jump forward several places.
Is this normal?**

To avoid getting stuck sampling from bad bounding distributions (see above),
``dynesty`` automatically triggers a bounding update whenever the number of 
likelihood calls exceeds `update_interval` while sampling from a particular
bound. This can lead to multiple bounds being constructed before the sample
is accepted.

**A constant expansion factor seems arbitrary and I want to try 
out bootstrapping. How many bootstrap realizations do I need?**

Sec. 6.1 of `Buchner (2014) <https://arxiv.org/abs/1407.5459>`_ discusses
the basic behavior of bootstrapping and how many iterations are needed to
ensure that realizations do not include the same live point over some number
of realizations. `bootstrap = 20` appears to work well in practice, although
this is more aggressive than the `bootstrap = 50` recommended by
Buchner.

**When bootstrapping is on, sometimes during a run a bound 
will be really large. This then leads to a large number of log-likelihood calls
before the bound shrinks back to a reasonable size again. 
Why is this happening? Is this a bug?**

This isn't (technically) a bug, but rather Monte Carlo noise
associated with the bootstrapping process.
Depending on the chosen method, sometimes bounds can be unstable, leading
to large variations between bootstraps and subsequently large expansions
factors. Some of this is explored in the
:ref:`Gaussian Shells` and :ref:`Hyper-Pyramid` examples. In general,
this is a sign that you don't have enough live points to robustly determine
your log-likelihood bounds at a given iteration, and should likely be running
with more. Note that "robustly" is the key word here, since it can often
take a (some might find "excessively") large number of live points 
to confidently determine that you aren't missing any 
hidden prior volume.

Pool/Parallelization Questions
------------------------------

**My provided** `pool` **is crashing. What do I do?**

First, check that all relevant variables, functions, etc. are properly
accessible and that the `pool.map` function is working as intended. Sometimes
pools can have issues passing variables to/from members or executing tasks
(a)synchronously depending on the setup.

Second, check if your pool has issues pickling some types of functions 
or evaluating some of the functions in :mod:`~dynesty.sampling`. In general,
nested functions require more advanced pickling (e.g., ``dill``), 
which is not enabled with some pools by default.

If those quick fixes don't work, feel free to raise an issue. 
However, as multi-threading and multi-processing are notoriously 
difficult to debug, especially on a problem I'm not familiar with, 
it's likely that I might not be able to help all that much.


**How to decide on the number of processes in a pool and how to set queue_size**

Assuming that you decided on the number of live-points K that you want to use and that the likelihood evaluation is not very quick, you should use as many processes as you can up to around K. The queue_size should be equal the number of processes. If you are using the the number of processes that M is smaller than K, you may want to use  :math:`M=K//2` or :math:`M=K//3` i.e integer fractions. So if you are using 1024 live-points all powers of two up to 1024 would be good choice for the number of processes.


**I would like to run dynesty across multiple nodes on a cluster. How do I do that ?**

The best way is to use the
`schwimmbad <https://schwimmbad.readthedocs.io/en/latest/>`_ package
and its `MPIPool`. You should be able to use this pool in the same way you would use the `multiprocessing.Pool`. (see `schwimmbad` docs for more info). Here is a small example::
  
  from schwimmbad import MPIPool
  import numpy as np, sys, dynesty

  def ptform(x):
    return 10 * x - 5

  def func(x):
    return -0.5 * np.sum(x**2)

  if __name__ == '__main__':
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    dns = dynesty.DynamicNestedSampler(
        func, ptform, 10, pool=pool)
    dns.run_nested()

**When running on a cluster I run into a time limit before dynesty finishes. What should I do?**

You should use the checkpointing ability of dynesty to save the state
of the sampler during sampling process. Then you should be able to restart
the sampling even if it was previously killed by the scheduler.


**When trying to use checkpointing I'm receiving errors because my function cannot be pickled**

If you receive the error like "Can't pickle local object", this is an error that means that python is not able to save the sampler due to the limitations of the python's pickler. The alternative is to use another pickling module like `dill`.
You can easily replace the pickling module by executing this::
  
  import dill
  import dynesty.utils
  dynesty.utils.pickle_module = dill

before the checkpointing/saving code and that will force dynesty to use dill.
