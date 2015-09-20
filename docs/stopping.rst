==================
Stopping Criterion
==================

Nested sampling has no well-defined stopping point. As iterations
continue, the active points sample a smaller and smaller region of
prior space. This can continue indefinitely (up to the limits of your
machine). Unlike typical MCMC methods, we don't gain any additional
precision on the results by letting the algorithm run longer; the
precision is determined at the outset by the number of active
points. So, we want to stop iterations as soon as we think the active
points are doing a "pretty good job" sampling the remaining prior
volume -- once we've converged to the highest-likelihood regions such
that the likelihood is relatively flat within the remaining prior
volume.

For an arbitrary likelihood surface, there's no way to ever guarantee
that we've found the highest likelihood region. There could always be
some vanishingly small volume with such high likelihood that it
dominates the evidence integral (but this is a generic problem with
sampling methods). In practice, there are a couple heuristic stopping
criteria that work pretty well for determining when we've found the
highest-likelihood regions.

Estimated remaining evidence
----------------------------

In this criteria, we estimate the maximum remaining evidence and stop
when the contribution of the remaining evidence to the total evidence
falls below some threshold. The remaining evidence is difficult to
estimate (this is what we are trying to calculate in the algorithm
after all!), but the current active points give *some* idea of how
much evidence could remain. More specifically, the current active
point with the highest likelihood gives an estimate of the higest
likelihood in the region. We estimate the remaining evidence at iteration i
according to

.. math::
   \mathcal{Z}_\mathrm{est} = \mathcal{L}_\mathrm{max} X_i

where :math:`X_i` is the remaining prior volume.
The stopping criteria is then based on the ratio between the estimated
total evidence and the current evidence:

.. math::
   \log (\mathcal{Z}_i + \mathcal{Z}_\mathrm{est}) - \log \mathcal{Z}_i <
   \mathrm{thresh}

where thresh is a user-defined parameter that defaults to 0.5. This is the
``dlogz`` option in `nestle.sample`.

One can show that the expectation value of :math:`\mathcal{L}_\mathrm{max}` is 
:math:`\mathcal{L}_{i + N log N}`, where *N* is the number of active points.
This means that the highest-likelihood active point will be equivalent to a
point saved :math:`N log N` iterations later, after the remaining volume has
shrunk by a factor of *N*.

Declining weight of saved samples
---------------------------------

This method is based on looking for a flattening of the likelihood region
being traversed. While the likelihood of saved samples will always increase,
the increase will become smaller and smaller as the likelihood flattens.
We look for a critical point where the likelihood increase becomes smaller than
the decrease in the prior volume.

.. math::
   \mathcal{L}_{i+1} / \mathcal{L}_i &< X_i / X_{i+1}  \\
                                     &< e^{1/N}        \\
   \log \mathcal{L}_{i+1} - \log \mathcal{L}_i &< 1/N

This is equivalent to a declining sample weight (likelihood times
sample prior volume).  There is significant sampling noise in the
likelihood values, so we cannot stop immediately once this condition
is fulfilled. Instead, we look stop when this condition is fulfilled
for a number of repeated samples. In `nestle.sample`, this is set by
the ``decline_factor`` parameter, which instructs the algorithm to
stop after ``decline_factor * nsamples`` consecutive samples
fulfilling the condition. This stopping criteria is inactive by
default.
