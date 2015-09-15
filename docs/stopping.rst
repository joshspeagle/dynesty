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


