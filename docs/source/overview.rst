==========
Background
==========

Bayesian Inference
==================

In the context of `Bayesian inference
<https://en.wikipedia.org/wiki/Bayesian_inference>`_, we are often interested
in estimating the **posterior** :math:`P(\boldsymbol{\Theta} | \mathbf{D}, M)`
of a set of **parameters** :math:`\boldsymbol{\Theta}` for a given **model**
:math:`M` given some **data** :math:`\mathbf{D}`. This can be factored into a
form commonly known as **Bayes' Rule** to give

.. math::

    P(\boldsymbol{\Theta} | \mathbf{D}, M) = 
    \frac{P(\mathbf{D} | \boldsymbol{\Theta}, M) 
    P(\boldsymbol{\Theta} | M)}{P(\mathbf{D} | M)} \equiv
    \frac{\mathcal{L}(\boldsymbol{\Theta}) \pi(\boldsymbol{\Theta})}
    {\mathcal{Z}}

where 

.. math::

    P(\mathbf{D} | \boldsymbol{\Theta}, M) \equiv 
    \mathcal{L}(\boldsymbol{\Theta})

is the **likelihood**,

.. math::

    P(\boldsymbol{\Theta}| M) \equiv \pi(\boldsymbol{\Theta})

is the **prior**, and 

.. math::

    P(\mathbf{D} | M) \equiv \mathcal{Z} = \int_{\Omega_{\boldsymbol{\Theta}}}
    \mathcal{L}(\boldsymbol{\Theta}) \pi(\boldsymbol{\Theta}) \, 
    d\boldsymbol{\Theta}

is the **evidence**, with the integral taken over the entire domain
:math:`\Omega_{\boldsymbol{\Theta}}` of :math:`\boldsymbol{\Theta}` 
(i.e. over all possible :math:`\boldsymbol{\Theta}`).

For complicated data and models, the posterior is often intractable and must
be estimated using numerical methods (see, e.g.,
`here <https://arxiv.org/abs/1506.08640>`__).

Nested Sampling
===============

Overview
--------

`Nested sampling <https://en.wikipedia.org/wiki/Nested_sampling_algorithm>`_
is a method for estimating the Bayesian evidence :math:`\mathcal{Z}` first
proposed and developed by `John Skilling
<https://dx.doi.org/10.1063%2F1.1835238>`_. The basic idea
is to approximate the evidence by integrating
the prior in nested "shells" of constant likelihood. 
Unlike `Markov Chain Monte Carlo (MCMC)
<https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ methods
which can only generate samples *proportional to* the posterior, Nested
Sampling simultaneously estimates both the evidence and the posterior.
It also has a variety of appealing statistical properties, which include:

* well-defined stopping criteria for terminating sampling,

* generating a sequence of independent samples,

* flexibility to sample from complex, multi-modal distributions,

* the ability to derive how statistical and sampling uncertainties impact
  results *from a single run*, and

* being trivially parallelizable.

`Dynamic Nested Sampling <https://arxiv.org/abs/1704.03459>`_ goes even 
further by allowing samples to be allocated adaptively during the course of 
a run to better sample areas of parameter space to maximize a chosen 
objective function. This allows a particular Nested Sampling algorithm to 
adapt to the shape of the posterior in real time, improving both accuracy 
and efficiency.

These points will be discussed elsewhere in the documentation when relevant.

How It Works
------------

Nested Sampling attempts to estimate :math:`\mathcal{Z}` by treating the
integral of the posterior over all :math:`\boldsymbol{\Theta}` as instead an
integral over the **prior volume**

.. math::

    X(\lambda) \equiv \int_{\boldsymbol{\Theta} : 
    \mathcal{L}(\boldsymbol{\Theta}) > \lambda} 
    \pi(\boldsymbol{\Theta}) \, d\boldsymbol{\Theta}

contained within an **iso-likelihood contour** set by
:math:`\mathcal{L}(\boldsymbol{\Theta}) = \lambda` via:

.. math::

    \mathcal{Z} = \int_{0}^{+\infty} X(\lambda) \, d\lambda = 
    \int_{0}^{1} \mathcal{L}(X) \, dX

assuming :math:`\mathcal{L}(X(\lambda)) = \lambda` exists.
In other words, if we can evaluate the iso-likelihood contour 
:math:`\mathcal{L}_i \equiv \mathcal{L}(X_i)` associated with a bunch of
samples from the prior volume :math:`1 > X_0 > X_1 > \dots > X_N > 0`,
we can compute the evidence using standard numerical integration techniques
(e.g., the `trapezoid rule <https://en.wikipedia.org/wiki/Trapezoidal_rule>`_).
Computing the evidence using these "nested shells" is what gives Nested
Sampling its name.

Basic Algorithm
---------------

Draw :math:`K` **"live" points** (i.e. particles) from the prior
:math:`\pi(\boldsymbol{\Theta})`. At each iteration :math:`i`, remove the live
point with the lowest likelihood :math:`\mathcal{L}_i` and replace it with a
new live point *sampled from the prior* subject to the constraint
:math:`\mathcal{L}_{i+1} \geq \mathcal{L}_i`. It can be shown through some neat
statistical arguments (see :ref:`Nested Sampling Errors`) that this sampling
procedure actually allows us to estimate the prior volume of the *previous*
live point (a **"dead" point**) as:

.. math::

    \ln X_i \approx -\frac{i \pm \sqrt{i}}{K}

Once some stopping criteria are reached and sampling terminates, the remaining
set of live points are distributed uniformly within the final prior volume.
These can then be "recycled" and added to the list of samples.

Evidence Estimation
-------------------

The evidence integral can be numerically approximated using a set of
:math:`N` dead points via

.. math::

    \mathcal{Z} = \int_{0}^{1} \mathcal{L}(X) \, dX \approx \hat{\mathcal{Z}} =
    \sum_{i=1}^{N} \, f(\mathcal{L}_i) \, f(\Delta X_i) \equiv 
    \sum_{i=1}^{N} \, \hat{w}_i

where :math:`\hat{w}_i` is each point's estimated weight.
For a simple linear integration scheme using rectangles, we can take
:math:`f(\mathcal{L}_i) = \mathcal{L}_i` and
:math:`f(\Delta X_i) = X_{i-1} - X_i`.
For a quadratic integration scheme using trapezoids (as used in ``dynesty``),
we instead can take
:math:`f(\mathcal{L}_i) = (\mathcal{L}_{i-1} + \mathcal{L}_i) / 2`.

Posterior Estimation
--------------------

We can subsequently estimate posteriors "for free" from the same :math:`N`
dead points by assigning each sample its associated **importance weight**

.. math::

    P(\boldsymbol{\Theta}_i) = P(X_i) \equiv p_i \approx \hat{p}_i =
    \frac{\hat{w}_i}{\sum_{i=1}^{N} \hat{w}_i} = 
    \frac{\hat{w}_i}{\hat{\mathcal{Z}}}

Stopping Criteria
-----------------

The remaining evidence :math:`\Delta \hat{\mathcal{Z}}_i` at iteration
:math:`i` can roughly be bounded by

.. math::

    \Delta \hat{\mathcal{Z}}_i \approx \mathcal{L}_{\max} X_i

where :math:`\mathcal{L}_{\max}` is the the maximum likelihood point
contained within the remaining set of :math:`K` live points. This essentially
assumes that the remaining prior volume interior to the last dead point is a
uniform slab with likelihood :math:`\mathcal{L}_{\max}`.

This can be turned into a relative stopping criterion by using the (log-)ratio
between the current estimated evidence :math:`\hat{\mathcal{Z}}_i` and the
remaining evidence :math:`\Delta \hat{\mathcal{Z}}_i`:

.. math::

    \Delta \ln \hat{\mathcal{Z}}_i \equiv 
    \ln \left(\hat{\mathcal{Z}}_i + \Delta \hat{\mathcal{Z}}_i \right) -
    \ln \hat{\mathcal{Z}}_i

Stopping at a given :math:`\Delta \ln \hat{\mathcal{Z}}_i` value (`dlogz`)
then means sampling until only a *fraction* of the evidence remains unaccounted
for.

In general, this error estimate serves as a (rough) upper bound
(since :math:`X_i` is not exactly known) that can be used for deciding
when to stop sampling from an arbitrary distribution while estimating the
evidence. Other stopping criteria are discussed in
:ref:`Dynamic Nested Sampling`.

Challenges
----------

Nested Sampling has two main main theoretical requirements:

#. Samples must be evaluated *sequentially* subject to the likelihood
   constraint :math:`\mathcal{L}_{i+1} \geq \mathcal{L}_{i}`, and

#. All samples used to compute/replace live points must be **independent
   and identically distributed (i.i.d.)** random variables *drawn from
   the prior*.

The first requirement is entirely algorithmic and straightforward to satisfy
(even when sampling in parallel). The second requirement, however, is much more
challenging if we hope to sample efficiently: while it is straightforward to
generate samples from the prior, by design Nested Sampling makes this simple
scheme increasingly more inefficient since the remaining prior volume
shrinks *exponentially* over time.

Solutions to this problem often involve some combination of:

#. Proposing new live points by "evolving" a copy of one (or more) current
   live points to new (independent) positions subject to the likelihood
   constraint, and

#. Bounding the iso-likelihood contours using simple but flexible functions
   in order to exclude regions with lower likelihoods.

In both cases, it is much easier to deal with uniform (rather than arbitrary)
priors. As a result, most nested sampling algorithms/packages (including
``dynesty``) are designed to sample within the :math:`D`-dimensional unit cube.
Samples are transformed samples back to the original parameter space
"on the fly" only when needed to evaluate the likelihood.
Accomplishing this requires an appropriate **prior transform**, described
in more detail under :ref:`Prior Transforms`.

Typical Sets
============

One of the elegant features of Nested Sampling is it directly incorporates the
ideas behind a `typical set
<http://mc-stan.org/users/documentation/case-studies/curse-dims.html>`_
into the estimation. Since this concept is **crucial** in most
Bayesian inference applications but rarely discussed explicitly in 
applied methods such as MCMC, it is important to take some time to 
discuss it in more detail.

Quick Overview
--------------

In general, the contribution to the posterior at a given value (position)
:math:`\boldsymbol{\Theta}` has two components. The first arises from the
particular *value* of the posterior itself, :math:`P(\boldsymbol{\Theta})`.
The second arises from the total (differential) *volume*
:math:`dV(\boldsymbol{\Theta})` encompassed by all
:math:`\boldsymbol{\Theta}`'s with the particular
:math:`P(\boldsymbol{\Theta})`. We can understand this intuitively: the
contributions from a small region with large posterior values can be
overwhelmed by contributions from much larger regions with small posterior
values. 

This "tug of war" between the two elements means that the regions which
contribute the most to the overall posterior are those that maximize the
joint quantity

.. math::

    w(\boldsymbol{\Theta}) \propto P(\boldsymbol{\Theta}) \,
    dV(\boldsymbol{\Theta})

This region typically forms a "shell" surrounding the mode (i.e. the
`maximum a posteriori (MAP) 
<https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>`_ value)
and is what is usually referred to as the **typical set**. This behavior
becomes more accentuated as the dimensionality increases: since volume scales
as :math:`r^D`, increasing the dimensionality of the problem creates
exponentially more volume further away from the posterior mode.

Typical Sets in Nested Sampling
-------------------------------

Under the framework of Nested Sampling, this concept naturally emerges from
the concept of integrating the evidence in shells of "prior volume":

.. math::

    \mathcal{Z} = \int_{0}^{1} \mathcal{L}(X) \, dX
    \approx \sum_{i=1}^{N} \, f(\mathcal{L}_i) \, f(\Delta X_i)
    
We can see directly that the contribution of a particular iso-likelihood
contour :math:`\mathcal{L}(X)` to the integral depends both on its 
"amplitude" :math:`\mathcal{L}(X)` along with the (differential) prior volume
:math:`dX` it occupies. This is maximized when both these quantities are
jointly maximized, which occurs over points that represent the typical set.
Because of the contribution from the "density" and "volume" terms
are clearly seen here, this is sometimes also referred to as
the **posterior mass**.
Since the posterior importance weights

.. math::

    \hat{p}_i \propto \hat{w}_i = f(\mathcal{L}_i) \, f(\Delta X_i)

are also directly proportional to these quantities, Nested Sampling also
naturally weights samples by their contribution to the typical set.

Priors in Nested Sampling
=========================

Unlike MCMC or similar methods, Nested Sampling starts by randomly sampling
from the entire parameter space specified by the prior. This is not possible
unless the priors are "proper" (i.e. that they integrate to 1). So while
Normal priors spanning (:math:`-\infty`, :math:`+\infty`) are fine, Uniform
priors spanning the same range are not and must be bounded. 

**It cannot be
stressed enough that the evidence is entirely dependent on the "size" of the
prior.** For instance, a wider Uniform prior will decrease the contribution of
high-likelihood regions to the evidence estimate, leading to a lower overall
value. Priors should thus be carefully chosen to ensure models can be properly
compared using the evidences computed from Nested Sampling.

In addition to affecting the evidence estimate, the prior also directly affects
the overall expected runtime. Since, in general, the posterior
:math:`P(\boldsymbol{\Theta})` is (much) more localized that the prior
:math:`\pi(\boldsymbol{\Theta})`, the "information" we gain from updating
from the prior to the posterior can be characterized by the
**Kullback-Leibler (KL) divergence** (see
`here <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__
for more information):

.. math::

    H \equiv \int_{\Omega_{\boldsymbol{\Theta}}} P(\boldsymbol{\Theta})
    \ln\frac{P(\boldsymbol{\Theta})}{\pi(\boldsymbol{\Theta})} \,
    d\boldsymbol{\Theta}

It can be shown/argued that the total number of steps :math:`N` needed to
integrate over the posterior roughly scales as:

.. math::

    N \propto HK

In other words, increasing the size of the prior *directly* impacts the amount
of time needed to integrate over the posterior. This highlights one of the
main drawbacks of nested sampling: **using less "informative" priors will
increase the expected number of nested sampling iterations**.
