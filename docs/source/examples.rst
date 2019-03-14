========
Examples
========

This page highlights several examples on how ``dynesty``
can be used in practice, illustrating both simple and more advanced
aspects of the code. Jupyter notebooks containing more details are available
`on Github <https://github.com/joshspeagle/dynesty/tree/master/demos>`_.

Gaussian Shells
===============

The "Gaussian shells" likelihood is a useful test case for illustrating the
ability of nested sampling to deal with oddly-shaped distributions that
can be difficult to probe with simple random-walk MCMC methods.

.. image:: ../images/examples_shells_001.png
    :align: center

``dynesty`` returns the following posterior estimate:

.. image:: ../images/examples_shells_002.png
    :align: center

25-D Correlated Normal
======================

``dynesty`` supports three tiers of sampling techniques: uniform sampling for
low dimensional problems, random walks for low-to-moderate dimensional
problems, and slice sampling for high-dimensional problems. The performance
of our three slice sampling algorithms is shown below with `'slice'` in blue,
`'rslice'` in orange, and `'hslice'` in green:

.. image:: ../images/examples_25d_001.png
    :align: center

The recovery of the mean and variances also looks reasonable:

.. rst-class:: sphx-glr-script-out

Out::

    Mean:

    slice:  [ 0.00464139  0.01779515 -0.00059102  0.01545529  0.00196981 
             -0.01102295  0.0224263   0.00128731  0.0165881   0.01004745  
              0.02403447  0.02490094  0.02634624  0.0221378   0.01095415  
              0.01451562  0.00810768  0.0186859   0.01441112  0.0142988   
              0.00562206  0.01013293  0.03261655  0.01514425 -0.00096744]

    rslice: [-0.01262438 -0.02215703 -0.02894398  0.00412    -0.01239199  
              0.00537269 -0.00883646 -0.01124688 -0.00622417 -0.02345228 
             -0.01226172 -0.01741414 -0.00340907 -0.02107367 -0.0440053 
             -0.00461723  0.00210266 -0.00553831 -0.0342508  -0.04259448 
             -0.03088255  0.00615101 -0.00708561 -0.01839912 -0.01779207]

    hslice: [-0.00770402 -0.02471443 -0.042487   -0.01095513 -0.03419283 
             -0.01587577 -0.0073069  -0.01633131 -0.01914578 -0.02243197 
             -0.01922804 -0.03532052 -0.0229837  -0.02118347 -0.00624797 
             -0.02247772 -0.02899336 -0.02943324 -0.02318504 -0.02445557 
             -0.02455501 -0.00661906 -0.029969   -0.01755215 -0.01973601]

    Variance:

    slice:  [0.99505849 0.97200556 1.03652393 0.99880137 0.98862659
             1.00075338 0.99494738 0.99976605 0.9965513  1.01882192
             0.97857377 0.99662175 0.98167938 0.98594533 0.99283048
             1.01748035 0.97116046 1.00298012 1.0111866  1.0202167
             0.99495185 1.04121714 0.99569076 1.00889279 0.97541806]

    rslice: [0.99821846 1.02606283 0.99985712 0.99358811 1.00021096
             0.98121015 0.99658629 0.99363295 1.00926491 0.99788756
             0.98109713 0.93410622 1.02039019 1.01227682 0.97890678
             1.03568037 1.01749501 0.98945627 0.99539522 0.98519908
             0.97363697 0.99418089 0.99500449 0.92339752 0.9456492 ]

    hslice: [0.99093439 1.00783986 1.00908646 0.98642076 1.02137535 
             1.00685834 0.98783381 1.02676786 1.021385   1.00869302
             0.96427675 0.96278338 0.98425856 1.00001262 1.00796364 
             1.01546776 1.01142401 1.01775994 0.99990323 1.00081825 
             1.00426292 1.00153755 0.99376306 0.99011333 0.98264584]

Eggbox
======

The "Eggbox" likelihood is a useful test case that demonstrates Nested
Sampling's ability to properly sample/integrate over multi-modal
distributions.

.. image:: ../images/examples_eggbox_001.png
    :align: center

The evidence estimates from two independent runs look reasonable:

.. image:: ../images/examples_eggbox_002.png
    :align: center

The posterior estimate also looks quite good:

.. image:: ../images/examples_eggbox_003.png
    :align: center

Exponential Wave
================

This toy problem was originally suggested by
`suggested <https://github.com/joshspeagle/dynesty/issues/111>`_ 
by Johannes Buchner for being multimodal with two roughly equal-amplitude
solutions. We are interested in modeling periodic data of the form:

.. math::

    y(x) = \exp\left[ n_a \sin(f_a x + p_a) + n_b \sin(f_b x + p_b) \right]

where :math:`x` goes from :math:`0` to :math:`2\pi`.

.. image:: ../images/examples_expwave_001.png
    :align: center

This model has six free parameters controling the relevant amplitude,
period, and phase of each component (which have periodic boundary conditions). 
We also have a seventh, :math:`\sigma`, corresponding to the amount of scatter.

The results are shown below.

.. image:: ../images/examples_expwave_002.png
    :align: center

.. image:: ../images/examples_expwave_003.png
    :align: center

Linear Regression
=================

Linear regression is ubiquitous in research. In this example we'll fit a line 

.. math::
    y = mx + b 

to data where the error bars have been over/underestimated by some fraction
of the observed value :math:`f` and need to be decreased/increased.
Note that this example is taken directly from the ``emcee`` `documentation 
<http://dan.iel.fm/emcee/current/user/line/>`_.

.. image:: ../images/examples_line_001.png
    :align: center

The trace plot and corner plot show reasonable parameter recovery.

.. image:: ../images/examples_line_002.png
    :align: center

.. image:: ../images/examples_line_003.png
    :align: center

Hyper-Pyramid
=============

One of the key assumptions of :ref:`Static Nested Sampling` (extended by
:ref:`Dynamic Nested Sampling`) is that we "shrink" the prior volume 
:math:`X_i` at each iteration :math:`i` as

.. math::

    X_{i} = t_i X_{i-1} ~ , \quad t_i \sim \textrm{Beta}(K, 1)

at each iteration with :math:`t_i` a random variable with distribution 
:math:`\textrm{Beta}(K, 1)` where :math:`K` is the total number of live points.
We can empirically test this assumption by using functions whose volumes can
be analytically computed directly from the position/likelihood of a sample.

One example of this is the "hyper-pyramid" function
from `Buchner (2014) <https://arxiv.org/abs/1407.5459>`_.

.. image:: ../images/examples_pyramid_001.png
    :align: center

We can compare the set of samples generated from ``dynesty``
with the expected theoretical shrinkage
using a `Kolmogorov-Smirnov (KS) Test 
<https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>`_.
When sampling uniformly from a set of bounding ellipsoids, we expect to be
more sensitive to whether they fully encompass the bounding volume. Indeed,
running on default settings in higher dimensions yields shrinkages that
are inconsistent with our theoretical expectation (i.e. we shrink too fast):

.. image:: ../images/examples_pyramid_003.png
    :align: center

If bootstrapping is enabled so that ellipsoid expansion factors are determined
"on the fly", we can mitigate this problem:

.. image:: ../images/examples_pyramid_002.png
    :align: center

Alternately, using a sampling method other than `'unif'` can also avoid this
issue by making our proposals less sensitive to the exact size/coverage
of the bounding ellipsoids:

.. image:: ../images/examples_pyramid_004.png
    :align: center

LogGamma
========

The multi-modal Log-Gamma distribution is useful for stress testing the
effectiveness of bounding distributions since it contains multiple modes
coupled with long tails.

.. image:: ../images/examples_loggamma_001.png
    :align: center

``dynesty`` is able to sample from this distribution in :math:`d=2` dimensions
without too much difficulty:

.. image:: ../images/examples_loggamma_003.png
    :align: center

Although the analytic estimate of the evidence error diverges (requiring us
to compute it numerically following :ref:`Nested Sampling Errors`),
we are able to recover the evidence and the shape of the posterior quite well:

.. image:: ../images/examples_loggamma_002.png
    :align: center

.. image:: ../images/examples_loggamma_004.png
    :align: center

Our results in :math:`d=10` dimensions are also consistent with the expected
theoretical value:

.. image:: ../images/examples_loggamma_005.png
    :align: center

200-D Normal
============

We examine the impact of gradients for sampling from high-dimensional
problems using a 200-D iid normal distribution with an associated
200-D iid normal prior. With Hamiltonian slice sampling (`'hslice'`), we find
we are able to recover the appropriate evidence:

.. image:: ../images/examples_200d_001.png
    :align: center

Our posterior recovery also appears reasonable, as evidenced by the
small snapshot below:

.. image:: ../images/examples_200d_002.png
    :align: center

We also find unbiased recovery of the mean and covariances in line with
the accuracy we'd expect given the amount of live points used:

.. image:: ../images/examples_200d_003.png
    :align: center

.. image:: ../images/examples_200d_004.png
    :align: center

Importance Reweighting
======================

Nested sampling generates a set of samples and associated importance weights,
which can be used to estimate the posterior. As such, it is trivial to
re-weight our samples to target a slightly different distribution using
**importance reweighting**. To illustrate this, we run ``dynesty`` on two 3-D
multivariate Normal distributions with and without strong covariances.

.. image:: ../images/examples_reweight_001.png
    :align: center

.. image:: ../images/examples_reweight_002.png
    :align: center

We then use the built-in utilities in ``dynesty`` to reweight each set of
samples to approximate the other distribution. Given that both samples have
non-zero coverage over each target distribution, we find that the results
are quite reasonable:

.. image:: ../images/examples_reweight_003.png
    :align: center

.. image:: ../images/examples_reweight_004.png
    :align: center

Noisy Likelihoods
=================

It is possible to sample from noisy likelihoods in
``dynesty`` just like with MCMC provided they are *unbiased*. While there
are additional challenges to sampling from noisy likelihood surfaces,
the largest is the fact that over time we expect the likelihoods to be biased
high due to the baised impact of random fluctuations on sampling: while
fluctuations to lower values get quickly replaced, fluctuations to higher
values can only be replaced by fluctuations to higher values elsewhere. This
leads to a natural bias that gets "locked in" while sampling, which can
substantially broaden the likelihood surface and thus the inferred posterior.

We illustrate this by adding in some random noise to a 3-D iid Normal
distribution. While the allocation of samples is almost identical, the
estimated evidence is substantially larger and the posterior substantially
broader due to the impact of these positive fluctuations.

.. image:: ../images/examples_noisy_001.png
    :align: center

.. image:: ../images/examples_noisy_002.png
    :align: center

If we know the "true" underlying likelihood, it is straightforward to
use :ref:`Importance Reweighting` to adjust the distribution to match:

.. image:: ../images/examples_noisy_003.png
    :align: center

However, in most cases these are not available. In that case, we have to rely
on being able to generate multiple realizations of the noisy likelihood at the
set of evaluated positions in order to obtain more accurate (but still noisy)
estimates of the underlying likelihood. These can then be used to get an
estimate of the true distribution through the appropriate
importance reweighting scheme:

.. image:: ../images/examples_noisy_004.png
    :align: center
