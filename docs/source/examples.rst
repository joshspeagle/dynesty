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
of our slice sampling algorithms is shown below with `'slice'` in blue
and `'rslice'` in orange:

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

This toy problem was originally
`suggested <https://github.com/joshspeagle/dynesty/issues/111>`_
by Johannes Buchner for being multimodal with two roughly equal-amplitude
solutions. We are interested in modeling periodic data of the form:

.. math::

    y(x) = \exp\left[ n_a \sin(f_a x + p_a) + n_b \sin(f_b x + p_b) \right]

where :math:`x` goes from :math:`0` to :math:`2\pi`.

.. image:: ../images/examples_expwave_001.png
    :align: center

This model has six free parameters controlling the relevant amplitude,
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
<https://emcee.readthedocs.io/en/stable/>`_.

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
When sampling uniformly from a set of bounding ellipsoids, this tests whether
they fully encompass the likelihood contour. With few live points in higher
dimensions (here 50 live points in 10 dimensions), the ellipsoids tend to miss
the corners of the cube-shaped contours, so sampling without any enlargement
(`enlarge=1, bootstrap=0`) or with a constant enlargement factor
(`enlarge=1.25`) shrinks the prior volume too fast. Determining the enlargement
factor by bootstrapping (the default for `'unif'`) gives the correct
shrinkage, but at a large computational cost. Random slice sampling
(`'rslice'`) does not rely on the size of the bounds, but it needs enough
slices for the new points to be independent of the live points they start
from: with the default number of slices (3+d) it also shrinks too fast, while
with 50 slices the shrinkage is correct. The figure shows the
fraction of iterations with a shrinkage larger than a given value for each
of these cases (runs that shrink too fast lie above the expected line; see
the notebook for details):

.. image:: ../images/examples_pyramid_002.png
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

We examine the impact of sampling methods for high-dimensional
problems using a 200-D iid normal distribution with an associated
200-D iid normal prior. We find we are able to recover the appropriate evidence:

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

