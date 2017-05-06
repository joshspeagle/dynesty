.. dynesty documentation master file, created by
   sphinx-quickstart on Sat May  6 18:44:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
dynesty
=======

Dynesty is a dynamic nested sampling package for estimating Bayesian
evidences and posteriors. The latest development version can be found at
http://github.com/joshspeagle/dynesty. Note that the code (and these docs) are
under active development.


Background
==========

`Nested sampling <https://en.wikipedia.org/wiki/Nested_sampling_algorithm>`_
is a method for estimating Bayesian evidences first proposed by
`John Skilling <https://dx.doi.org/10.1063%2F1.1835238>`_ by integrating
the prior in nested "shells" of constant likelihood. 
Unlike `Markov Chain Monte Carlo (MCMC) 
<https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ methods
which can only generate samples proportional to the posterior, nested sampling
inherently estimates both the evidence and the posterior during a single run.
It also has a variety of appealing statistical properties, which include:
* well-defined criteria for terminating sampling,
* generating nearly independent samples from the posterior,
* the ability to sample from complex, multi-modal distributions,
* deriving uncertainties on the result from a single run, and
* being trivially parallelizable.

`Dynamic nested sampling <https://arxiv.org/abs/1704.03459>`_ goes even 
further by allowing samples to be allocated adaptively during the course of 
a run to better sample areas of parameter space to maximize a chosen 
objective function. This allows a particular nested sampling algorithm to 
adapt to the shape of the posterior in real time, improving both accuracy 
and efficiency.


Overview
========

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   priors
   stopping

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

