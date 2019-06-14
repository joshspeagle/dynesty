=======
dynesty
=======

.. image:: ../images/title.gif
    :align: center

``dynesty`` is a Pure Python, MIT-licensed `Dynamic Nested Sampling
<https://arxiv.org/abs/1704.03459>`_ package for estimating Bayesian posteriors
and evidences. See :ref:`Crash Course` and :ref:`Getting Started`
for more information. The latest development version can be found `here
<http://github.com/joshspeagle/dynesty>`_.

**The release paper describing the code can be found**
`here <https://github.com/joshspeagle/dynesty/tree/master/paper/dynesty.pdf>`_.

Installation
============

``dynesty`` is compatible with both Python 2.7 and Python 3.6. It requires
``numpy`` (for arithmetic), 
``scipy`` (for special functions), 
``matplotlib`` (for plotting), and 
``six`` (to enforce Python 2/3 compliance). 
Installing the most recent stable version of the package is as easy as::

    pip install dynesty

Alternately, for users who might want newer development versions, it can also
be installed directly from a local copy of the repository by running::

    python setup.py install

Changelog
=========

.. image:: ../images/logo.gif
    :align: center

0.9.7 (2019-06-13)
-------------------
* Ensemble bounds can now adapt to elongated distributions (with
  `Johannes Buchner <https://github.com/JohannesBuchner>`_).

* Random walks now behave differently near boundaries (with
  `Gregory Ashton <https://github.com/GregoryAshton>`_).

* Pickling sampler states should now work better in Python 3 (with
  `Dustin Lang <https://github.com/dstndstnr>`_.

* Doubled output errors in default approximation in line with theoretical
  expectations.

* Small bugfixes and docfixes (with
  `Patricio Cubillos <https://github.com/dstndstnr>`_).


0.9.5.3 (2019-03-29)
-------------------
* Various small bugfixes, with contributions by
  `Gregory Ashton <https://github.com/GregoryAshton>`_ and
  `Johannes Buchner <https://github.com/JohannesBuchner>`_.

0.9.5 (2019-03-14)
-------------------
* Added support for periodic boundary conditions.

* Set up basic tests for continuous integration.

0.9.4 (2019-03-07)
-------------------
* Added a logo!

* Updated and reorganized documentation and demos.

* Added proper support for gradients.

* Changed defaults and added several "quality of life" improvements.

0.9.3 (2019-02-10)
-------------------
* Updated documentation.

* Modified re-scaling behavior to better deal with inefficient proposals.

* Improved stability of the current ellipsoid decomposition algorithm.

* Added new `'auto'` options and changed a number of defaults to make things
  easier for general users.

* Plotting now defaults to 95% credible intervals instead of 68%.

0.9.2 (2018-03-17)
------------------

* Added in a fast approximation option for `jitter_run` and `simulate_run`.

* Modified the default stopping heuristic. It now evaluates significantly
  faster but is a less accurate probe of the "true" KL divergence.

* Modified `'rwalk'` behavior to better deal with edge cases.

* Changed defaults so performance should now be more stable (albiet slower) 
  for the average user.

* Improved the stability of bounding ellipsoids.

* Fixed performance issues with `'rslice'` and `'hslice'`.

* Small plotting improvements.

0.9.1 (2018-03-01)
------------------

* Fixed a minor bootstrapping bug that affected performance for some users.

* Fixed a serious bug associated with the new singular decomposition algorithm
  and changed its behavior so it no longer auto-kills user runs when it fails.

0.9.0 (2018-02-25)
------------------

* `dynesty` is now on PyPI!

0.8.4 (2018-02-24)
------------------

* Added two new slice sampling options (`'rslice'` and `'hslice'`).

* Changed internals to allow user to access quantities during dynamic batch
  allocation. **WARNING: Breaks some aspects of backwards compatibility
  for advanced users utilizing generators.**

* Simplified parallelism options.

* Fixed a singular decomposition bug that occasionally appeared during runtime.

* Small plotting/utility improvements.

0.8.3 (2017-12-13)
------------------

* Fixed additional Python 2/3 compatibility bugs.

* Added the ability to pass user-specified custom print functions.

* Added importance reweighting.

* Small improvements to plotting utilities.

* Small changes to improve user outputs and basic functionality.

0.8.2 (2017-09-15)
------------------

* Fixed `map` bugs that broke compatibility between Python 2 and 3.

* Fixed a bug where the sampler could break during the first update from the
  unit cube when using a `pool`.

0.8.1 (2017-09-12)
------------------

* Introduced a function wrapper for `prior_transform` and `loglikelihood`
  functions to allow users to pass `args` and `kwargs`.

* Fixed a small bug that could cause bounding ellipsoids to fail.

* Introduced a stability fix to the default 
  `~dynesty.dynamicsampler.weight_function` when computing evidence-based
  weights.

0.8.0 (2017-09-08)
------------------

Initial beta release.

.. toctree::
   :hidden:
   :maxdepth: 3

   crashcourse
   overview
   quickstart
   dynamic
   errors
   examples
   faq
   references
   api
