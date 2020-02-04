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
While not required, ``tqdm`` also allows for a nice progress bar.

Installing the most recent stable version of the package is as easy as::

    pip install dynesty

Alternately, for users who might want newer development versions, it can also
be installed directly from a local copy of the repository by running::

    python setup.py install

Citations
=========

If you find `dynesty` useful in your research, a list of papers that you
should cite can be generated directly from the `sampler` object by calling::

    print(sampler.citations)

This will return a list of relevant papers and corresponding links to download
citation information such as BibTex files.

This list will by default include the following papers:

* Code:
  `Speagle (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190402180S>`_.

* Nested Sampling:
  `Skilling (2004) <http://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S>`_
  and `Skilling (2006) <https://projecteuclid.org/euclid.ba/1340370944>`_.

If you use the Dynamic Nested Sampling functionality
(via `DynamicNestedSampler`), this will also include:

* Dynamic Nested Sampling:
  `Higson et al. (2017b)
  <http://ui.adsabs.harvard.edu/abs/2017arXiv170403459H>`_.

Depending on your specific bounding and sampling options, this may also include
the following papers:

* Single ellipsoid bound:
  `Mukherjee, Parkinson & Liddle (2006)
  <http://ui.adsabs.harvard.edu/abs/2006ApJ...638L..51M>`_.

* Multiple ellipsoid bounds:
  `Feroz, Hobson & Bridges (2009)
  <http://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F>`_.

* Overlapping balls/cubes:
  `Buchner (2016) <http://ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B>`_ and
  `Buchner (2017) <https://ui.adsabs.harvard.edu/abs/2017arXiv170704476B>`_.

* Random walks/staggers:
  `Skilling (2006) <https://projecteuclid.org/euclid.ba/1340370944>`_.

* Multivariate/Random slice sampling:
  `Neal (2003) <https://projecteuclid.org/euclid.aos/1056562461>`_,
  `Handley, Hobson & Lasenby (2015a)
  <http://ui.adsabs.harvard.edu/abs/2015MNRAS.450L..61H>`_, and
  `Handley, Hobson & Lasenby (2015b)
  <http://ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H>`_.

* Hamiltonian/Reflective slice sampling:
  `Neal (2003) <https://projecteuclid.org/euclid.aos/1056562461>`_,
  `Skilling (2012) <https://aip.scitation.org/doi/abs/10.1063/1.3703630>`_,
  `Feroz & Skilling (2013)
  <https://ui.adsabs.harvard.edu/abs/2013AIPC.1553..106F>`_, and
  `Speagle (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190402180S>`_.

Finally, if you have utilized some of the error analysis features through
the provided utility functions, you should also cite: 

* Nested Sampling error analysis:
  `Chopin & Robert (2010)
  <http://ui.adsabs.harvard.edu/abs/2008arXiv0801.3887C>`_,
  `Higson et al. (2017a)
  <http://ui.adsabs.harvard.edu/abs/2017arXiv170309701H>`_,
  and `Speagle (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190402180S>`_.

See :ref:`References and Acknowledgements` for additional details.

Changelog
=========

.. image:: ../images/logo.gif
    :align: center

1.0.1 (2020-01-17)
-------------------
* Small quality-of-life improvements to plotting.

* Added citation tool.

1.0.0 (2019-09-22)
-------------------
* Added support for period and reflective boundaries (with
  `Gregory Ashton <https://github.com/GregoryAshton>`_).

* Added support for interactive progress bar (with
  `Daniel Foreman-Mackey <https://github.com/dfm>`_).

* Added support for stopping criterion based on ESS (with
  `Colm Talbot <https://github.com/ColmTalbot>`_).

* Small bugfixes to code and documentation.

* Small quality-of-life improvements.

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
