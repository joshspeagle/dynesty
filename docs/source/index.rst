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

**The release paper describing dynesty 1.0 can be found**
`here <https://github.com/joshspeagle/dynesty/tree/master/paper/dynesty.pdf>`__.

As a multi-purpose sampler, ``dynesty`` is designed to perform
"reasonably well" across a large array of problems but is not optimized for
any single one. In particular, please take caution when applying ``dynesty`` to
estimate Bayesian posteriors and evidences for large-dimensional
(>30 dimensions or so) problems.

Installation
============

``dynesty`` is compatible wit Python 3.6+. It requires
``numpy`` (for arithmetic), 
``scipy`` (for special functions), 
``matplotlib`` (for plotting), and 
While not required, ``tqdm`` also allows for a nice progress bar.

Installing the most recent stable version of the package is as easy as::

    pip install dynesty

Alternately, for users who might want newer development versions, it can also
be installed directly from a local copy of the repository by running::

    python setup.py install

Citations
=========

If you find `dynesty` useful in your research, please cite the
`software <https://doi.org/10.5281/zenodo.3348367>`_ and relevant papers.
A list of papers that you
should cite can be generated directly from the `sampler` object by calling::

    print(sampler.citations)

This will return a list of relevant papers and corresponding links to download
citation information such as BibTex files. As an example::

    import dynesty
    sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim,
                                           bound='balls', sample='rwalk')
    print(sampler.citations)

Returns the following list of papers that should be cited::

    Code and Methods:
    ================
    Speagle (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S

    Nested Sampling:
    ===============
    Skilling (2004): ui.adsabs.harvard.edu/abs/2004AIPC..735..395S
    Skilling (2006): projecteuclid.org/euclid.ba/1340370944

    Dynamic Nested Sampling:
    =======================
    Higson et al. (2019): doi.org/10.1007/s11222-018-9844-0

    Bounding Method:
    ===============
    Buchner (2016): ui.adsabs.harvard.edu/abs/2014arXiv1407.5459B
    Buchner (2017): ui.adsabs.harvard.edu/abs/2017arXiv170704476B

    Sampling Method:
    ===============
    Skilling (2006): projecteuclid.org/euclid.ba/1340370944

If you have utilized some of the error analysis features available through
the provided utility functions (see :ref:`Nested Sampling Errors`),
you should also cite
`Chopin & Robert (2010)
<http://ui.adsabs.harvard.edu/abs/2008arXiv0801.3887C>`_,
`Higson et al. (2018)
<projecteuclid.org/euclid.ba/1508897094>`_,
and `Speagle (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S>`_.

See :ref:`References and Acknowledgements` for additional details.

Changelog
=========

.. image:: ../images/logo.gif
    :align: center

1.2.1 (2022-04-04)
------------------
Small bug fix release

* The arguments of prior_transform and likelihood function are now explicitely copied, so the sampling can work if those function apply changes to argument vectors ( #362 )
* Fix the compilation of the docs, and update them a bit
	    
1.2.0 (2022-03-31)
------------------

This version has multiple changes that should improve stability and speed. The default dynamic sampling behaviour has been changed to focus on the effective number of posterior samples as opposed to KL divergence. The rstagger sampler has been removed and the default choice of the sampler may be different compared to previous releases depending on the dimensionality of the problem. dynesty should now provide 100% reproduceable results if the rstate object is provided. It needs to be a new generation Random Generator (as opposed to numpy.RandomState)

Most of the changes in the release have been contributed by [Sergey Koposov](https://github.com/segasai) who has joined the dynesty project.

* Saving likelihood. It is now possible to save likelihood calls history during sampling into HDF5 file (this is not compatible with parallel sampling yet). The relevant options are  save_history=False, history_filename=None (#235)
* add_batch() function now has the mode parameter that allows you to manually chose the logl range for the batch (#328)
* More testing with code coverage of >90% + validation on test problems
* Internal refactoring reducing code duplication (saved_run, integral calculations, different samplers etc)
* Multiple speedups: ellipsoid bounds, bootstrap, jitter_run (#239, #256, #329)
* Exception is raised if unknown arguments are provided for static/dynamic samplers (#295)

* Migrate to new numpy random generator functionality from RandomState (#280)
* Make dynesty fully deterministic if random state is provided (#292)

* Remove the pointvol parameter used in internal calculations, such as ellipsoid splits (#284)
* Get rid of vol_dec parameter (#286)
* Improve bounding ellipsoids algorithms, for example how we are dealing with degenerate ellipsoids (#264, #268)
* Introduce more stable multi-ellipsoidal splitting using BIC (#286)

* Do not use KL divergence function for stopping criteria based on posterior, instead use the criterion based on the number of effective samples. The old behaviour can still be achieved by using the dynesty.utils.old_stopping_function (#332)
* Fix bugs in dynamic sampling that can lead to sampler not finding points in the interval (#244)
* Major refactor of rslice/slice sampling increasing its stability (#269, #271)
* Disable ncdim when slice sampling (#271)
* Change the defaults for slices/walks/bootstrap (vs number of dimensions) (#297)
* Change default samplers (vs ndim, i.e. use rslice for high dimensions) (#286)
* Remove rstagger sampler, as it was producing incorrect results/based on non-Markovian chains (#322)
* Fix rwalk sampler. Previously the chains were not Markovian (#319, #323, #324)
* Change step adaptation of rwalk and rslice (#260, #323)
* Change the calculation of evidence uncertainties, remove factor the unnecessary factor two, and improve numerical stability of calculations (#306, #360)
* Refactor the addition of batches in dynamic sampling, preventing possibly infinite loop (#326)
* Improve stability of resample_equal (#351)
* New Results interface with a dedicated object, rather than a wrapper around the dictionary (#330)
 

1.1 (2021-04-05)
------------------
* Improved behavior and stability of the bounding distributions (with
  `Sergey Koposov <https://github.com/segasai>`_ and
  `Johannes Buchner <https://github.com/johannesbuchner>`_).

* Added support for specifying the number of clustering dimensions (`'ncdim'`)
  in case these may differ from the number of prior dimensions (`'npdim'`)
  (with `Colm Talbot <https://github.com/ColmTalbot>`_).

* Fixed a bug where ``dynesty`` was not properly enforcing nested sampling's
  monotonically-increasing likelihood condition when sampling
  (with `Colm Talbot <https://github.com/ColmTalbot>`_).

* Improved ability to save sampler objects to disk to backup progress (with
  `Patrick Sheehan <https://github.com/psheehan>`_ and
  `Alex Nitz <https://github.com/ahnitz>`_).

* Limited support for user-defined proposal strategies (with
  `Gregory Ashton <https://github.com/GregoryAshton>`_).

* Additional small bugfixes, references, and documentation updates.

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
  `Johannes Buchner <https://github.com/JohannesBuchner>`__).

* Random walks now behave differently near boundaries (with
  `Gregory Ashton <https://github.com/GregoryAshton>`_).

* Pickling sampler states should now work better in Python 3 (with
  `Dustin Lang <https://github.com/dstndstnr>`_.

* Doubled output errors in default approximation in line with theoretical
  expectations.

* Small bugfixes and docfixes (with
  `Patricio Cubillos <https://github.com/dstndstnr>`_).


0.9.5.3 (2019-03-29)
--------------------
* Various small bugfixes, with contributions by
  `Gregory Ashton <https://github.com/GregoryAshton>`_ and
  `Johannes Buchner <https://github.com/JohannesBuchner>`__.

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
