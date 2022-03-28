# Changelog

All notable changes to dynesty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [1.2.0] - XXXXX_DATE

This version has multiple changes that should improve stability and speed. The default dynamic sampling behaviour has been changed to focus on the effective number of posterior samples as opposed to KL divergence. The rstagger sampler has been removed and the default choice of the sampler may be different compared to previous releases depending on the dimensionality of the problem. dynesty should now provide 100% reproduceable results if the rstate object is provided. It needs to be a new generation Random Generator (as opposed to numpy.RandomState)

### Added

- Saving likelihood. It is now possible to save likelihood calls values during sampling. (it is not compatible with parallel sampling yet) ( #235 )
- add_batch() function now has the mode parameter that allows you to manually chose the logl range for the batch (#328)

### Changed

- More testing with code coverage of >87% + validation on test problems
- Internal refactoring reducing code duplication (saved_run, integral calculations, different samplers etc)
- Multiple speedups (ellipsoid bounds, bootstrap, jitter_run, #239, #256, #329)
- Exception is raised if unknown arguments are provided for nestedsampler/dynamic samplers ( #295 )

- Migrate to new numpy random generator functionality from RandomState (#280)
- Make dynesty fully deterministic if random state is provided (#292)

- Remove the pointvol parameter used in internal calculations, such as ellipsoid splits (#284)
- Get rid of vol_dec parameter ( #286 )
- Improve bounding ellipsoids algorithms, for example how we are dealing with degenerate ellipsoids #264, #268
- Introduce more stable multi-ellipsoidal splitting using BIC (#286)

- Do not use KL divergence function for stopping criteria based on posterior, instead use the criterion based on the number of effective samples. The old behaviour can still be achieved by using the dynesty.utils.old_stopping_function (#332)
- Fix bugs in dynamic sampling that can lead to sampler not finding points in the interval (#244)
- Major refactor of rslice/slice sampling increasing its stability (#269, #271)
- Disable ncdim when slice sampling ( #271 )
- Change the defaults for slices/walks/bootstrap (vs number of dimensions) (#297)
- Change default samplers (vs ndim, i.e. use rslice for high dimensions) (#286)
- Remove rstagger sampler, as it was producing incorrect results/based on non-Markovian chains (#322)
- Fix rwalk sampler. Previously the chains were not Markovian (#319, #323, #324)
- Change step adaptation of rwalk and rslice (#260, #323)
- Change the calculation of evidence uncertainties, remove factor the unnecessary factor two, and improve numerical stability of calculations (#306, #360)
- Refactor the addition of batches in dynamic sampling, preventing possibly infinite loop ( #326)
- Improve stability of resample_equal ( #351)
- New Results interface with a dedicated object, rather than a wrapper around the dictionary (#330)
