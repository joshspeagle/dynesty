# Changelog

All notable changes to dynesty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [1.2.0] - XXXXX_DATE

This version has multiple changes that should improve stability and speed.
Specifically the default dynamic sampling behaviour has been changed to
focus on the effective number of posterior samples as opposed to KL divergence.
The samplers such as rstagger has been removed and the default choice of the sampler may be different depending on the dimensionality of the problem.


### Added

- Saving likelihood. It is now possible to save likelihood calls values during sampling. (it is not compatible with parallel sampling yet) ( #235 )
- add_batch() function now has the mode parameter that allows you to manually chose the logl range for the batch (#328)

### Changed

- More code testing with code coverage of >85% + validation
- Many speedups (ellipsoid bounds, bootstrap, jitter_run, #239, #256, #329, )
- do not use KL divergence function for stopping, instead use the criterion based on the number of effective samples. The old behaviour can still be achieved by using the dynesty.utils.old_stopping_function ( #332 )
- Remove the pointvol parameter used in internal calculations, such as ellipsoid splits (#284)
- Internal refactoring reducing code duplication (saved_run, integral calculations, different samplers etc)
- Make dynesty fully deterministic if random state is provided (#292)
- Migrate to new numpy random generator functionality from RandomState (#280)
- Fix bugs in dynamic sampling that leads to sampler not finding points in the interval ( dynamic sampler bug, incorrect logl interval #244)
- Improve bounding ellipsoids algorithms, for example how we are dealing with degenerate ellipsoids #264, #268
- Major refactor of rslice/slice sampling increasing its stability (#269, #271)
- disable ncdim when slice sampling ( #271 )
- change the defaults for slices/walks/bootstrap (vs number of dimensions) (#297)
- change default samplers (vs ndim, i.e. use rslice for high dimensions) (#286)
- remove rstagger sampler (#322)
- fix rwalk sampler. Previously the chains were not Markovian (#319, #323, #324)
- change step adaptation of rwalk and rslice (#260, #323)
- get rid of vol_dec parameter ( #286 )
- more stable multi-ellipsoidal splitting using BIC (#286)
- change the calculation of evidence uncertainties, remove factor the unnecessary factor two (#306)
- exception is raised if unknown arguments are provided for nestedsampler/dynamic samplers ( #295 )
- refactor the addition of batches in dynamic sampling, preventing possibly infinite loop ( #326)
- improve stability of resample_equal ( #351)
- new Results interface (#330)
