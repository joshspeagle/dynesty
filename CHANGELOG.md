# Changelog


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [1.2.0] - XXXXX_DATE

This version has multiple changes that should improve stability and speed.
Specifically the default dynamic sampling behaviour has been changed to
focus on the effective number of posterior samples as opposed to KL divergence.
The samplers such as rstagger has been removed and the default choice of the sampler may be different depending on the dimensionality of the problem.


### Added

### Changed

- More code testing + validation
- Saving likelihood
- Many speedups (ellipsoid bounds, bootstrap, jitter_run, #239, #256, #329, )
- Remove the pointvol parameter used in internal calculations, such as ellipsoid splits (#284)
- Internal refactoring reducing duplication (saved_run, integral calculations, different samplers etc)
- Make dynesty fully deterministic if random state is provided (#280)
- Migrate to new numpy random generator functionality
- Fix bugs in dynamic sampling that leads to sampler not finding points in the interval ( dynamic sampler bug, incorrect logl interval #244)
- Improve bounding ellipsoids algorithms Possible improvement to the way we are dealing with degenerate ellipsoids #264, Tiny improvement to #264 #268
- Major refactor of rslice/slice sampling increasing its stability (#269, #271)
- disable ncdim when slice sampling
- change the defaults for slices/walks/bootstrap (vs ndim)
- change default samplers (vs ndim, i.e. use rslice for highdim)
- remove rstagger sampler (#322)
- fix rwalk. Before the chains were not markovian (#319, #323, #324)
- change adaptation of rwalk and rslice
- get rid of vol_dec parameter ( #286 )
- more stable multi-ellipsoidal splitting using BIC 
- make uncertainties smaller (more correct; i.e. remove factor of two in the calculations of evidence uncertainty #306)
- fix update_interval
- exception is raised if unknown arguments are provided for nestedsampler/dynamic samplers
- fix use_pool:loglikelihood
- do not use  kl_divergence f-n for stopping ( #332 )
- refactor the addition of batches in dynamic sampling, preventing possibly infinite loop
- improve stability of resample_equal ( #351)
- new results interface (#330)
- add_batch() function now has the mode parameter that allows you to manually chose the logl range for the batch (#328) 
