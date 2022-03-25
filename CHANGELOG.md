# Changelog


All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [1.2.0] - XXXXX_DATE

### Added

### Changed

- More code testing + validation
- Saving likelihood
- Many speedups (ellipsoid bounds, bootstrap, jitter_run, #256, #329, )
- Pointvol fixes
- Internal refactoring reducing duplication (saved_run, integral calculations, different samplers etc)
- Make dynesty fully deterministic if random state is provided (also migrate to new numpy random generator funcitonality)
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
- make uncertainties smaller (more correct) (remove factor of two)
- fix update_interval
- exception if unknown arguments in nestedsampler/dynamic
- fix use_pool:loglikelihood
- get rid of kl_divergence f-n
- refactor the addition of batches in dynamic sampling, preventing possibly infinite loop
- improve stability of resample_equal ( #351)
