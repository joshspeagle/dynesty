nestpy
======

Python implementation of nested sampling algorithms for evaluating Bayesian
evidence.

nest
----

This is a single-ellipsoid implementation. It works, but it can be very slow
when the likelihood surface is multimodal.


mnest
-----

Implementation of multimodal nested sampling algorithm. Currently, this
is a work in progress. The goal:

1. A pure-python (numpy-based) implementation of the algorithm.
2. If necessary for performance, speed up with cython.

The goal is to keep this in a single python module that can be dropped into
any application.
