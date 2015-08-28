nestle
======

/ˈnesəl/ (rhymes with "wrestle")

Pure Python implementation of nested sampling algorithms for
evaluating Bayesian evidence.

[![Build Status](https://img.shields.io/travis/kbarbary/nestle.svg?style=flat-square)](https://travis-ci.org/kbarbary/nestle)
[![Coverage Status](http://img.shields.io/coveralls/kbarbary/nestle.svg?style=flat-square)](https://coveralls.io/r/kbarbary/nestle?branch=master)

Install
-------

```
./setup.py install
```

Requires numpy v1.6 or later. Optional requirement on scipy v0.11 or
later for multi-ellipsoidal method.


Usage
-----

```python
import numpy as np
import nestle

# Define a likelihood function
def loglikelihood(theta):
    data_x = np.array([1., 2., 3.])
    data_y = np.array([1.4, 1.7, 4.1])
    data_yerr = np.array([0.2, 0.15, 0.2])

    # theta[1] is slope and theta[0] is intercept
    y = theta[1] * data_x + theta[0]
    chisq = np.sum(((data_y - y) / data_yerr)**2)
    return -chisq / 2.


# Define a prior that takes an array of floats in the range (0, 1) and
# returns The following is equivalent to a flat prior in both m and b:
# (-5.0 < m < 5.0) and (-5.0 < b < 5.0), because it takes numbers in
# the range (0, 1) and maps them onto the range (-5.0, 5.0). Note that
# the input value will be a 1-d numpy array
def prior(x):
    return 10.0 * x - 5.0

# Run nested sampling.
res = nestle.sample(loglikelihood, prior, 2)

res.logz     # log evidence
res.logzerr  # numerical uncertainty on log evidence
res.samples  # array of sample parameters
res.weights  # array of weights
res.keys()   # list of all attributes of `res`
```


Examples
--------

Check out some example notebooks! These can also be found in the `examples`
directory.

* [Fitting a line](http://nbviewer.ipython.org/github/kbarbary/nestle/tree/master/examples/line.ipynb)
* [Eggbox likelihood](http://nbviewer.ipython.org/github/kbarbary/nestle/tree/master/examples/eggbox.ipynb)

About the Algorithms
--------------------

### Single-ellipsoid method: `method='single'`

Determines a single ellipsoid that bounds all active points, enlarges the
ellipsoid by a factor of `enlarge` in volume, and selects a new point at random
from within the ellipsoid.

### Multi-ellipsoid method: `method='multi'`

In cases where the posterior is multi-modal, the single-ellipsoid method can be
extremely inefficient: In such situations, there are clusters of active points
on separate high-likelihood regions separated by regions of lower likelihood.
Bounding all points in a single ellipsoid means that the ellipsoid includes the
lower-likelihood regions we wish to avoid sampling from.

The solution is to detect these clusters and bound them in separate ellipsoids.
For this, we use a recursive process where we perform K-means clustering with
K=2. If the resulting two ellipsoids have a significantly lower total volume
than the parent ellipsoid (less than half), we accept the split and repeat the
clustering and volume test on each of the two subset of points. This process
continues recursively. Alternatively, if the total ellipse volume is
significantly greater than expected (based on the expected density of points)
this indicates that there may be more than two clusters and that K=2 was not an
appropriate cluster division. We therefore still try to subdivide the clusters
recursively. However, we still only accept the final split into N clusters if
the total volume decrease is significant.


Run test(s)
-----------
Requires the `pytest` package to be installed.

```
./runtests.py
./runtests.py --cov=nestle  # report coverage in term (requires pytest-cov)
```

License
-------

The license is MIT. See `LICENSE.md`.

Contributors
------------

- @kbarbary
- @ipashchenko
- @RuthAngus
