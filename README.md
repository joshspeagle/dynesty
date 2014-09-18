nestle
======

*/ˈnesəl/*

Pure Python implementation of nested sampling algorithms for
evaluating Bayesian evidence.


Install
-------

```
./setup.py install
```


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
res = nestle.nest(loglikelihood, prior, 2)

res.logz  # log evidence
res.logzerr  # numerical uncertainty on log evidence
res.samples  # array of sample parameters
res.weights  # array of weights
res.keys()   # list of all attributes of `res`
```

View the docstring:

```python
import nestle
help(nestle.nest)
```

Notes
-----

This is a single-ellipsoid implementation of nested sampling. It
works, but it can be very slow when the likelihood surface is
multimodal.

Implementation of multimodal nested sampling algorithm is in progress.
The goal:

1. A pure-python (numpy-based) implementation of the algorithm.
2. If necessary for performance, speed up with cython.

License
-------

BSD.

Contributors
------------

@ipashchenko