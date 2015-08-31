Nestle
======

/ˈnesəl/ *(rhymes with "wrestle")*

Pure Python, MIT-licensed implementation of nested sampling algorithms for
evaluating Bayesian evidence.


Install
-------

Nestle is compatible with Python 2 and 3. Requires numpy 1.6+ and
scipy (optional). Install with pip::

    pip install nestle

For the latest development version, see http://github.com/kbarbary/nestle.


.. toctree::
   :hidden:
   :maxdepth: 2

   examples/index
   api
   devdocs

:doc:`examples/index`
---------------------


API
---

.. autosummary::
   
   nestle.sample
   nestle.print_progress
   nestle.mean_and_cov
   nestle.Result

References
----------

Feroz, Hobson, Bridges 2009, *MNRAS*, **398**, 1601

Shaw, Bridges, Hobson 2007, *MNRAS*, **378**, 1365

Skilling, J. (2004). Nested Sampling. In *Maximum entropy and Bayesian
methods in science and engineering* (ed. G. Erickson, J.T. Rychert,
C.R. Smith). *AIP Conf. Proc.*, **735**, 395-405.


See also http://www.inference.phy.cam.ac.uk/bayesys/ .
