Developers' Docs
================

Running tests
-------------

Running tests requires the ``pytest`` package, available via pip or
conda. Tests are currently not installed with the package itself, thus
can only be run from the source repository. Execute ::

    ./runtests.py

To also report code coverage (requires ``pytest-cov``)::

    ./runtests.py --cov=nestle


Building the documentation
--------------------------

Requirements are ``sphinx``, ``sphinx_rtd_theme`` and the development
version of ``sphinx_gallery``. In the ``docs`` directory, ::

    make html
