The testing suite is executed using pytest with all the tests inside the `tests/` folder

A few notes on running and writing tests

### Requirements
The test suite makes use of packages not included in `requirements.txt`. To install these additional packages, use

```
pip install -r requirements-dev.txt
```

### Slow vs fast tests
The majority of tests are designed to take together ~ 30 min of wall time, but a few tests take a long time. To exclude the slow tests you can run the pytest as

```
pytest -m 'not slow'
```
Long tests include notebook tests and some high-dimensional validation tests

### Testing in parallel
If you have many CPUs, you can run the tests in parallel using
```
pytest  -m 'not slow' --dist loadgroup  -n 10
```
(if you have 10 CPUs). This way the test suite should finish in a few minutes.

### Random seeds
By default the testing suite uses the same random seed for all the tests to ensure the consistency. It is occasionally useful to run with different seeds. For this you can set the environment variable `DYNESTY_RANDOM_SEED` to any integer value.
If you are writing a new test you need to use the the `get_rstate()` function (from `tests/utils.py`)


### Test printing
By default most of the tests will not print any output. You can enable the printing of progress by setting the environment variable `DYNESTY_TEST_PRINTING=1` and use `-s` flag of pytest

### Jupyter notebooks tests
To test that the Jupyter notebooks run, you can run the `tests/test_notebooks.py` test. These tests rely on dynesty 
being installed in the Python environment referred to by the default `python3` Jupyter kernel.
