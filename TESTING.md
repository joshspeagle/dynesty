
The testing suite is executed using pytest with all the tests inside the tests/
folder

A few notes

* Slow vs fast tests
The majority of tests are designed to take together ~ 30 min of wall time, but a few tests take a long time. To exclude the slow tests you can run the pytest as

```
pytest -m 'not slow'
```

* Random seeds
By default the testing suite uses the same random seed for all the tests to ensure the consistency. It is occasionally useful to run with different seeds. For this you can set the environment variable DYNESTY_RANDOM_SEED to any integer value.
If you are writing a new test you need to use the the get_rstate() function (from tests/utils.py)


* Test printing.
By default none of the tests print any output. You can enable the printing of progress by setting the environment variable DYNESTY_TEST_PRINTING=1 and use -s flag of pytest

* jupyter notebooks tests
To test that the jupyter notebooks run, you can use the tests/test_notebooks.py test
