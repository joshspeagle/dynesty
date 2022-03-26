This is an internal memo for things that need to be done for the release

* Make sure the docs are up to date with the changes
* Double check serious pylint warnings
```
pylint --fail-under=8 --extension-pkg-whitelist=scipy.special --disable=R,C
```
* Make sure the tests run (ideally for many random seeds)
```
for a in `seq 0 35` ; do env OMP_NUM_THREADS=1 DYNESTY_TEST_RANDOMSEED=$a PYTHONPATH=py:tests:$PYTHONPATH pytest -m 'not slow'  > /tmp/ulog.${a}  & done
```
* Make sure that the jupyter notebooks run with the latest release i.e. with this
```
 env OMP_NUM_THREADS=1 PYTHONPATH=py:tests:$PYTHONPATH  pytest  --workers=100 tests/test_notebooks.py
```
It may be also needed to update the actual .ipynb files in the repo (and/or look at the results)
* update the changelog
* change the internal version number
* git tag on github
* release on pypi
* update AUTHORS
* add a zenodo record
