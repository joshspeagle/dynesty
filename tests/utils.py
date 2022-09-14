import numpy as np
import os
'''
Here we setup a common seed for all the tests
But we also allow to set the seed through DYNESTY_TEST_RANDOMSEED
environment variable.
That allows to run long tests by looping over seed value to catch
potentially rare behaviour
'''


def get_rstate(seed=None):
    if seed is None:
        kw = 'DYNESTY_TEST_RANDOMSEED'
        if kw in os.environ:
            seed = int(os.environ[kw])
        else:
            seed = 56432
        # seed the random number generator
    return np.random.default_rng(seed)


def get_printing():
    kw = 'DYNESTY_TEST_PRINTING'
    if kw in os.environ:
        return int(os.environ[kw])
    else:
        return False


class NullContextManager(object):
    # https://stackoverflow.com/questions/45187286/how-do-i-write-a-null-no-op-contextmanager-in-python
    # this is to make it work for 3.6
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass
