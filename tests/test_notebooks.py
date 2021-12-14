import pytest
import glob
import os
import re


@pytest.mark.slow
@pytest.mark.parametrize("nb", glob.glob('demos/*nb'))
def test_notebooks(nb):
    """ Test that notebooks run fine """
    if re.match('.*nbconvert.*', nb) is not None:
        # nbconvert leaves files like nbconvert.ipynb
        # we don't want to run on those
        return
    cmd = f'jupyter nbconvert --to notebook --execute "{nb}"'
    stat = os.system(cmd)
    assert stat == 0
