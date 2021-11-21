import pytest
import glob
import os


@pytest.mark.slow
@pytest.mark.parametrize("nb", glob.glob('demos/*nb'))
def test_notebooks(nb):
    """ Test that notebooks run fine """
    cmd = f'jupyter nbconvert --to notebook --execute "{nb}"'
    stat = os.system(cmd)
    assert stat == 0
