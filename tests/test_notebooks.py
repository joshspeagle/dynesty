import pytest
import glob
import os
import re


@pytest.mark.slow
@pytest.mark.parametrize("nb", glob.glob('demos/*.ipynb'))
def test_notebooks(nb):
    """ Test that notebooks run fine """
    if re.match('.*nbconvert.*', nb) is not None:
        # nbconvert leaves files like nbconvert.ipynb
        # we don't want to run on those
        return

    # Here I'm trying to determine if the converted nb exists
    # if it does not I'll try to delete it after we are done

    postfl = 6
    assert nb[-postfl:] == '.ipynb'
    fname_converted = nb[:-postfl] + '.nbconvert.ipynb'
    if not os.path.exists(fname_converted):
        delete = True
    else:
        delete = False

    try:
        cmd = f'jupyter nbconvert --to notebook --execute "{nb}"'
        stat = os.system(cmd)
        assert stat == 0
    finally:
        if delete:
            try:
                os.unlink(fname_converted)
            except:
                pass
