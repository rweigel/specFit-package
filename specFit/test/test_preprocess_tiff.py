import os
import pytest
import numpy as np

Nfiles = [3, 4, 255, 256]
Nx = 7
Ny = 7
spectra_type = 1

def raw_dir(Nfiles):
    return os.path.join('tmp', 'images', 'raw', str(Nfiles))


def processed_dir(Nfiles):
    return os.path.join('tmp', 'images', 'processed', str(Nfiles))


@pytest.mark.parametrize("Nfiles", Nfiles)
def test_file_creation(Nfiles):
    """Test files needed by specFit have been created"""
    command = ('python specFit/demo/demo_preprocess_tiff.py '
               '--processed_dir {} --raw_dir {} --Nfiles {} --Nx {} --Ny {} --spectra_type {}')\
               .format(processed_dir(Nfiles), raw_dir(Nfiles), Nfiles, Nx, Ny, spectra_type)
    os.system(command)
    assert os.path.exists(os.path.join(processed_dir(Nfiles), 'dataCube.npy'))
    assert os.path.exists(os.path.join(processed_dir(Nfiles), 'timestamps.npy'))
    assert os.path.exists(os.path.join(processed_dir(Nfiles), 'exposures.npy'))
    assert os.path.exists(os.path.join(processed_dir(Nfiles), 'visual.npy'))


@pytest.mark.parametrize("Nfiles", Nfiles)
def test_file_dimensions(Nfiles):
    """Test files needed by specFit have correct dimensions"""
    path = os.path.join(processed_dir(Nfiles), 'dataCube.npy')
    cube_shape = np.load(path).shape
    ts_size = np.load(os.path.join(processed_dir(Nfiles), 'timestamps.npy')).size
    exp_size = np.load(os.path.join(processed_dir(Nfiles), 'exposures.npy')).size
    assert ts_size == exp_size == cube_shape[2] == Nfiles


@pytest.mark.parametrize("Nfiles", Nfiles)
def test_periodic(Nfiles):
    """Test that image intensities have periodic features."""
    data = np.load(os.path.join(processed_dir(Nfiles), 'dataCube.npy'))[0, 0]
    ts = np.linspace(0, Nfiles-1, Nfiles)
    sin = 128*np.sin(4*np.pi*ts/Nfiles) + 127
    assert (data - sin).sum() <= Nfiles
    

#def test_average():
    # TODO: Test that visual average is zero?
