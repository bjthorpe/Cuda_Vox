""" Various sanity tests to check that files are read and handled correctly."""
import pytest
import cad2vox
import meshio
import os
import tifffile as tf
import numpy as np
import glob

# use the test case folder as their working directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

@pytest.fixture()
def cleanup():
    print("Starting Pytest")
    yield
    print("performing cleanup of output")
    csv_files = glob.glob('*.csv')
    for string in csv_files:
        os.remove(string)
    tiff_files = glob.glob('outputs/*.tiff')
    for string in tiff_files:
        os.remove(string)

############## Tests for setting asymmetric gridsize
def test_asym_grid(cleanup):
# the code should run given a non symetric grid
    cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_assym",gridsize = [150,120,110],Orientation='XY')
    output = tf.imread("outputs/AMAZE_assym.tiff")
    shape_test = np.shape(output)
    assert shape_test == (150,120,110)

############ tests for inputfile

def test_filename_not_string():
    # Test non-string gives error
    with pytest.raises(TypeError):
        filename = 6
        cad2vox.voxelise(filename,"AMAZE_Sample",gridsize = [100,100,100])
    assert TypeError

def test_filename_not_exist():
# give an input file that does not exist
    filename = "I-dont-exist.med"
    with pytest.raises(meshio._exceptions.ReadError):
        cad2vox.voxelise(filename,"AMAZE_Sample",gridsize = [100,100,100])
    
    assert meshio._exceptions.ReadError

def test_no_Tertahedrons(cleanup):
    # Sphere.stl only has triangle data so should fail only if we set use_tetra
    # This also techinally tests no material data as stl files dont have that.
    cad2vox.voxelise("inputs/Sphere.stl","outputs/Sphere",gridsize = [100,100,100])
    os.remove('outputs/Sphere.tiff')

def test_no_Tertahedrons_use_tetra(cleanup):
    # Sphere.stl only has triangle data so should fail if we set use_tetra
    with pytest.raises(ValueError):
        cad2vox.voxelise("inputs/Sphere.stl","outputs/Sphere",gridsize = [100,100,100],use_tetra=True)
    
    assert  ValueError

def test_no_mat_data(cleanup):
    # Sphere.med has no material data defined so should produce a tiff stack
    # containing only two values 0 and 255.
    cad2vox.voxelise("inputs/Sphere.med","outputs/Sphere_nomats",gridsize = [100,100,100])
    # read back in the output as an np array
    output = tf.imread('outputs/Sphere_nomats.tiff')
    os.remove('outputs/Sphere_nomats.tiff')
    print(np.unique(output))
    #check the tiff stack contains only 0 and 255
    assert np.all([np.any(output == value) for value in np.sort([0,255])])

############## Tests for Greyscale files
def test_greyscale_not_exist(cleanup):
# give a greyscale file that does not exist the code should create it
    cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/Sphere",
    greyscale_file="outputs/I-dont-exist.csv",gridsize = [100,100,100])
    assert os.path.exists("outputs/I-dont-exist.csv")

def test_generate_greyscale_file(cleanup):
# test if greyscale.csv file is auto-generated if None given

    cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_Sample",gridsize = [100,100,100])
    os.remove('outputs/AMAZE_Sample.tiff')
    assert os.path.exists("greyscale.csv")

def test_greyscale_non_8_bit():
    filename = "inputs/invalid_greyscale_range.csv"
    # one greyscale value in this file exceeds 255 so is not a valid 8 bit number
    with pytest.raises(ValueError):
        cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/Sphere",
        greyscale_file=filename,gridsize = [100,100,100])
    
    assert  ValueError


def test_vaild_greyscale_float(cleanup):
    filename = "inputs/valid_greyscale_float.csv"
    # greyscale values in this file are floats which are valid as they can be cast as an int
    cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_Sample",
    greyscale_file=filename,gridsize = [100,100,100])
    os.remove('outputs/AMAZE_Sample.tiff')
 
def test_invalid_greyscale_string():
    filename = "inputs/invalid_greyscale_string.csv"
    # one greyscale value in this file is the string 25o which is
    # invalid as it can't be cast as an int
    with pytest.raises(ValueError):
        cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/Sphere",
        greyscale_file=filename,gridsize = [100,100,100])
    
    assert  ValueError
