""" Package to provide user interface to the CudaVox package."""
import csv
from os.path import exists
import errno
import os
from unittest.mock import NonCallableMock
from PIL import Image, ImageOps
import numpy as np
import tifffile as tf
import meshio
from CudaVox import run
import pandas as pd
from .utill import check_greyscale, find_the_key, check_voxinfo, padding, check_padding


def voxelise(input_file, output_file, **kwargs):
    """

    Wrapper Function to setup the CudaVox python bindings for the C++ code and provide the main user
    interface.

    This function will first try to perform the voxelisation using a CUDA capable GPU. If that fails
    or CUDA is unavailable it will fallback to running on CPU with the maximum number of available
    threads.

    Parameters:
    :param input_file: (string): Hopefully self explanatory, Our recommended (i.e. tested) format is Salome
    med. However, theoretically any of the approx. 30 file formats supported by meshio will
    work. Provided they are using either tetrahedrons or triangles as there element type
    (see https://github.com/nschloe/meshio for the full list).

    :param output_file: (string): Filename for output as 8 bit greyscale images. Note do not include the
    extension as it will automatically be appended based on the requested image format.
    The default is a virtual tiff stack other formats can be selected with the im_format option.

    :param gridsize: (list of 3 +ve non-zero ints): Number of voxels in each axis orientated as [x,y,z]
    the resulting output will be a series of z images with x by y pixels.

    :param unit_length: (list of 3 +ve non-zero floats): size of each voxel in mesh co-ordinate space.

    *****************************************************************************************
      Note: Since one is caculated from the other you can only set one of unit_length or
      Gridsize. If you set a Gridsize but do not set unit_length it will calculate unit length
      for you with the image boundaries based on the max and min of the mesh.

      Similarly, if you define unit_length but not GridSize it will calculate the number of
      voxels in each dimension for you, again with the image boundaries based on max and min
      of the mesh.
    *****************************************************************************************

    Optional kwargs:

    :param greyscale_file: (string/None): csv file for defining custom Greyscale values. If not given the
    code evenly distributes greyscale values from 0 to 255 across all materials defined in the
    input file. It also auto-generates a file 'greyscale.csv' with the correct formatting which
    you can then tweak to your liking.

    :param use_tetra: (bool): flag to specifically use Tetrahedrons instead of Triangles. This only applies
    in the event that you have multiple element types defined in the same file. Normally the code
    defaults to triangles however this flag overrides that.

    :param cpu: (bool): Flag to ignore any CUDA capable GPUS and instead use the OpenMp implementation.
    By default the code will first check for GPUS and only use OpenMP as a fallback. This flag
    overrides that and forces the use of OpenMP. Note: if you wish to use CPU permanently,
    as noted in the build docs, you can safely compile CudaVox without CUDA in which case the code
    simply skips the CUDA check altogether and permanently runs on CPU.

    :param Solid: (bool): This Flag can be set if you want to auto-fill the interior when using a Surface
    Mesh (only applies to Triangles). If you intend to use this functionality there are three
    Caveats to briefly note here:

    1) This flag will be ignored if you only supply Tetrahedron data or set use_tetra since in
    both cases that is by definition not a surface mesh.

    2) The algorithm currently used is considerably slower and not robust (can lead to artifacts and
    holes in complex meshes).

    3) Setting this flag turns off greyscale values (background becomes 0 and the mesh becomes 255).
    This is because we dont have any data as to what materials are inside the mesh so this seems a
    sensible default.

    The only reason 2 and 3 exist is because this functionality is not actively being used by our
    team so there has been no pressing need to fix them. However, if any of these become an
    issue either message b.j.thorpe@swansea.ac.uk or raise an issue on git repo as they can easily
    be fixed and incorporated into a future release.

    :param im_format: (string): The default output is a virtual Tiff stack. This option however, when set
    allows you to output each slice as a separate image in any format supported by Pillow
    (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for the full
    list). Simply specify the format you require as a sting e.g. im_format="png".

    Note: by default you will get a series of z images in the x-y plane. However, you can also control
    the slice orientation with the "orientation" option.

    :param Num_Threads: number of threads used by OMP in cpu calculations.  Note: this is ignored if cpu flag is
    set to false and a suitable cuda capable gpu is detected.

    :param Orientation: (String): String to define the orientation of the output images when using pillow.
    Default is "XY" must be one of "XY","XZ" or "YZ".

    :param Output_Resolution: (list of 3 +ve non-zero ints x, y, z) used for creating an output image
    of a specific size. Normally the image boundaries and final resolution are set by the
    gridsize/unit_length and the boundaries of the mesh. This option allows you to evenly pad the
    voxelised image with zeros to match any resolution desired keeping the mesh in the centre.
    Note: If the specified resolution is smaller than the gridisize in any dimension it will
    raise a type error. Default is None.
    """
    # get Kwargs if set or use default values
    gridsize = kwargs.get("gridsize", [0, 0, 0])
    unit_length = kwargs.get("unit_length", [0.0, 0.0, 0.0])
    greyscale_file = kwargs.get("greyscale_file", None)
    use_tetra = kwargs.get("use_tetra", False)
    cpu = kwargs.get("cpu", False)
    solid = kwargs.get("solid", False)
    im_format = kwargs.get("im_format", None)
    Num_Threads = kwargs.get("Num_threads", None)
    Bbox_Centre = kwargs.get("Bbox_Centre", "mesh")
    Orientation = kwargs.get("Orientation", "XY")
    Output_Resolution = kwargs.get("Output_Resolution", None)

    # read in data from file
    input_file = os.path.abspath(input_file)
    mesh = meshio.read(input_file)

    # extract np arrays of mesh data from meshio
    points = mesh.points
    triangles = mesh.get_cells_type("triangle")
    tetra = mesh.get_cells_type("tetra")

    if not np.any(triangles) and not np.any(tetra):
        raise ValueError("Input file must contain one of either Tets or Triangles")

    if not np.any(triangles) and not use_tetra:
        # no triangle data but trying to use triangles
        raise ValueError(
            "User asked to use triangles but input file does "
            "not contain Triangle data"
        )

    if not np.any(tetra) and use_tetra:
        # no tetra data but trying to use tets
        raise ValueError(
            "User asked to use tets but file does not contain Tetrahedron data"
        )

    # extract dict of material names and integer tags
    try:
        all_mat_tags = mesh.cell_tags
    except AttributeError:
        all_mat_tags = {}
    if not all_mat_tags:
        print(
            "[WARN] No materials defined in input file so using default greyscale values."
        )
        mat_tag_dict = {0: ["Un-Defined"]}
        if use_tetra:
            mat_ids = np.zeros(np.shape(tetra), dtype=int)
        else:
            mat_ids = np.zeros(np.shape(triangles), dtype=int)
    else:
        # pull the dictionary containing material id's for the elemnt type (either triangles or tets)
        # and the np array of ints that label the material in each element.
        if use_tetra:
            mat_ids = mesh.get_cell_data("cell_tags", "tetra")
            if np.any(mat_ids == 0):
                all_mat_tags[0] = ["Un-Defined"]
            mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))
        else:
            mat_ids = mesh.get_cell_data("cell_tags", "triangle")
            if np.any(mat_ids == 0):
                all_mat_tags[0] = ["Un-Defined"]
            mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))

    if greyscale_file is None:
        greyscale_file = "greyscale.csv"

    greyscale_file = os.path.abspath(greyscale_file)

    if os.path.exists(greyscale_file):
        greyscale_array = read_greyscale_file(greyscale_file, mat_ids)
    else:
        greyscale_array = generate_greyscale(greyscale_file, mat_tag_dict, mat_ids)

    # define boundray box for mesh
    mesh_min_corner = np.array(
        [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
    )
    mesh_max_corner = np.array(
        [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]
    )

    # check the values that have been defined by the user
    vox_dict = {
        "gridsize": gridsize,
        "unit_length": unit_length,
        "mesh_min": mesh_min_corner,
        "mesh_max": mesh_max_corner,
    }
    vox_info = check_voxinfo(**vox_dict)
    # Set number of OMP Thread if needed
    if Num_Threads != None:
        os.environ["OMP_NUM_THREADS"] = Num_Threads
    # call c++ library to perform the voxelisation
    vox = (
        run(
            Triangles=triangles,
            Tetra=tetra,
            Greyscale=greyscale_array,
            Points=points,
            Bbox_min=vox_info["Bbox_min"],
            Bbox_max=vox_info["Bbox_max"],
            solid=solid,
            gridsize=vox_info["gridsize"],
            use_tetra=use_tetra,
            forceCPU=cpu,
        )
    ).astype("uint8")
    if Output_Resolution != None:
        check_padding(gridsize, Output_Resolution)
        print(
            f"padding final output image to x = {Output_Resolution[0]}, "
            f"y = {Output_Resolution[1]}, z={Output_Resolution[2]}"
        )
        vox = padding(
            vox, Output_Resolution[0], Output_Resolution[1], Output_Resolution[2]
        )
    write_image(output_file, vox, im_format, Orientation)
    # write resultant 3D NP array as tiff stack


def write_image(output_file, vox, im_format=None, Orientation="XY"):
    print(np.shape(vox))
    if im_format:
        if Orientation == "XY":
            for I in range(0, np.shape(vox)[2]):
                im = Image.fromarray(vox[:, :, I])
                im = ImageOps.grayscale(im)
                im_output = "{}_{}.{}".format(output_file, I, im_format)
                im.save(im_output)
        elif Orientation == "XZ":
            for I in range(0, np.shape(vox)[1]):
                im = Image.fromarray(vox[:, I, :])
                im = ImageOps.grayscale(im)
                im_output = "{}_{}.{}".format(output_file, I, im_format)
                im.save(im_output)
        elif Orientation == "YZ":
            for I in range(0, np.shape(vox)[0]):
                im = Image.fromarray(vox[I, :, :])
                im = ImageOps.grayscale(im)
                im_output = "{}_{}.{}".format(output_file, I, im_format)
                im.save(im_output)
        else:
            raise ValueError(
                f'Unknown Orientation {Orientation} must be one of "XY", "XZ" or "YZ".'
            )
    else:
        im_output = "{}.tiff".format(output_file)
        if Orientation == "XY":
                tf.imwrite(im_output,vox,bigtiff=True, append=True)
        elif Orientation == "XZ":
               vox = np.rot90(vox)
               tf.imwrite(im_output,vox,bigtiff=True, append=True)
        elif Orientation == "YZ":
            with tf.TiffWriter(im_output) as f:
                for I in range(0, np.shape(vox)[0]):
                    f.write(vox[I, :, :], contiguous=True, photometric="minisblack")
        else:
            raise ValueError(
                f'Unknown Orientation {Orientation} must be one of "XY", "XZ" or "YZ".'
            )
    return


def generate_greyscale(greyscale_file, mat_tags, mat_ids):
    """Function to generate Greyscale values if none are defined"""
    # create list of tags and greyscale values for each material used
    mat_index = list(mat_tags.keys())
    mat_names = list(mat_tags.values())
    num_mats = len(mat_names)
    greyscale_values = np.linspace(
        255 / num_mats, 255, endpoint=True, num=num_mats
    ).astype(int)
    print("writing greyscale values to " + greyscale_file)
    with open(greyscale_file, "w", encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Material Name", "index", "Greyscale Value"])
        for i, names in enumerate(mat_names):
            row = [" ".join(names), mat_index[i], greyscale_values[i]]
            writer.writerow(row)

    # replace the material tag in the array its the integer greyscale value
    for i, tag in enumerate(mat_index):
        mat_ids[mat_ids == tag] = greyscale_values[i]

    return mat_ids


def read_greyscale_file(greyscale_file, mat_ids):
    """Function to Read Greyscale values from file if a file is defined by the user."""
    greyscale_file = os.path.abspath(greyscale_file)

    if not exists(greyscale_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), greyscale_file)

    print("reading greyscale values from " + greyscale_file)
    df = pd.read_csv(greyscale_file)

    for i, row in enumerate(df["Greyscale Value"]):
        # checking the data that is being read in
        check_greyscale(row)
    mat_index = df["index"].values
    greyscale_values = df["Greyscale Value"].values

    # replace the material tag in the array with its the integer greyscale value
    for i, tag in enumerate(mat_index):
        mat_ids[mat_ids == tag] = greyscale_values[i]

    return mat_ids
