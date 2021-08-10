import os
import numpy as np
import h5py
import sys
import importlib

__all__ = [
    "read_h5",
    "read_tileset",
    "to_symbol_string",
    "to_symbol_array",
]

"""
Functions and utilities corresponding to filenames and input/output. The convention which we hope others
will abide is to create a .h5 file for each symmetry type for each equation. Within each file, each h5py.Group
represents an orbit, which in turn contains the state information, parameters, and perhaps additional data (which
would require overloading the Orbit.to_h5() method). Therefore, general import statements only require the
filename and the group names, as the group entries for each orbit should be uniform. The downside of this is
that currently there is no query method for the HDF5 files built in to orbithunter currently. In order to query
an h5 file, it is required to use the h5py API explicitly.   
"""


def read_h5(filename, *datanames, validate=False, **orbitkwargs):
    """
    Parameters
    ----------
    filename : str or Path
        Absolute or relative path to .h5 file
    datanames : str or tuple, optional
        Names of either h5py.Datasets or h5py.Groups within .h5 file. Recursively returns all orbits (h5py.Datasets)
        associated with all names provided. If nothing provided, return all datasets in file.
    validate : bool
        Whether or not to access Orbit().preprocess a method which checks the 'integrity' of the imported data;
        in terms of its status as a solution to its equations, NOT the actual file integrity.
    orbitkwargs : dict
        Any additional keyword arguments relevant for construction of specified Orbit instances. .

    Returns
    -------
    Orbit or list of Orbits or list of list of Orbits
        The imported data; If a single h5py.Dataset's name is specified, an Orbit is returned.
        If multiple Datasets are specified, a list of Orbits are returned.
        If a h5py.Group name is specified, then a list of Orbits is returned.
        If a combination of h5py.Dataset and h5py.Group names are provided, then the result is a list interspersed
        with either Orbits or lists of Orbits, arranged in the order based on the provided names.

    Notes
    -----
    The 'state' data should be saved as a dataset. The other attributes which define an Orbit,
    which are required for expected output are 'basis', 'class', 'parameters'; all attributes included by
    default are 'discretization' (state shape in physical basis, not necessarily the shape of the saved state).

    This searches through provided h5py.Groups recursively to extract all datasets. If you need to distinguish
    between two groups which are in the same parent group, both names must be provided separately.

    As it takes a deliberate effort, keyword arguments passed to read_h5 are favored over saved attributes
    This allows for control in case of incorrect attributes; the dictionary update avoids sending more than
    one value to the same keyword argument.
    This passes all saved attributes, tuple or None for parameters, and any additional keyword
    arguments to the class

    """

    # unpack tuples so providing multiple strings or strings in a tuple yield same results.
    if len(datanames) == 1 and isinstance(*datanames, tuple):
        datanames = tuple(*datanames)

    datasets = []
    imported_orbits = []

    # # This SHOULD avoid circular imports yet still provide a resource to retrieve class constructors.
    if "orbithunter" not in sys.modules:
        module = importlib.import_module("orbithunter")
    else:
        module = sys.modules["orbithunter"]

    # With orbit_names now correctly instantiated as an iterable, can open file and iterate.
    with h5py.File(os.path.abspath(filename), "r") as file:
        # define visititems() function here to use variables in current namespace
        def parse_datasets(h5name, h5obj):
            # Orbits are stored as h5py.Dataset(s) . Collections or orbits are h5py.Group(s).
            if isinstance(h5obj, h5py.Dataset):
                groupsets.append(h5obj)

        # iterate through all names provided, extract all datasets from groups provided.
        # If no Dataset/Group names were provided, iterate through the entire file.
        for name in datanames or file:
            if isinstance(file[name], h5py.Group):
                groupsets = []
                file[name].visititems(parse_datasets)
                datasets.append(groupsets)
            elif isinstance(file[name], h5py.Dataset):
                datasets.append([file[name]])

        for orbit_collection in datasets:
            orbit_group = []
            for obj in orbit_collection:
                # Get the class from metadata
                class_ = getattr(module, obj.attrs["class"])

                # Next step is to ensure that parameters that are passed are either tuple or NoneType, as required.
                try:
                    parameters = tuple(obj.attrs.get("parameters", None))
                except TypeError:
                    parameters = None

                try:
                    discretization = tuple(obj.attrs.get("discretization", None))
                except TypeError:
                    discretization = None
                # Use the imported data to initialize a new instance. Tuple datatype is imported as list.
                orbit_ = class_(
                    state=obj[...],
                    **{
                        **dict(obj.attrs.items()),
                        "parameters": parameters,
                        "discretization": discretization,
                        **orbitkwargs,
                    }
                )

                # If there is a single orbit in the collection (i.e. a dataset and not a group) then do not append as a
                # list.
                if len(orbit_collection) == 1:
                    orbit_group = orbit_
                else:
                    orbit_group.append(orbit_)
            imported_orbits.append(orbit_group)

    if validate and len(imported_orbits) == 1:
        return imported_orbits[0].preprocess()
    elif len(imported_orbits) == 1:
        return imported_orbits[0]
    elif validate:
        return [x.preprocess() for x in imported_orbits]
    else:
        return imported_orbits


def read_tileset(filename, keys, orbit_names, validate=False, **orbitkwargs):
    """Importation of data as tiling dictionary

    Parameters
    ----------
    filename : str
        The relative/absolute location of the file.
    keys : tuple
        Strings representing the labels to give to the orbits corresponding to orbit_names, respectively.
    orbit_names : tuple
        Strings representing the dataset names within the .h5 file.
    validate : bool
        Whether or not to call preprocess method on each imported orbit.
    orbitkwargs : dict
        Keyword arguments that user wants to provide for construction of orbit instances.

    Returns
    -------
    dict :
        Keys are those provided, values are orbit instances loaded from h5 file.
    """
    # If the number of keys does not equal the number of imported orbits, zip will truncate which will
    # return a dict with fewer key-value pairs than expected.
    assert len(keys) == len(orbit_names)

    # Read the orbits from file. this returns a list of orbits.
    list_of_orbits = read_h5(filename, orbit_names, validate=validate, **orbitkwargs)
    return dict(zip(keys, list_of_orbits))


def to_symbol_string(symbol_array):
    symbolic_string = symbol_array.astype(str).copy()
    shape_of_axes_to_contract = symbol_array.shape[1:]
    for i, shp in enumerate(shape_of_axes_to_contract):
        symbolic_string = [
            (i * "_").join(list_)
            for list_ in np.array(symbolic_string).reshape(-1, shp).tolist()
        ]
    symbolic_string = ((len(shape_of_axes_to_contract)) * "_").join(symbolic_string)
    return symbolic_string


def to_symbol_array(symbol_string, symbol_array_shape):
    return (
        np.array([char for char in symbol_string.replace("_", "")])
        .astype(int)
        .reshape(symbol_array_shape)
    )
