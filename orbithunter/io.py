import os
import sys
import numpy as np
import pandas as pd
import itertools
import h5py
import importlib


__all__ = ['read_h5', 'read_tileset', 'convergence_log', 'refurbish_log', 'write_symbolic_log',
           'read_symbolic_log', 'to_symbol_string', 'to_symbol_array']

"""
Functions and utilities corresponding to filenames and input/output. The convention which we hope others
will abide is to create a .h5 file for each symmetry type for each equation. Within each file, each h5py.Group
represents an orbit, which in turn contains the state information, parameters, and perhaps additional data (which
would require overloading the Orbit.to_h5() method). Therefore, general import statements only require the
filename and the group names, as the group entries for each orbit should be uniform. The downside of this is
that currently there is no query method for the HDF5 files built in to orbithunter currently. In order to query
an h5 file, it is required to use the h5py API explicitly.   
"""


def _find_parent_class_from_module(module):
    """ Parse a string to find the longest class name that occurs as a substring.
    Parameters
    ----------
    module :
        The module whose classes are to be investigated.

    Returns
    -------
    cls : class generator
        The "parent" class of a submodule. This is the class that is only a subclass of itself (with
        respect to the other classes in the module).
    """
    classes = [cls for name, cls in module.__dict__.items() if isinstance(cls, type)]
    # Recursively search for the root class of the module
    for cls in classes:
        # For each class, count how many subclasses it has. The "parent" class is only its own
        # subclass; therefore if the count equals 1 then we are done.
        subclass_count = np.sum([int(issubclass(cls, cls2)) for cls2 in classes])
        if subclass_count == 1:
            return cls
    else:
        raise ValueError('Did not find parent class in module')


def _find_class_from_str(string, module):
    """ Parse a string to find the longest class name that occurs as a substring.
    Parameters
    ----------
    string : str
        A string to search for class names as substrings
    module :
        The module that the class names are being pulled from

    Returns
    -------
    name : str
        The (longest) class name that occurs in the provided string. Longest returned to avoid substrings i.e.
        "SubclassOrbit" vs. "Orbit"
    """
    names = [str(name) for name, cls in module.__dict__.items() if isinstance(cls, type)]
    # Recursively search for the root class of the module
    # Using the longest names first means that we do not pick out substrings.
    for name in sorted(names, key=len, reverse=True):
        if string.count(name) > 0:
            return name
    else:
        raise ValueError('Did not find class name as substring.')


def read_h5(filename, *datanames, validate=False, **orbitkwargs):
    """
    Parameters
    ----------
    filename : str
        Absolute or relative path to .h5 file
    datanames : str or tuple
        Names of either h5py.Datasets or h5py.Groups within .h5 file. Recursively returns all orbits (h5py.Datasets)
        associated with all names provided. If nothing provided, return all datasets in file.
    validate : bool
        Whether or not to access Orbit().preprocess, presumably checking the integrity of the imported data.
    orbitkwargs : dict
        Any additional keyword arguments relevant for instantiation.

    Returns
    -------
    Orbit or list of Orbits.
        The imported data; either a single orbit instance or a list of orbit instances.

    Notes
    -----
    datanames can either be a single string, iterable, or nothing at all. If nothing provided then all datasets
    in the h5 file are imported. This function is to make importation of multiple orbits more efficient,
    i.e. avoiding multiple dynamic imports of the same package. The purpose of this function is only to import data
    whose file and group names are known. Querying .h5 files must be done through the h5py API instead.

    The state data should be saved as a dataset. The attributes which are required for expected output
     are 'basis', 'class', 'parameters'; all attributes included by default are 'discretization'
     (state shape in physical basis, not necessarily the shape of the saved state).
    """

    # unpack tuples so providing multiple strings or strings in a tuple yield same results.
    if len(datanames) == 1 and isinstance(*datanames, tuple):
        datanames = tuple(*datanames)

    datasets = []
    imported_orbits = []

    # With orbit_names now correctly instantiated as an iterable, can open file and iterate.
    with h5py.File(os.path.abspath(filename), 'r') as file:
        # define visititems() function here to use variables in current namespace
        def parse_datasets(h5name, h5obj):
            # Orbits are stored as h5py.Dataset(s) . Collections or orbits are h5py.Group(s).
            if isinstance(h5obj, h5py.Dataset):
                datasets.append(h5name)

        # iterate through all names provided, extract all datasets from groups provided.
        # If no Dataset/Group names were provided, iterate through the entire file.
        for name in (datanames or file):
            if isinstance(file[name], h5py.Group):
                file[name].visititems(parse_datasets)
            elif isinstance(file[name], h5py.Dataset):
                datasets.append(name)

        for orbitname in datasets:
            obj = file[orbitname]
            try:
                # The following loads the module for the correct equation dynamically
                # when the function is called, to prevent imports of all equations and possible circular dependencies.
                # Module name needs to be included as suffix of class name or included as h5 attribute for this to work.
                # Only the former is actually expected; hence the try-except statement is really built around the second
                # clause.
                module_name = obj.attrs.get('module', None) or obj.attrs['class'].split('Orbit')[-1].lower()
            except AttributeError:
                # If not derived from h5 'class' attribute, equation must be provided.
                module_name = orbitkwargs.get('module', None)

            # For each h5py.Dataset, need to get the corresponding module which contains its class.
            if module_name not in sys.modules:
                # module name assumed to be in main orbithunter directory.
                module = importlib.import_module(''.join(['.', module_name]), 'orbithunter')
            else:
                # If already imported, do not load again.
                module = sys.modules[module_name]

            try:
                try:
                    # Get the class from metadata if possible
                    class_ = getattr(module, obj.attrs['class'])
                except (KeyError, AttributeError):
                    # If class name is not in metadata, then try to get it from Dataset or filename, in that order.
                    try:
                        # Attempt to get class from h5py.Dataset str
                        class_ = getattr(module, _find_class_from_str(orbitname, module))
                    except (ValueError, AttributeError):
                        # If substring does not exist or is not a class the attempt to retrieve from filename
                        class_ = getattr(module, _find_class_from_str(filename, module))
            except (ValueError, AttributeError):
                # if all methods fail, use the parent class of the module as a last resort.
                class_ = _find_parent_class_from_module(module)

            # Next step is to ensure that parameters that are passed are either tuple or NoneType, as required.
            try:
                parameters = tuple(obj.attrs.get('parameters', None))
            except TypeError:
                parameters = None

            # As it takes a deliberate effort, keyword arguments passed to read_h5 are favored over saved attributes
            # This allows for control in case of incorrect attributes; the dictionary update avoids sending more than
            # one value to the same keyword argument.
            # This passes all saved attributes, tuple or None for parameters, and any additional keyword
            # arguments to the class
            orbit_ = class_(state=obj[...],  **{**dict(obj.attrs.items()), 'parameters': parameters, **orbitkwargs})
            imported_orbits.append(orbit_)

    if validate and len(imported_orbits) == 1:
        return imported_orbits[0].preprocess()
    elif len(imported_orbits) == 1:
        return imported_orbits[0]
    elif validate:
        return [x.preprocess() for x in imported_orbits]
    else:
        return imported_orbits


def read_tileset(filename, keys, orbit_names, validate=False, **orbitkwargs):
    """ Importation of data as tiling dictionary

    Parameters
    ----------
    filename : str
        The relative/absolute location of the file.
    keys : tuple
        Strings representing the labels to give to the orbits corresponding to orbit_names, respectively.
    orbit_names : tuple
        Strings representing the dataset names within the .h5 file; h5py.Groups not allowed.
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


def convergence_log(initial_orbit, minimize_result, log_path, spectrum='random', method='adj'):
    """ Function to log the results of applying orbithunter.optimize.minimize

    Parameters
    ----------
    initial_orbit : Orbit
        The initial guess orbit
    minimize_result :
        The
    log_path
    spectrum
    method

    Returns
    -------

    """
    initial_condition_log_ = pd.read_csv(log_path, index_col=0)
    # To store all relevant info as a row in a Pandas DataFrame, put into a 1-D array first.
    dataframe_row = [[initial_orbit.parameters, initial_orbit.shapes()[0],
                     np.abs(initial_orbit.transform(to=initial_orbit.bases()[0]).state).max(),
                      minimize_result.orbit.residual(),
                     minimize_result.status, spectrum, method]]
    labels = ['parameters', 'shape', ''.join([initial_orbit.bases()[0], '_magnitude']),
              'residual', 'status', 'spectrum', 'numerical_method']
    new_row = pd.DataFrame(dataframe_row, columns=labels)
    initial_condition_log_ = pd.concat((initial_condition_log_, new_row), axis=0)
    initial_condition_log_.reset_index(drop=True).drop_duplicates().to_csv(log_path)
    return initial_condition_log_


def refurbish_log(orbit_, filename, log_filename, overwrite=False, **kwargs):
    if not os.path.isfile(log_filename):
        refurbish_log_ = pd.Series(filename).to_frame(name='filename')
    else:
        refurbish_log_ = pd.read_csv(log_filename, index_col=0)

    if not overwrite and filename in np.array(refurbish_log_.values).tolist():
        orbit_.to_h5(filename, **kwargs)
        orbit_.plot(filename=filename, **kwargs)
        refurbish_log_ = pd.concat((refurbish_log_, pd.Series(filename).to_frame(name='filename')), axis=0)
        refurbish_log_.reset_index(drop=True).drop_duplicates().to_csv(log_filename)


def write_symbolic_log(symbol_array, minimize_result, log_filename, tileset='default',
                       comoving=False):
    symbol_string = to_symbol_string(symbol_array)
    dataframe_row_values = [[symbol_string, minimize_result.orbit.parameters, minimize_result.orbit.shapes()[0],
                             minimize_result.orbit.residual(),
                             minimize_result.status, tileset, comoving, symbol_array.shape]]
    labels = ['symbol_string', 'parameters', 'shape', 'residual', 'status', 'tileset',
              'comoving', 'tile_shape']

    dataframe_row = pd.DataFrame(dataframe_row_values, columns=labels).astype(object)
    log_path = os.path.abspath(log_filename)

    if not os.path.isfile(log_path):
        file_ = dataframe_row.copy()
    else:
        file_ = pd.read_csv(log_path, dtype=object, index_col=0)
        file_ = pd.concat((file_, dataframe_row), axis=0)
    file_.reset_index(drop=True).drop_duplicates().to_csv(log_path)
    # To store all relevant info as a row in a Pandas DataFrame, put into a 1-D array first.
    return None


def read_symbolic_log(symbol_array, log_filename, overwrite=False, retry=False):
    """ Check to see if a combination has already been searched for locally.

    Returns
    -------

    Notes
    -----
    Computes the equivariant equivalents of the symbolic array being searched for.
    Strings can become arbitrarily long but I think this is unavoidable unless symbolic dynamics are redefined
    to get full shift.

    This checks the records/logs as to whether an orbit or one of its equivariant arrangements converged with
    a particular method.
    """
    all_rotations = itertools.product(*(list(range(a)) for a in symbol_array.shape))
    axes = tuple(range(len(symbol_array.shape)))
    equivariant_str = []
    for rotation in all_rotations:
        equivariant_str.append(to_symbol_string(np.roll(symbol_array, rotation, axis=axes)))

    log_path = os.path.abspath(log_filename)
    if not os.path.isfile(log_path):
        return False
    else:
        symbolic_df = pd.read_csv(log_path, dtype=object, index_col=0)
        symbolic_intersection = symbolic_df[(symbolic_df['symbol_string'].isin(equivariant_str))].reset_index(drop=True)
        if len(symbolic_intersection) == 0:
            return False
        elif overwrite:
            return False
        # If success, then one of the 'status' values has been saves as -1. Count the number of negative ones
        # and see if there is indeed such a value.
        elif len(symbolic_intersection[symbolic_intersection['status'] == -1]) == 0 and retry:
            return False
        else:
            return True


def to_symbol_string(symbol_array):
    symbolic_string = symbol_array.astype(str).copy()
    shape_of_axes_to_contract = symbol_array.shape[1:]
    for i, shp in enumerate(shape_of_axes_to_contract):
        symbolic_string = [(i*'_').join(list_) for list_ in np.array(symbolic_string).reshape(-1, shp).tolist()]
    symbolic_string = ((len(shape_of_axes_to_contract))*'_').join(symbolic_string)
    return symbolic_string


def to_symbol_array(symbol_string, symbol_array_shape):
    return np.array([char for char in symbol_string.replace('_', '')]).astype(int).reshape(symbol_array_shape)
