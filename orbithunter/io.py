import os
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

def _find_parent_class_in_module(module):
    classes = [cls for name, cls in module.__dict__.items() if isinstance(cls, type)]
    # Recursively search for the root class of the module
    for cls in classes:
        # For each class, count how many subclasses it has. The "parent" class is only its own
        # subclass; therefore if the count equals 1 then we are done.
        subclass_count = np.sum([int(issubclass(cls, cls2)) for cls2 in classes])
        if subclass_count == 1:
            return cls


def read_h5(path, orbit_group, equation, class_names=None, validate=False, **orbitkwargs):
    """
    Parameters
    ----------
    path : str
        Absolute or relative path to .h5 file
    orbit_names : str or tuple of str
        The h5py.Group (s) of the corresponding .h5 file
    equation : str
        The abbreviation for the corresponding equation's module.
    class_namess : str or iterable of str
        If iterable of str then it needs to be the same length as orbit_names
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
    orbit_name can either be a single string or an iterable; this is to make importation of multiple .h5 files
    more efficient, i.e. avoiding multiple dynamic imports of the same package.

    The purpose of this function is only to import data whose file and group names are known. To query a HDF5
    file, use query_h5 instead.

    """
    # The following loads the module for the correct equation dynamically when the function is called, to prevent
    # imports of all equations and possible circular dependencies.
    module = importlib.import_module(''.join(['.', equation]), 'orbithunter')

    if isinstance(orbit_group, str):
        # If string, then place inside a len==1 tuple so that iteration doesn't raise an error
        orbit_names = (orbit_group,)
    elif isinstance(orbit_group, tuple):
        pass
    else:
        raise TypeError('Incorrect type for hdf5 group names; needs to be str or a tuple of str')

    # With orbit_names now correctly instantiated as an iterable, can open file and iterate.
    with h5py.File(os.path.abspath(path), 'r') as file:
        imported_orbits = []
        for i, orbit_group_name in enumerate(orbit_names):
            # class_names needs to be constant (str input) or specified for each group (tuple with len == len(orbit_names))
            try:
                if isinstance(class_names, str):
                    class_ = getattr(module, class_names)
                elif isinstance(class_names, tuple):
                    class_ = getattr(module, class_names[i])
                elif class_names is None:
                    # If the class generator is not provided, it is assumed to be able to be inferred from the filename.
                    # This is simply a convenience tool because typically the classes are the best partitions of the
                    # full data set.
                    class_ = getattr(module, str(orbit_group_name.split('_')[0]))
                else:
                    raise AttributeError
            except AttributeError:
                class_ = _find_parent_class_in_module(module)

            orbit_ = class_(state=file[''.join([orbit_group_name, '/', class_.bases()[0]])][...],
                            parameters=tuple(file[''.join([orbit_group_name, '/parameters'])]),
                            basis=class_.bases()[0], **orbitkwargs)
            if validate and len(orbit_names) == 1:
                return orbit_.preprocess()[0]
            elif len(orbit_names) == 1:
                return orbit_
            elif validate:
                imported_orbits.append(orbit_.preprocess()[0])
            else:
                imported_orbits.append(orbit_)

    return imported_orbits


def read_tileset(filename, orbit_names, keys, equation, class_names, validate=False, **orbitkwargs):
    """ Importation of data as tiling dictionary

    Parameters
    ----------
    filename : str
        The relative/absolute location of the file.
    orbit_names : str or tuple of str
        The orbit names within the .h5 file
    keys :
        The labels to give to the orbits corresponding to orbit_names, respectively.
    equation : str
        The module name for the corresponding equation, i.e. 'ks'
    class_names : str
        The name of the Orbit class/subclass
    validate :
        Whether or not to call preprocess method on each imported orbit.
    orbitkwargs :
        Keyword arguments that user wants to provide for construction of orbit instances.

    Returns
    -------
    dict :
        Dictionary whose values are given by the orbits specified by orbit_names.
    """
    orbits = read_h5(filename, orbit_names, equation, class_names, validate=validate, **orbitkwargs)
    # if keys and orbits are not the same length, it will only form a dict with min([len(keys), len(orbits)]) items
    return dict(zip(keys, orbits))


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
