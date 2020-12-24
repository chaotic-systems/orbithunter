import os
import numpy as np
import pandas as pd
import itertools
import h5py
import importlib

def ks_fpo_dictionary(tileset='default', comoving=False, rescaled=False):
    """ Template tiles for Kuramoto-Sivashinsky equation.


    Parameters
    ----------
    tileset : str
        Which tileset to use: ['default', 'complete', 'extra_padded', 'extra_padded_space', 'original', 'padded',
                               'padded_space', 'padded_time', 'resized']
    comoving : bool
        Whether to use the defect tile in the physical or comoving frame.
    rescaled : bool
        Whether to rescale the dictionary to the maximum of all tiles.

    Returns
    -------
    tile_dict : dict
    Dictionary which contains defect, streak, wiggle tiles for use in tiling and gluing.

    Notes
    -----
    The dictionary is setup as follows : {0: streak, 1: defect, 2: wiggle}
    """
    fpo_filenames = []
    directory = os.path.abspath(os.path.join(__file__, '../../../data/ks/tiles/', ''.join(['./', tileset])))
    if rescaled:
        directory = os.path.abspath(os.path.join(directory, './rescaled/'))

    # streak orbit
    fpo_filenames.append(os.path.abspath(os.path.join(directory, './OrbitKS_streak.h5')))

    if comoving:
        # padded defect orbit in physical frame.
        fpo_filenames.append(os.path.abspath(os.path.join(directory, './OrbitKS_defect_comoving.h5')))
    else:
        # padded defect Orbit in comoving frame
        fpo_filenames.append(os.path.abspath(os.path.join(directory, './OrbitKS_defect.h5')))

    # wiggle orbit
    fpo_filenames.append(os.path.abspath(os.path.join(directory, './OrbitKS_wiggle.h5')))

    return dict(zip(range(len(fpo_filenames)), fpo_filenames))


__all__ = ['read_h5', 'read_fpo_set', 'convergence_log', 'refurbish_log', 'write_symbolic_log',
           'read_symbolic_log', 'to_symbol_string', 'to_symbol_array', 'parse_class']

def read_h5(h5_file, h5_group, basis='field', class_name=None,
            validate=False, orbithunter_archive=True, **orbitkwargs):
    """
    Parameters
    ----------
    h5_file
    h5_group : str
        The h5py.Group from which field and parameters are imported.
    basis
    import_cls
    validate
    orbithunter_archive
    orbitkwargs

    Returns
    -------

    """
    # For ease of use, point to the saved data incorporated into the package itself.
    if orbithunter_archive:
        h5_file = os.path.abspath(os.path.join('../../data/', h5_file))

    # The first substring after '/' in h5_group is required to be the equation's abbreviation.
    # the following loads the module
    module = importlib.import_module(''.join(['.', h5_group.split('/')[1]]), 'orbithunter')
    if class_name is None:
        # If the class generator is not provided, it is assumed to be able to be inferred from the filename.
        class_ = getattr(module, str(os.path.basename(h5_file).split('.h5')[0]))
    else:
        class_ = getattr(module, class_name)

    with h5py.File(os.path.abspath(h5_file), 'r') as file:
        orbit_ = class_(state=file[h5_group]['field'][...], parameters=tuple(file[h5_group]['parameters']),
                        basis='field', **orbitkwargs)

    # verify typically returns Orbit, code; just want the orbit instance here
    if validate:
        # The automatic importation attempts to validate symmetry/class type.
        return orbit_.verify_integrity()[0].transform(to=basis)
    else:
        # If the class name is provided, force it through without verification.
        return orbit_.transform(to=basis)


def parse_class(filename, class_dict):
    name_string = os.path.basename(filename).split('_')[0]
    keys = np.array(list(class_dict.keys()))

    if name_string in keys:
        cls_str = name_string
    else:
        name_index = np.array([filename.find(cls_str) for cls_str in keys], dtype=float)
        name_index[name_index < 0] = np.inf
        cls_str = np.array(keys)[np.argmin(name_index)]

    return class_dict.get(cls_str, class_dict['base'])


def read_fpo_set(equation='ks', **kwargs):
    if equation == 'ks':
        fpo_dict = ks_fpo_dictionary
    else:
        raise ValueError('The function read_fpo_set encountered an unrecognizable equation.')
    # Require fpos to be in field basis; pop it off kwargs lest it throws an error for multiples
    kwargs.pop('basis', None)
    return {key: read_h5(val, basis='field', **kwargs) for key, val in fpo_dict(**kwargs).items()}


def convergence_log(initial_orbit, converge_result, log_path, spectrum='random', method='hybrid'):
    initial_condition_log_ = pd.read_csv(log_path, index_col=0)
    # To store all relevant info as a row in a Pandas DataFrame, put into a 1-D array first.
    dataframe_row = [[initial_orbit.parameters, initial_orbit.field_shape,
                     np.abs(initial_orbit.transform(to='field').state).max(), converge_result.orbit.residual(),
                     converge_result.status, spectrum, method]]
    labels = ['parameters', 'field_shape', 'field_magnitude', 'residual', 'status', 'spectrum', 'numerical_method']
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


def write_symbolic_log(symbol_array, converge_result, log_filename, tileset='default',
                             comoving=False):
    symbol_string = to_symbol_string(symbol_array)
    dataframe_row_values = [[symbol_string, converge_result.orbit.parameters, converge_result.orbit.field_shape,
                             converge_result.orbit.residual(),
                             converge_result.status, tileset, comoving, symbol_array.shape]]
    labels = ['symbol_string', 'parameters', 'field_shape', 'residual', 'status', 'tileset',
              'comoving', 'tile_shape']

    dataframe_row = pd.DataFrame(dataframe_row_values, columns=labels).astype(object)
    log_path = os.path.abspath(os.path.join(__file__, '../../data/logs/', log_filename))

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

    log_path = os.path.abspath(os.path.join(__file__, '../../data/logs/', log_filename))
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
