from .orbit_ks import OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, \
    AntisymmetricOrbitKS, EquilibriumOrbitKS, RelativeEquilibriumOrbitKS
import os
import numpy as np
import h5py
import pandas as pd
import itertools

__all__ = ['read_h5', 'parse_class', 'convergence_log', 'refurbish_log', 'write_symbolic_log',
           'read_symbolic_log', 'to_symbol_string', 'to_symbol_array']


def read_h5(filename, directory='local', data_format='orbithunter', equation='ks',
            basis='modes', class_name=None, check=False, **orbitkwargs):

    if class_name is None:
        class_generator = parse_class(filename, equation=equation)
    elif isinstance(class_name, str):
        class_generator = parse_class(class_name, equation=equation)
    else:
        class_generator = class_name

    if directory == 'local':
        directory = os.path.abspath(os.path.join(__file__, ''.join(['../../data/local/',
                                                                    class_generator.__name__, '/'])))

    with h5py.File(os.path.abspath(os.path.join(directory, filename)), 'r') as f:
        if equation == 'ks':
            if data_format == 'orbithunter':
                field = np.array(f['field'])
                params = tuple(f['parameters'])
                orbit = class_generator(state=field, basis='field', parameters=params, **orbitkwargs)
            elif data_format == 'orbithunter_old':
                field = np.array(f['field'])
                L = float(f['space_period'][()])
                T = float(f['time_period'][()])
                S = float(f['spatial_shift'][()])
                orbit = class_generator(state=field, basis='field', parameters=(T, L, S), **orbitkwargs)
            else:
                fieldtmp = f['/data/ufield']
                L = float(f['/data/space'][0])
                T = float(f['/data/time'][0])
                field = fieldtmp[:]
                S = float(f['/data/shift'][0])
                orbit = class_generator(state=field, basis='field', parameters=(T, L, S), **orbitkwargs)

    # verify typically returns Orbit, code; just want the orbit instance here
    if check:
        # The automatic importation attempts to validate symmetry/class type.
        return orbit.verify_integrity()[0].convert(to=basis, inplace=True)
    else:
        # If the class name is provided, force it through without verification.
        return orbit.convert(to=basis, inplace=True)


def parse_class(filename, equation='ks'):
    name_string = os.path.basename(filename).split('_')[0]
    if equation == 'ks':
        old_names = ['none', 'full', 'rpo', 'reqva', 'ppo', 'eqva', 'anti']
        new_names = ['OrbitKS', 'RelativeOrbitKS', 'RelativeEquilibriumOrbitKS', 'ShiftReflectionOrbitKS',
                     'AntisymmetricOrbitKS', 'EquilibriumOrbitKS']

        all_names = np.array(old_names + new_names)
        if name_string in all_names:
            class_name = name_string
        else:
            name_index = np.array([filename.find(class_name) for class_name in all_names], dtype=float)
            name_index[name_index < 0] = np.inf
            class_name = np.array(all_names)[np.argmin(name_index)]

        class_dict = {'none': OrbitKS, 'full': OrbitKS, 'OrbitKS': OrbitKS,
                      'anti': AntisymmetricOrbitKS, 'AntisymmetricOrbitKS': AntisymmetricOrbitKS,
                      'ppo': ShiftReflectionOrbitKS, 'ShiftReflectionOrbitKS': ShiftReflectionOrbitKS,
                      'rpo': RelativeOrbitKS, 'RelativeOrbitKS': RelativeOrbitKS,
                      'eqva': EquilibriumOrbitKS, 'EquilibriumOrbitKS': EquilibriumOrbitKS,
                      'reqva': RelativeEquilibriumOrbitKS, 'RelativeEquilibriumOrbitKS': RelativeEquilibriumOrbitKS}
        class_generator = class_dict.get(class_name, OrbitKS)
        return class_generator
    else:
        return None


def convergence_log(initial_orbit, converge_result, log_path, spectrum='random', method='hybrid'):
    initial_condition_log_ = pd.read_csv(log_path, index_col=0)
    # To store all relevant info as a row in a Pandas DataFrame, put into a 1-D array first.
    dataframe_row = [[initial_orbit.parameters, initial_orbit.field_shape,
                     np.abs(initial_orbit.convert(to='field').state).max(), converge_result.orbit.residual(),
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


def write_symbolic_log(symbol_array, converge_result, log_filename, padding=False,
                             comoving=False):
    symbol_string = to_symbol_string(symbol_array)
    dataframe_row_values = [[symbol_string, converge_result.orbit.parameters, converge_result.orbit.field_shape,
                             converge_result.orbit.residual(),
                             converge_result.status, padding, comoving, symbol_array.shape]]
    labels = ['symbol_string', 'parameters', 'field_shape', 'residual', 'status', 'padding',
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
        elif symbolic_intersection.loc[0, 'status'] == '1' and overwrite:
            return False
        elif symbolic_intersection.loc[0, 'status'] != '1' and retry:
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


