from orbithunter.orbit import OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, \
    AntisymmetricOrbitKS, EquilibriumOrbitKS, RelativeEquilibriumOrbitKS
import warnings
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

__all__ = ['read_h5', 'parse_class']


def read_h5(filename, directory='local', data_format='orbithunter', equation='ks',
            state_type='modes', class_name=None, check=False, **orbitkwargs):
    if directory == 'local':
        directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/local/')), '')
    elif directory == 'orbithunter':
        directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')

    if class_name is None:
        class_generator = parse_class(filename, equation=equation)
    else:
        class_generator = parse_class(class_name, equation=equation)

    with h5py.File(os.path.abspath(os.path.join(directory, filename)), 'r') as f:
        if equation == 'ks':
            if data_format == 'orbithunter':
                field = np.array(f['field'])
                L = float(f['space_period'][()])
                T = float(f['time_period'][()])
                S = float(f['spatial_shift'][()])
                orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S, **orbitkwargs)
            elif data_format == 'orbithunter_old':
                field = np.array(f['field'])
                L = float(f['L'][()])
                T = float(f['T'][()])
                S = float(f['spatial_shift'][()])
                orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S, **orbitkwargs)
            else:
                fieldtmp = f['/data/ufield']
                L = float(f['/data/space'][0])
                T = float(f['/data/time'][0])
                field = fieldtmp[:]
                S = float(f['/data/shift'][0])
                orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S, **orbitkwargs)

    # verify typically returns Orbit, code; just want the orbit instance here
    if check:
        # The automatic importation attempts to validate symmetry/class type.
        return orbit.verify_integrity()[0].convert(to=state_type)
    else:
        # If the class name is provided, force it through without verification.
        return orbit.convert(to=state_type)


def parse_class(filename, equation='ks'):
    name_string = os.path.basename(filename).split('_')[0]

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


def _make_proper_pathname(pathname_tuple,folder=False):
    if folder:
        return os.path.join(os.path.abspath(os.path.join(*pathname_tuple)), '')
    else:
        return os.path.abspath(os.path.join(*pathname_tuple))

