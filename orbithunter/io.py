from orbithunter.orbit import OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, \
    AntisymmetricOrbitKS, EquilibriumOrbitKS, RelativeEquilibriumOrbitKS
import warnings
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

__all__ = ['read_h5', 'parse_class']


def read_h5(filename, directory='', data_format='orbithunter', state_type='modes'):
    if directory == 'local':
        directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/local/')), '')

    class_generator = parse_class(filename)
    with h5py.File(os.path.abspath(os.path.join(directory, filename)), 'r') as f:
        if data_format == 'orbithunter':
            field = np.array(f['field'])
            L = float(f['space_period'][()])
            T = float(f['time_period'][()])
            S = float(f['spatial_shift'][()])
            orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S)
        elif data_format == 'orbithunter_old':
            field = np.array(f['field'])
            L = float(f['L'][()])
            T = float(f['T'][()])
            S = float(f['spatial_shift'][()])
            orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S)
        else:
            fieldtmp = f['/data/ufield']
            L = float(f['/data/space'][0])
            T = float(f['/data/time'][0])
            field = fieldtmp[:]
            S = float(f['/data/shift'][0])
            orbit = class_generator(state=field, state_type='field', T=T, L=L, S=S)

    return verify_integrity(orbit).convert(to=state_type)

def parse_class(filename):
    name_string = os.path.basename(filename).split('_')[0]

    old_names = ['none', 'full', 'rpo', 'reqva', 'ppo', 'eqva', 'anti']
    new_names = ['OrbitKS', 'RelativeOrbitKS', 'RelativeEquilibriumOrbitKS', 'ShiftReflectionOrbitKS',
                 'AntisymmetricOrbitKS', 'EquilibriumOrbitKS']

    all_names = np.array(old_names + new_names)
    if name_string in all_names:
        class_name = name_string
    else:
        name_count = np.array([filename.count(class_name) for class_name in all_names])
        class_name = np.array(all_names)[np.argmax(name_count)]

    class_dict = {'none': OrbitKS, 'full': OrbitKS, 'OrbitKS': OrbitKS,
                  'anti': AntisymmetricOrbitKS, 'AntisymmetricOrbitKS': AntisymmetricOrbitKS,
                  'ppo': ShiftReflectionOrbitKS, 'ShiftReflectionOrbitKS': ShiftReflectionOrbitKS,
                  'rpo': RelativeOrbitKS, 'RelativeOrbitKS': RelativeOrbitKS,
                  'eqva': EquilibriumOrbitKS, 'EquilibriumOrbitKS': EquilibriumOrbitKS,
                  'reqva': RelativeEquilibriumOrbitKS, 'RelativeEquilibriumOrbitKS': RelativeEquilibriumOrbitKS}

    class_generator = class_dict.get(class_name, RelativeOrbitKS)
    return class_generator

def _make_proper_pathname(pathname_tuple,folder=False):
    if folder:
        return os.path.join(os.path.abspath(os.path.join(*pathname_tuple)),'')
    else:
        return os.path.abspath(os.path.join(*pathname_tuple))

def verify_integrity(orbit):
    if orbit.__class__.__name__ in ['RelativeOrbitKS', 'RelativeEquilibriumOrbitKS']:
        residual_imported_S = orbit.residual()
        orbit_inverted_shift = orbit.__class__(state=orbit.state, state_type=orbit.state_type,
                                               T=orbit.T, L=orbit.L, S=-1.0*orbit.S)
        residual_negated_S = orbit_inverted_shift.residual()
        if residual_imported_S > residual_negated_S:
            return orbit_inverted_shift
        else:
            return orbit
    else:
        return orbit


