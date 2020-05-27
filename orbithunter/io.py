import warnings
import os
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
from orbithunter.orbit import Orbit, RelativeOrbit, ShiftReflectionOrbit, AntisymmetricOrbit, EquilibriumOrbit

__all__ = ['read_h5', 'parse_class']

def read_h5(filename, directory='', statetype='field'):
    if directory == 'default':
        directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')

    class_generator, data_format = parse_class(filename)
    with h5py.File(os.path.abspath(os.path.join(directory, filename)), 'r') as f:
        if data_format=='new':
            field = np.array(f['field'])
            L = float(f['speriod'][()])
            T = float(f['period'][()])
            S = float(f['spatial_shift'][()])
            orbit = class_generator(state=field, statetype='field', T=T, L=L, S=S)
        else:
            fieldtmp = f['/data/ufield']
            L = float(f['/data/space'][0])
            T = float(f['/data/time'][0])
            field = fieldtmp[:]
            S = float(f['/data/shift'][0])
            orbit = class_generator(state=field, statetype='field', T=T, L=L, S=S)

    return verify_integrity(orbit).convert(to=statetype)

def parse_class(filename):
    name_string = os.path.basename(filename).split('_')[0]

    old_names = ['none','full', 'rpo', 'ppo', 'eqva', 'anti']
    new_names = ['Orbit', 'RelativeOrbit', 'ShiftReflectionOrbit',
                 'AntisymmetricOrbit', 'EquilibriumOrbit']

    all_names = np.array(old_names + new_names)
    if name_string in all_names:
        class_name = name_string
    else:
        name_count = np.array([filename.count(class_name) for class_name in all_names])
        class_name = np.array(all_names)[np.argmax(name_count)]

    if class_name in new_names:
        data_format = 'new'
    else:
        data_format = 'old'

    class_dict = {'none': Orbit, 'full': Orbit, 'Orbit': Orbit,
                  'anti': AntisymmetricOrbit, 'AntisymmetricOrbit': AntisymmetricOrbit,
                  'ppo': ShiftReflectionOrbit, 'ShiftReflectionOrbit': ShiftReflectionOrbit,
                  'rpo': RelativeOrbit, 'RelativeOrbit': RelativeOrbit,
                  'eqva': EquilibriumOrbit, 'EquilibriumOrbit': EquilibriumOrbit}

    class_generator = class_dict.get(class_name, RelativeOrbit)
    return class_generator, data_format

def _make_proper_pathname(pathname_tuple,folder=False):
    if folder:
        return os.path.join(os.path.abspath(os.path.join(*pathname_tuple)),'')
    else:
        return os.path.abspath(os.path.join(*pathname_tuple))

def verify_integrity(orbit):
    if orbit.__class__.__name__ == 'RelativeOrbit':
        residual_imported_S = orbit.residual()
        orbit_inverted_shift = orbit.__class__(state=orbit.state, statetype=orbit.statetype,
                                           T=orbit.T, L=orbit.L, S=-1.0*orbit.S)
        residual_negated_S = orbit_inverted_shift.residual()
        if residual_imported_S > residual_negated_S:
            return orbit_inverted_shift
        else:
            return orbit
    else:

        return orbit


