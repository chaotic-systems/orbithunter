from ..core import Orbit
import numpy as np


__all__ = ['OrbitHN']

class OrbitHN(Orbit):
    """ Spatiotemporal Henon map

    """
    def __init__(self, state=None, basis='field', parameters=(0, 0), **kwargs):
        # state is numpy array
        self.state = state
        self.T = parameters[0]
        self.basis = 'field'

    def dae(self, **kwargs):
        F = np.roll(self.state, 1) + 1.4*(self.state**2) - 0.3*np.roll(self.state, -1) - 1
        return OrbitHN(state=F, parameters=(len(F), 1))

    def jacobian(self, **kwargs):
        J = (2*1.4*np.diag(self.state.ravel()) + np.roll(np.eye(len(self.state)), 1, axis=1) \
             - 0.3*np.roll(np.eye(len(self.state)), -1, axis=1))
        return J

    def residual(self, **kwargs):
        F = self.dae().state
        return 0.5 * F.dot(F)

    def from_numpy_array(self, state, **kwargs):
        return OrbitHN(state=state, parameters=(len(state), 1))

    @property
    def shape(self):
        return self.state.size


def ks_fpo_dictionary(tileset='default', comoving=False, rescaled=False, **kwargs):
    """ Template tiles for Kuramoto-Sivashinsky equation.


    Parameters
    ----------
    padded : bool
    Whether to use the zero-padded versions of the tiles
    comoving : bool
    Whether to use the defect tile in the physical or comoving frame.

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


def _parsing_dictionary():
    class_dict = {'base': OrbitKS, 'none': OrbitKS, 'full': OrbitKS, 'OrbitKS': OrbitKS,
                  'anti': AntisymmetricOrbitKS, 'AntisymmetricOrbitKS': AntisymmetricOrbitKS,
                  'ppo': ShiftReflectionOrbitKS, 'ShiftReflectionOrbitKS': ShiftReflectionOrbitKS,
                  'rpo': RelativeOrbitKS, 'RelativeOrbitKS': RelativeOrbitKS,
                  'eqva': EquilibriumOrbitKS, 'EquilibriumOrbitKS': EquilibriumOrbitKS,
                  'reqva': RelativeEquilibriumOrbitKS, 'RelativeEquilibriumOrbitKS': RelativeEquilibriumOrbitKS}
    return class_dict


def _orbit_instantiation(f, class_generator, data_format='orbithunter', **orbitkwargs):
    if data_format == 'orbithunter':
        field = np.array(f['field'])
        params = tuple(f['parameters'])
        orbit_ = class_generator(state=field, basis='field', parameters=params, **orbitkwargs)
    elif data_format == 'orbithunter_old':
        field = np.array(f['field'])
        L = float(f['space_period'][()])
        T = float(f['time_period'][()])
        S = float(f['spatial_shift'][()])
        orbit_ = class_generator(state=field, basis='field', parameters=(T, L, S), **orbitkwargs)
    else:
        fieldtmp = f['/data/ufield']
        L = float(f['/data/space'][0])
        T = float(f['/data/time'][0])
        field = fieldtmp[:]
        S = float(f['/data/shift'][0])
        orbit_ = class_generator(state=field, basis='field', parameters=(T, L, S), **orbitkwargs)
    return orbit_


def ks_parsing_util():
    return _parsing_dictionary(), _orbit_instantiation