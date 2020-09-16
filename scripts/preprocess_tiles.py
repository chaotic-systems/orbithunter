from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../data/tiles/originals')
    w = read_h5('AntisymmetricOrbitKS_L17p590_T17p146.h5', directory='../data/tiles/originals')
    m = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='../data/tiles/originals')

    # Two versions of the tiles; one has zero padding to have the relative sizes remain accurate.
    sM = 24
    mM = 48

    for new_M in range(s.M, 22, -2):
        s = rediscretize(s, new_shape=(s.N, new_M))
        s_result = converge(s, method='lstsq', verbose=True)
        if s_result.exit_code:
            s = s_result.orbit
            if new_M == 24:
                padded_s_state = np.tile(np.concatenate((np.zeros([s.N, 20]),
                                                        s.convert(to='field').state,
                                                        np.zeros([s.N, 20])), axis=1), (64, 1))
                EquilibriumOrbitKS(state=padded_s_state, state_type='field', parameters=s.parameters).to_h5(
                    'EquilibriumOrbitKS_streak.h5', directory='../data/tiles/padded/', verbose=True)
                OrbitKS(state=padded_s_state, state_type='field', parameters=s.parameters).to_h5(
                    'OrbitKS_streak.h5', directory='../data/tiles/padded/', verbose=True)

    s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../data/tiles/originals')
    for new_M in range(s.M, 66, -2):
        s = rediscretize(s, new_shape=(s.N, new_M))
        s_result = converge(s, method='hybrid', verbose=True)
        if s_result.exit_code:
            s = s_result.orbit
    s = rediscretize(s, new_shape=(64, s.M))
    s.convert(to='field').to_h5('EquilibriumOrbitKS_streak.h5', directory='../data/tiles/')

    for new_M in range(w.M, 66, 2):
        new_shape = (w.N, new_M)
        w = rediscretize(w, new_shape=new_shape)
        w_result = converge(w, method='hybrid', verbose=True)
        if w_result.exit_code:
            w = w_result.orbit

    for new_N in range(w.N, 66, 2):
        new_shape = (new_N, w.M)
        w = rediscretize(w, new_shape=new_shape)
        w_result = converge(w, method='hybrid', verbose=True)
        if w_result.exit_code:
            w = w_result.orbit

    w.to_h5('AntisymmetricOrbitKS_wiggle.h5', directory='../data/tiles/')
    change_orbit_type(w, 'OrbitKS').to_h5('OrbitKS_wiggle.h5', directory='../data/tiles/')

    mM = 48
    for new_M in range(m.M, 64, 2):
        m = rediscretize(m, new_shape=(m.N, new_M))
        m_result = converge(m, method='lstsq', verbose=True)
        if m_result.exit_code:
            m = m_result.orbit
            if new_M == 48:
                padded_m_state = np.concatenate((np.zeros([m.N, 8]),
                                                 m.convert(to='field').state,
                                                 np.zeros([m.N, 8])), axis=1)

                RelativeOrbitKS(state=padded_m_state, state_type='field', parameters=m.parameters).to_h5(
                    'RelativeOrbitKS_merger.h5', directory='../data/tiles/padded/', verbose=True)
                padded_m_state = np.concatenate((np.zeros([m.N, 8]),
                                                 m.convert(to='field').change_reference_frame(to='physical').state,
                                                 np.zeros([m.N, 8])), axis=1)
                OrbitKS(state=padded_m_state, state_type='field', parameters=m.parameters).to_h5(
                    'OrbitKS_merger.h5', directory='../data/tiles/padded/', verbose=True)

    for new_M in range(m.M, 66, 2):
        new_shape = (m.N, new_M)
        m = rediscretize(m, new_shape=new_shape)
        m_result = converge(m, method='hybrid', verbose=True)
        if m_result.exit_code:
            m = m_result.orbit

    for new_N in range(m.N, 66, 2):
        new_shape = (new_N, m.M)
        m = rediscretize(m, new_shape=new_shape)
        m_result = converge(m, method='hybrid', verbose=True)
        if m_result.exit_code:
            m = m_result.orbit

    print(s.parameters['field_shape'], w.parameters['field_shape'], m.parameters['field_shape'])
    print(s.residual(), w.residual(), m.residual())
    s.to_h5('OrbitKS_streak.h5', directory='../data/tiles/')
    m.change_reference_frame(to='physical').to_h5('OrbitKS_merger.h5', directory='../data/tiles/')
    w.to_h5('OrbitKS_wiggle.h5', directory='../data/tiles/')

    s.plot(save=True, verbose=True, directory='../figs/tiles/original/')
    w.plot(save=True, verbose=True, directory='../figs/tiles/original/')
    m.plot(save=True, verbose=True, directory='../figs/tiles/original/')

    return None


if __name__=='__main__':
    sys.exit(main())