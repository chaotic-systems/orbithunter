from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import numpy as np
import matplotlib.pyplot as plt

def main(*args, **kwargs):
    s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../../data/tiles/original', basis='field', data_format='orbithunter_old')
    w = read_h5('AntisymmetricOrbitKS_L17p590_T17p146.h5', directory='../../data/tiles/original', basis='field', data_format='orbithunter_old')
    m = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='../../data/tiles/original', basis='field', data_format='orbithunter_old')

    simple = True
    rescale = True
    rscle = np.abs(m.convert(to='field').state).max()
    if simple:
        s = rediscretize(s, new_shape=(s.N, 24))
        if rescale:
            s_rescaled = s.rescale(rscle)
        else:
            s_rescaled = s
        s_shifted = s_rescaled.cell_shift(axis=1)
        padded_s_state_shifted = np.tile(np.concatenate((np.zeros([s.N, 20]),
                                                s_shifted.convert(to='field').state,
                                                np.zeros([s.N, 20])), axis=1), (48, 1))
        padded_s_state = np.tile(np.concatenate((np.zeros([s.N, 20]),
                                                s_rescaled.convert(to='field').state,
                                                np.zeros([s.N, 20])), axis=1), (48, 1))
        padded_s_state_shifted = np.concatenate((np.zeros([8,padded_s_state_shifted.shape[1]]), padded_s_state_shifted,
                                                 np.zeros([8,padded_s_state_shifted.shape[1]])), axis=0)
        padded_s_state_shifted = np.concatenate((np.zeros([8,padded_s_state.shape[1]]), padded_s_state_shifted,
                                             np.zeros([8,padded_s_state.shape[1]])), axis=0)

        padded_s_orbit = OrbitKS(state=padded_s_state, basis='field', parameters=s.parameters)
        padded_s_orbit_shifted = OrbitKS(state=padded_s_state_shifted, basis='field', parameters=s.parameters)

        s_orbit = convert_class(rediscretize(s_rescaled, new_shape=(64, 64)).convert(to='field'), OrbitKS)

        padded_s_orbit.to_h5('OrbitKS_streak.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_s_orbit.plot(filename='OrbitKS_streak.png', directory='../../figs/tiles/padded/',
                            verbose=True, show=False)

        s_orbit.to_h5('OrbitKS_streak.h5', directory='../../data/tiles/', verbose=True)
        s_orbit.plot(filename='OrbitKS_streak.png', directory='../../figs/tiles/', verbose=True, show=False)

        s_orbit_shifted = convert_class(rediscretize(s_shifted, new_shape=(64, 64)).convert(to='field'), OrbitKS)
        padded_s_orbit_shifted.to_h5('OrbitKS_streak_shifted.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_s_orbit_shifted.plot(filename='OrbitKS_streak_shifted.png', directory='../../figs/tiles/padded/',
                            verbose=True, show=False)

        s_orbit_shifted.to_h5('OrbitKS_streak_shifted.h5', directory='../../data/tiles/', verbose=True)
        s_orbit_shifted.plot(filename='OrbitKS_streak_shifted.png', directory='../../figs/tiles/', verbose=True, show=False)


        w = rediscretize(w, new_shape=(48, 64))
        wpad = np.concatenate((np.zeros([8, w.field_shape[1]]), w.state, np.zeros([8, w.field_shape[1]])), axis=0)
        if rescale:
            w_rescaled = w.rescale(rscle)
        else:
            w_rescaled = w
        # Note padded version does not exist because wiggle doesn't need it due to its size; other orbits padding
        # is based upon relative sizes.

        w_orbit = convert_class(w_rescaled.convert(to='field'), OrbitKS)
        w_orbit.to_h5('OrbitKS_wiggle.h5', directory='../../data/tiles/', verbose=True)
        w_orbit.plot(filename='OrbitKS_wiggle.png', directory='../../figs/tiles/', verbose=True, show=False)
        w_orbit.to_h5('OrbitKS_wiggle.h5', directory='../../data/tiles/padded/', verbose=True)
        w_orbit.plot(filename='OrbitKS_wiggle.png', directory='../../figs/tiles/padded/', verbose=True, show=False)

        w_orbit_shifted = w_orbit.cell_shift(axis=1)
        w_orbit_shifted.to_h5('OrbitKS_wiggle_shifted.h5', directory='../../data/tiles/', verbose=True)
        w_orbit_shifted.plot(filename='OrbitKS_wiggle_shifted.png', directory='../../figs/tiles/', verbose=True, show=False)
        w_orbit_shifted.to_h5('OrbitKS_wiggle_shifted.h5', directory='../../data/tiles/padded/', verbose=True)
        w_orbit_shifted.plot(filename='OrbitKS_wiggle_shifted.png', directory='../../figs/tiles/padded/', verbose=True, show=False)

        m = rediscretize(m, new_shape=(64, 48))
        if rescale:
            m_rescaled = m.rescale(rscle)
        else:
            m_rescaled = m

        padded_rel_m_state = np.concatenate((np.zeros([m.N, 8]),
                                         m_rescaled.convert(to='field').state,
                                         np.zeros([m.N, 8])), axis=1)

        padded_m_state = np.concatenate((np.zeros([m.N, 8]),
                                         m_rescaled.convert(to='field').change_reference_frame(to='physical').state,
                                         np.zeros([m.N, 8])), axis=1)

        padded_rel_m_state_shifted = np.concatenate((np.zeros([m.N, 8]),
                                         m_rescaled.cell_shift(axis=1).convert(to='field').state,
                                         np.zeros([m.N, 8])), axis=1)

        padded_m_state_shifted = np.concatenate((np.zeros([m.N, 8]),
                                         m_rescaled.cell_shift(axis=1).convert(to='field').change_reference_frame(to='physical').state,
                                         np.zeros([m.N, 8])), axis=1)

        padded_m = OrbitKS(state=padded_m_state, basis='field', parameters=m.parameters)
        padded_m.to_h5('OrbitKS_merger_fdomain.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_m.plot(filename='OrbitKS_merger_fdomain.png',
                          directory='../../figs/tiles/padded/', verbose=True, show=False)

        padded_mf = OrbitKS(state=padded_rel_m_state, basis='field', parameters=m.parameters)
        padded_mf.to_h5('OrbitKS_merger.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_mf.plot(filename='OrbitKS_merger.png',
                       directory='../../figs/tiles/padded/', verbose=True, padding=False, show=False)

        padded_m_shifted = OrbitKS(state=padded_m_state_shifted, basis='field', parameters=m.parameters)
        padded_m_shifted.to_h5('OrbitKS_merger_shifted_fdomain.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_m_shifted.plot(filename='OrbitKS_merger_shifted_fdomain.png',
                          directory='../../figs/tiles/padded/', verbose=True, show=False)

        padded_mf_shifted = OrbitKS(state=padded_rel_m_state_shifted, basis='field', parameters=m.parameters)
        padded_mf_shifted .to_h5('OrbitKS_merger_shifted.h5', directory='../../data/tiles/padded/', verbose=True)
        padded_mf_shifted .plot(filename='OrbitKS_merger_shifted.png',
                      directory='../../figs/tiles/padded/', verbose=True, padding=False, show=False)

        ####################

        m_rescaled = rediscretize(m_rescaled, new_shape=(64, 64))
        m_shifted = m_rescaled.cell_shift(axis=1)

        m_orbit_shifted = convert_class(m_shifted.convert(to='field'), OrbitKS)
        m_orbitf_shifted = convert_class(m_shifted.convert(to='field').change_reference_frame(to='physical'), OrbitKS)

        m_orbit_shifted.to_h5('OrbitKS_merger_shifted.h5', directory='../../data/tiles/', verbose=True)
        m_orbit_shifted.plot(filename='OrbitKS_merger_shifted.png', directory='../../figs/tiles/', verbose=True, show=False)

        m_orbitf_shifted.to_h5('OrbitKS_merger_shifted_fdomain.h5', directory='../../data/tiles/', verbose=True)
        m_orbitf_shifted.plot(filename='OrbitKS_merger_shifted_fdomain.png', directory='../../figs/tiles/', verbose=True, show=False)

        m_orbit = convert_class(m_rescaled.convert(to='field'), OrbitKS)
        m_orbitf = convert_class(m_rescaled.convert(to='field').change_reference_frame(to='physical'), OrbitKS)

        m_orbit.to_h5('OrbitKS_merger.h5', directory='../../data/tiles/', verbose=True)
        m_orbit.plot(filename='OrbitKS_merger.png', directory='../../figs/tiles/', verbose=True, show=False)

        m_orbitf.to_h5('OrbitKS_merger_fdomain.h5', directory='../../data/tiles/', verbose=True)
        m_orbitf.plot(filename='OrbitKS_merger_fdomain.png', directory='../../figs/tiles/', verbose=True, show=False)

    else:
        # Two versions of the tiles; one has zero padding to have the relative sizes remain accurate.
        sM = 24
        mM = 48

        for new_M in range(s.M, 22, -2):
            s = rediscretize(s, new_shape=(s.N, new_M))
            s_result = converge(s,  precision='machine', verbose=True)
            if s_result.exit_code:
                s = s_result.orbit
                if new_M == 24:
                    if rescale:
                        s_rescaled = s.rescale(rscle)
                    else:
                        s_rescaled = s
                    padded_s_state = np.tile(np.concatenate((np.zeros([s.N, 20]),
                                                            s_rescaled.convert(to='field').state,
                                                            np.zeros([s.N, 20])), axis=1), (64, 1))

                    padded_s = EquilibriumOrbitKS(state=padded_s_state, basis='field', parameters=s.parameters)
                    padded_s_orbit = OrbitKS(state=padded_s_state, basis='field', parameters=s.parameters)

                    padded_s.to_h5('EquilibriumOrbitKS_streak.h5', directory='../../data/tiles/padded/', verbose=True)
                    padded_s.plot(filename='EquilibriumOrbitKS_streak.png', directory='../../figs/tiles/padded/',
                                  verbose=True)

                    padded_s_orbit.to_h5('OrbitKS_streak.h5', directory='../../data/tiles/padded/', verbose=True)
                    padded_s_orbit.plot(filename='OrbitKS_streak.png', directory='../../figs/tiles/padded/',
                                        verbose=True)

        s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../../data/tiles/original')
        for new_M in range(s.M, 66, 2):
            s = rediscretize(s, new_shape=(s.N, new_M))
            s_result = converge(s, precision='machine',  verbose=True)
            if s_result.exit_code:
                s = s_result.orbit
        s = rediscretize(s, new_shape=(64, s.M))
        if rescale:
            s_rescaled = s.rescale(rscle)
        else:
            s_rescaled = s

        convert_class(s_rescaled.convert(to='field'), OrbitKS).to_h5('OrbitKS_streak.h5', directory='../../data/tiles/')
        convert_class(s_rescaled.convert(to='field'), OrbitKS).to_h5('OrbitKS_streak.png', directory='../../figs/tiles/')

        for new_M in range(w.M, 66, 2):
            new_shape = (w.N, new_M)
            w = rediscretize(w, new_shape=new_shape)
            w_result = converge(w, precision='machine',  verbose=True)
            if w_result.exit_code:
                w = w_result.orbit

        for new_N in range(w.N, 66, 2):
            new_shape = (new_N, w.M)
            w = rediscretize(w, new_shape=new_shape)
            w_result = converge(w, precision='machine',  verbose=True)
            if w_result.exit_code:
                w = w_result.orbit

        # Note padded version does not exist because wiggle doesn't need it due to its size; other orbits padding
        # is based upon relative sizes.
        if rescale:
            w_rescaled = w.rescale(rscle)
        else:
            w_rescaled = w

        w_orbit = convert_class(w_rescaled.convert(to='field'), OrbitKS)
        w_orbit.convert(to='field').to_h5('OrbitKS_wiggle.h5', directory='../../data/tiles/', verbose=True)
        w_orbit.convert(to='field').plot(filename='OrbitKS_wiggle.png', directory='../../figs/tiles/', verbose=True)

        w_rescaled.convert(to='field').to_h5('AntisymmetricOrbitKS_wiggle.h5', directory='../../data/tiles/padded/', verbose=True)
        w_rescaled.convert(to='field').plot(filename='AntisymmetricOrbitKS_wiggle.png', directory='../../figs/tiles/padded/', verbose=True)

        w_orbit.convert(to='field').to_h5('OrbitKS_wiggle.h5', directory='../../data/tiles/padded/', verbose=True)
        w_orbit.convert(to='field').plot(filename='OrbitKS_wiggle.png', directory='../../figs/tiles/padded/', verbose=True)
        mM = 48
        for new_N in range(m.N, 66, 2):
            new_shape = (new_N, m.M)
            m = rediscretize(m, new_shape=new_shape)
            m_result = converge(m, precision='machine',  verbose=True)
            if m_result.exit_code:
                m = m_result.orbit

        for new_M in range(m.M, 66, 2):
            m = rediscretize(m, new_shape=(m.N, new_M))
            m_result = converge(m, precision='machine',  verbose=True)
            if m_result.exit_code:
                m = m_result.orbit
                if new_M == 48:
                    if rescale:
                        m_rescaled = m.rescale(rscle)
                    else:
                        m_rescaled = m
                    padded_rel_m_state = np.concatenate((np.zeros([m.N, 8]),
                                                     m_rescaled.convert(to='field').state,
                                                     np.zeros([m.N, 8])), axis=1)
                    padded_m_state = np.concatenate((np.zeros([m.N, 8]),
                                                     m_rescaled.convert(to='field').change_reference_frame(to='physical').state,
                                                     np.zeros([m.N, 8])), axis=1)

                    padded_rel_m = RelativeOrbitKS(state=padded_rel_m_state, basis='field', parameters=m.parameters)
                    padded_m = OrbitKS(state=padded_m_state, basis='field', parameters=m.parameters)
                    padded_rel_m.to_h5('RelativeOrbitKS_merger.h5', directory='../../data/tiles/padded/', verbose=True)
                    padded_rel_m.plot(filename='RelativeOrbitKS_merger.png',
                                      directory='../../figs/tiles/padded/', verbose=True)

                    padded_rel_m = OrbitKS(state=padded_rel_m_state, basis='field', parameters=m.parameters)
                    padded_rel_m.to_h5('OrbitKS_merger_fdomain.h5', directory='../../data/tiles/padded/', verbose=True)
                    padded_rel_m.plot(filename='OrbitKS_merger_fdomain.png',
                                      directory='../../figs/tiles/padded/', verbose=True)

                    padded_m.to_h5('OrbitKS_merger.h5', directory='../../data/tiles/padded/', verbose=True)
                    padded_m.plot(filename='OrbitKS_merger.png',
                                  directory='../../figs/tiles/padded/', verbose=True, padding=False)

        if rescale:
            m_rescaled = m.rescale(rscle)
        else:
            m_rescaled = m

        rel_m = m_rescaled.copy()

        m = convert_class(m.convert(to='field').change_reference_frame(to='physical'), OrbitKS)
        m.to_h5('OrbitKS_merger.h5', directory='../../../data/tiles/', verbose=True)
        m.plot(filename='OrbitKS_merger.png', directory='../../figs/tiles/', verbose=True)

        print(s.field_shape, w.field_shape, m.field_shape)
        print(s.residual(), w.residual(), rel_m.residual(), m.residual())
    return None


if __name__=='__main__':
    sys.exit(main())