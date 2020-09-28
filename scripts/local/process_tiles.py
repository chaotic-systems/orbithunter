from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import numpy as np


def zero_pad_border(orbit_, space_padding_shape, time_padding_shape):
    # orbit_ = rediscretize(orbit_, new_shape=(48, 48))
    zeros_time = np.zeros(time_padding_shape)
    zeros_space = np.zeros(space_padding_shape)
    space_framed_state = np.concatenate((zeros_space, orbit_.convert(to='field').state, zeros_space), axis=1)
    if len(zeros_time) > 0:
        spacetime_framed_state = np.concatenate((zeros_time, space_framed_state, zeros_time), axis=0)
    else:
        spacetime_framed_state = space_framed_state
    spacetime_framed_orbit = orbit_.__class__(state=spacetime_framed_state, state_type='field',
                                              parameters=orbit_.parameters)
    return spacetime_framed_orbit


def main(*args, **kwargs):
    s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../../data/local/tiles/original',
                state_type='field')
    w = read_h5('AntisymmetricOrbitKS_L17p590_T17p146.h5', directory='../../data/local/tiles/original',
                state_type='field')
    m = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='../../data/local/tiles/original',
                state_type='field')

    m = dimension_continuation(m, new_size=w.T, verbose=True, method='hybrid',
                               hybrid_maxiter=(10000, 100), precision='very_high').orbit
    s = discretization_continuation(s, new_shape=(64, 64),
                                    verbose=True, method='hybrid',
                                    precision='very_high').orbit.convert(to='field')

    w = discretization_continuation(w, new_shape=(64, 64),
                                    verbose=True, method='hybrid',
                                    precision='very_high').orbit.convert(to='field')
    m = discretization_continuation(m, new_shape=(64, 64),
                                    verbose=True, method='hybrid',
                                    precision='very_high').orbit.convert(to='field')
    mf = m.change_reference_frame(to='physical')
    m_shifted = m.cell_shift(axis=1)
    s_shifted = s.cell_shift(axis=1)
    w_shifted = w.cell_shift(axis=1)
    mf_shifted = mf.cell_shift(axis=1)
    #
    filenames = ['OrbitKS_merger.h5', 'OrbitKS_streak.h5', 'OrbitKS_wiggle.h5', 'OrbitKS_merger_fdomain.h5',
                 'OrbitKS_merger_shifted.h5', 'OrbitKS_merger_fdomain_shifted.h5',
                 'OrbitKS_wiggle_shifted.h5', 'OrbitKS_streak_shifted.h5']
    #
    orbits = [m, s, w, mf, m_shifted, mf_shifted, w_shifted, s_shifted]
    orbits_rescaled = []
    for i, o in enumerate(orbits):
        o.to_h5(filename=filenames[i], directory='../../data/tiles/unpadded/')
        tmp = o.convert(to='field').rescale(np.abs(m.state).max())
        tmp.to_h5(filename=filenames[i], directory='../../data/tiles/rescaled/')
        orbits_rescaled.append(tmp)

    # orbits_rescaled = []
    # for i, o in enumerate(filenames):
    #     orbit_ = read_h5(filename=o, directory='../../data/tiles/rescaled/')
    #     print(orbit_.convert(to='field').state.max())
    #     orbits_rescaled.append(orbit_)

    m, s, w, mf, m_shifted, mf_shifted, w_shifted, s_shifted = orbits_rescaled
    s = rediscretize(s, new_shape=(48, 16))
    w = rediscretize(w, new_shape=(48, 36))
    m = rediscretize(m, new_shape=(48, 48))
    mf = m.change_reference_frame(to='physical')
    m_shifted = m.cell_shift(axis=1)
    s_shifted = s.cell_shift(axis=1)
    w_shifted = w.cell_shift(axis=1)
    mf_shifted = mf.cell_shift(axis=1)
    orbits = [m, s, w, mf, m_shifted, mf_shifted, w_shifted, s_shifted]

    for i, o in enumerate(orbits):
        o_space_not_time = rediscretize(o, new_shape=(64, o.shape[1]))
        o_space_not_time_pad = zero_pad_border(o_space_not_time, (64, (64-o.shape[1])//2),
                                               (0, o_space_not_time.shape[1]))
        o_pad = zero_pad_border(o, (o.shape[0], (64-o.shape[1])//2), ((64-o.shape[0])//2, 64))

        o_space_not_time_pad.to_h5(filename=filenames[i], directory='../../data/tiles/padded_space_unpadded_time/',
                                   verbose=True)
        o_pad.to_h5(filename=filenames[i],
                    directory='../../data/tiles/padded/', verbose=True)

    # m_pad = zero_pad_border(m, (48, 8), (8, 64))
    # s_pad = zero_pad_border(s, (48, 8), (8, 64))
    # w_pad = zero_pad_border(w, (48, 8), (8, 64))
    # mf_pad = zero_pad_border(mf, (48, 8), (8, 64))

    # m_shift_pad = zero_pad_border(m_shifted, (48, 8), (8, 64))
    # s_shift_pad = zero_pad_border(s_shifted, (48, 8), (8, 64))
    # w_shift_pad = zero_pad_border(w_shifted, (48, 8), (8, 64))
    # mf_shift_pad = zero_pad_border(mf_shifted, (48, 8), (8, 64))

    # m = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='../../data/tiles/original', state_type='field', data_format='orbithunter_old')
    # m_result = dimension_continuation(m, w.T, verbose=True, method='lstsq')
    # mlstsq = m_result.orbit
    # mlstsq.plot()

    return None


if __name__=='__main__':
    sys.exit(main())