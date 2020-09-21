from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import numpy as np
import matplotlib.pyplot as plt

def main(*args, **kwargs):
    s = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='../data/tiles/original', state_type='field')
    w = read_h5('AntisymmetricOrbitKS_L17p590_T17p146.h5', directory='../data/tiles/original', state_type='field')
    m = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='../data/tiles/original', state_type='field')
    re = read_h5('RelativeEquilibriumOrbitKS_L31p749_T20p147.h5', state_type='field')
    sr = read_h5('ShiftReflectionOrbitKS_L68p444_T61p659.h5', state_type='field')
    simple = False
    rescale = False
    # sr.plot(padding=False)
    sr_result = discretization_continuation(sr, (32, 96), verbose=True)
    # reqv = discretization_continuation(re, (58, 48), verbose=True)
    # streak_result = discretization_continuation(rediscretize(s, new_shape=(32, 32)), (64, 24), verbose=True)
    # wiggle_result = discretization_continuation(w, (64, 64), verbose=True)
    # merger_result = discretization_continuation(w, (58, 48), verbose=True)

    # s_ready_for_padding = streak_result.orbit
    # w_padded = wiggle_result.orbit
    # m_ready_for_padding = merger_result.orbit
    sr_result.orbit.plot()
    return None


if __name__=='__main__':
    sys.exit(main())