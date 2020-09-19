from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    test = RelativeOrbitKS(orbit_parameters=(1., 1., 1.), frame='physical')
    comoving=False
    if comoving:
        # padded merger Orbit in comoving frame
        merger = read_h5('OrbitKS_merger.h5', directory='../data/tiles/')
    else:
        # padded merger orbit in physical frame.
        merger =read_h5('OrbitKS_merger_fdomain.h5', directory='../data/tiles/')

    # padded streak orbit
    streak = read_h5('OrbitKS_streak.h5', directory='../data/tiles/')

    # padded wiggle orbit
    wiggle = read_h5('OrbitKS_wiggle.h5', directory='../data/tiles/')

    td = {0 : streak, 1: merger, 2: wiggle}
    test = OrbitKS()
    symbol_array = np.array([[0, 1, 2], [2, 2, 2], [1, 0, 0]])

    test = tile(symbol_array, td, OrbitKS)
    test.plot()
    testc = clip(test, ((None, None), (None, 20)))
    testc.plot()
    return None

if __name__=='__main__':
    sys.exit(main())