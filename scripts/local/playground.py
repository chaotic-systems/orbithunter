from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import glob

def main(*args, **kwargs):
    from orbithunter.gluing import tile_dictionary_ks
    td = tile_dictionary_ks(padded=False)
    # td = tile_dictionary_ks()

    symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    tiled_orbit = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).reshape(32, 32).convert(to='modes')
    res = converge(tiled_orbit, method='minres', verbose=True)
    res.orbit.plot()

    return None

if __name__=='__main__':
    sys.exit(main())