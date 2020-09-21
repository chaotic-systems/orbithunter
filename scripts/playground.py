from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    o = read_h5('none_1112_initial.h5', data_format='kstori')
    from orbithunter.gluing import tile_dictionary_ks
    td = tile_dictionary_ks(padded=False)
    symbol_array = np.array([[1,1,1,2]]).reshape(1, 4)
    o = rediscretize(tile(symbol_array, td, OrbitKS), new_shape=(32, 32))
    # o.plot()
    result = converge(o.rescale(2.5), precision='machine', orbit_maxiter=10000, method='gradient_descent', verbose=True)
    result = converge(result.orbit, precision='machine', orbit_maxiter=10000, method='lstsq', verbose=True)
    result.orbit.plot(show=True)
    return None

if __name__=='__main__':
    sys.exit(main())