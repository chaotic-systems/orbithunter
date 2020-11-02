from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import glob
from scipy.linalg import pinv

def main(*args, **kwargs):
    spectrum_tactics = ['linear','linear-exponential','plateau-linear','random']
    for st in spectrum_tactics:
        example_orbit = OrbitKS(seed=5, parameters=(66, 22, 0.), spectrum=st)
        example_orbit.plot()
        plt.matshow(np.log10(np.abs(example_orbit.convert(to='modes').state)))
        plt.colorbar()
        plt.show()


    # # orbit_ = tile(symbols, tile_dictionary, OrbitKS, stripwise=True, gluing_order=(1, 0)).reshape()
    # orbit_.to_h5(directory= '../../data/local/testing/')
    # result = converge(orbit_, method='hybrid', **kwargs)
    return None

if __name__=='__main__':
    sys.exit(main())