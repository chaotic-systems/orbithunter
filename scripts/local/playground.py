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
    tmp = OrbitKS(seed=0, parameters=(512, 512), M=1024)
    # tmp.plot()
    int1 = kse_integrate(tmp, verbose=True, integration_time=1000, step_size=0.25)
    # orbit_ = read_h5('ppo_L21p90_T96p68.h5', directory="C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\", data_format='kstori')
    # result = discretization_continuation(orbit_, new_shape=(32, 32), verbose=True, plot_intermediates=True, ftol=1e-8, precision='very_high', method='lstsq')
    # from orbithunter.gluing import tile_dictionary_ks
    # td = tile_dictionary_ks(padded=True)
    # tile_dictionary = rediscretize_tiling_dictionary(tile_dictionary_ks(padded=True), new_shape=(16, 16))
    # glue_shape = (3, 3)
    # symbol_arrays = generate_symbol_arrays(tile_dictionary, glue_shape=glue_shape, unique=True)
    #
    # for sa in symbol_arrays:
    #     tiled_initial_condition = tile(sa, tile_dictionary, OrbitKS)
    #     orbit_complex()
    #     result = converge(tiled_initial_condition, method='hybrid', precision='very_high', verbose=True)
    #
    # test = list(glob.glob('C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\trawl\\rpo\\data\\**\\*.h5', recursive=True))
    # for orbit_h5 in test:
    #     try:
    #         orbit_ = read_h5(orbit_h5, data_format='kstori', check=True)
    #     except KeyError:
    #         try:
    #             orbit_ = read_h5(orbit_h5, data_format='orbithunter_old', check=True)
    #         except KeyError:
    #             orbit_ = read_h5(orbit_h5,  check=True)
    #     print(orbit_, orbit_.field_shape)
    return None

if __name__=='__main__':
    sys.exit(main())