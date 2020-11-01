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
    from orbithunter.gluing import tile_dictionary_ks
    td = tile_dictionary_ks(padding=False, comoving=True)
    # td[0].plot()
    # td[1].plot()
    # td[2].plot()

    orbit_1_2 = tile(np.array([[1],[2]]), td, OrbitKS)

    # orbit_1_2.plot()
    result_12 = converge(orbit_1_2.reshape(), method='adj', verbose=True)
    result_12 = converge(orbit_1_2.reshape(8,8).reshape(16,16), method='adj', verbose=True)

    result_12.orbit.plot()
    print(result_12.orbit.residual())
    #
    # test2 = read_h5('ShiftReflectionOrbitKS_T124p436_L61p389.h5')
    # # test2.plot(fundamental_domain=False)
    # # test2 = test2.reshape(test.field_shape)
    # # test2.plot(fundamental_domain=False)
    # plt.matshow(np.log10(np.abs(test2.state)))
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # print(np.log10(np.abs(test.state)).mean(axis=0), np.log10(np.abs(test2.state)).mean(axis=0))
    # plt.plot(np.log10(np.abs(test.state)).mean(axis=0)[:int(2*test.L/(2*pi*np.sqrt(2)))])
    # plt.plot(np.log10(np.abs(test2.state)).mean(axis=0)[:int(2*test2.L/(2*pi*np.sqrt(2)))])
    # plt.show()
    # #
    # # plt.figure()
    # # print(np.log10(np.abs(test.state)).mean(axis=1), np.log10(np.abs(test2.state)+1).mean(axis=1))
    # # # plt.plot(np.log10(np.abs(test.state)).mean(axis=1))
    # # plt.plot(np.log10(np.abs(test2.state)).mean(axis=1))
    # # plt.show()
    #
    #
    # for padding in ['spacetime', 'space', False]:
    #     tile_dictionary = tile_dictionary_ks(padding=padding, comoving=False)
    #     symbols = np.array([[2, 0, 0],
    #                         [1, 0, 1],
    #                         [0, 0, 0],
    #                         [0, 2, 0],
    #                         [2, 2, 2]])
    #     orbit_ = tile(symbols, tile_dictionary, OrbitKS).reshape()
    #     orbit_.plot()
    #     plt.matshow(np.log10(np.abs(orbit_.convert(to='modes').state)))
    #     plt.colorbar()
    #     plt.show()



    # # orbit_ = tile(symbols, tile_dictionary, OrbitKS, stripwise=True, gluing_order=(1, 0)).reshape()
    # orbit_.to_h5(directory= '../../data/local/testing/')
    # result = converge(orbit_, method='hybrid', **kwargs)
    return None

if __name__=='__main__':
    sys.exit(main())