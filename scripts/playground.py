from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):

    L = 512
    T = 512

    sms = int(L / (2*pi*np.sqrt(2)))
    tms = int(np.sqrt(T/10))

    sigma_time = 20
    sigma_space = 15
    import matplotlib.pyplot as plt
    xscale = int(L / (2*pi*np.sqrt(2)))
    tscale = int(np.sqrt(T/10))
    tvar = 20**2
    xvar = 15**2

    o2 = OrbitKS(state_type='modes', T=512, L=512, seed=0, spectrum='gaussian',
                 xscale=xscale, tscale=tscale, tvar=tvar, xvar=xvar)

    o2 = o2.rescale(0.33, method='power')
    o2 = o2.rescale(2.5, method='absolute')

    result = converge(o, method='grad', verbose=True)

    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/tiles')), '')
    # padded merger Orbit in comoving frame
    merger = read_h5(os.path.abspath(os.path.join(directory, "./OrbitKS_merger_fdomain.h5")))
    # # padded merger orbit in physical frame.
    # merger = read_h5(os.path.abspath(os.path.join(directory, "./OrbitKS_merger_fdomain.h5")))
    # padded wiggle orbit
    wiggle = read_h5(os.path.abspath(os.path.join(directory, "./OrbitKS_wiggle.h5")))
    # padded streak orbit
    streak = read_h5(os.path.abspath(os.path.join(directory, "./OrbitKS_streak.h5")))

    symbol_block = np.array([[1, 2, 0], [0, 1, 1], [2, 1, 1]])
    o = tile(symbol_block)
    test = converge(o, verbose=True)
    print(test.orbit.residual())
    # tscale, xscale, x_var, t_var = (7, 57, 7.54983443527075, 2.6457513110645907)
    # OrbitKS(T=512, L=512, seed=0, magnitude=0.33, rescaling_method='power', spectrum='gaussian', tscale=tscale, xscale=xscale, xvar=x_var, tvar=t_var).plot()
    # s = rediscretize(read_h5('EquilibriumOrbitKS_L6p43.h5', directory='../data/tiles/', state_type='field'), new_shape=(64, 64), class_name='OrbitKS')
    # w = read_h5('AntisymmetricOrbitKS_L17p593_T17p143.h5', directory='../data/tiles/', state_type='field')
    # m = read_h5('RelativeOrbitKS_L13p026_T15p856_fdomain.h5', directory='../data/tiles/', state_type='field')
    # s = change_orbit_type(s, OrbitKS).rescale(magnitude=3.)
    # w = change_orbit_type(w, OrbitKS).rescale(magnitude=3.)
    # m = change_orbit_type(m, OrbitKS).rescale(magnitude=3.)
    #
    # orbit_array = np.array([[m,w,s], [s,s,m], [w,w,s]])
    # glued_orbit = glue(orbit_array).rescale(magnitude=4)
    # glued_orbit = rediscretize(glued_orbit, new_shape=(32,32))
    # glued_orbit.to_h5(directory='../data/gluing/')
    # glued_orbit.plot(save=True, directory='../figs/gluing/')
    # gluing_result = converge(glued_orbit, method='hybrid', verbose=True)
    # gluing_result.orbit.to_h5(directory='../data/gluing/')
    # gluing_result.orbit.plot(save=True, directory='../figs/gluing/')
    return None


if __name__=='__main__':
    sys.exit(main())