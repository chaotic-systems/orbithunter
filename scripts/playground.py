from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    s = rediscretize(read_h5('EquilibriumOrbitKS_L6p43.h5', directory='../data/tiles/', state_type='field'), new_shape=(64, 64), class_name='OrbitKS')
    w = read_h5('AntisymmetricOrbitKS_L17p593_T17p143.h5', directory='../data/tiles/', state_type='field')
    m = read_h5('RelativeOrbitKS_L13p026_T15p856_fdomain.h5', directory='../data/tiles/', state_type='field')
    s = change_orbit_type(s, OrbitKS).rescale(magnitude=3.)
    w = change_orbit_type(w, OrbitKS).rescale(magnitude=3.)
    m = change_orbit_type(m, OrbitKS).rescale(magnitude=3.)

    orbit_array = np.array([[m,w,s], [s,s,m], [w,w,s]])
    glued_orbit = glue(orbit_array).rescale(magnitude=4)
    glued_orbit = rediscretize(glued_orbit, new_shape=(32,32))
    glued_orbit.to_h5(directory='../data/gluing/')
    glued_orbit.plot(save=True, directory='../data/gluing/')
    gluing_result = converge(glued_orbit, method='hybrid', verbose=True)
    gluing_result.orbit.to_h5(directory='../data/gluing/')
    gluing_result.orbit.plot(save=True, directory='../data/gluing/')
    return None


if __name__=='__main__':
    sys.exit(main())