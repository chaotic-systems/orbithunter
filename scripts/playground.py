from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    # test = OrbitKS(seed=0, orbit_parameters=(178.8029091781049, 41.70053340290405, 0.),
    #                spectrum='exponential', tscale=1, xscale=52, xvar=20, tvar=1)

    orbit_ = read_h5('OrbitKS_L26p744_T54p129.h5', directory='../data/examples/', state_type='modes')


    orbit_.L += 0.01
    converge_result = converge(orbit_,  method='gradient_descent', orbit_tol=10**-15,
                               verbose=True, orbit_maxiter=1000)

    return None

if __name__=='__main__':
    sys.exit(main())