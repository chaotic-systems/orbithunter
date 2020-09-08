from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, method='adj', **kwargs):
    test = read_h5('OrbitKS_L37p297_T79p778.h5', directory='local')
    orbit_ = rediscretize(test, new_N=64, new_M=32)
    np.random.seed(0)
    orbit_ = orbit_ + OrbitKS(state=0.1*np.random.randn(*orbit_.state.shape), state_type='field')
    orbit_ = orbit_.convert(to='modes')
    outer_tol = orbit_.N*orbit_.M*10**-6
    result = converge(orbit_, method='gradient', verbose=True)
    # t0 = time.time()
    # result0 = converge(orbit_, method='lsqr', verbose=True)
    # t1 = time.time()
    # print('cg off', (1-(result0.orbit.residual()/orbit_.residual()))/(t1-t0), result0.orbit.residual()<outer_tol)

    # t0 = time.time()
    # result0 = converge(orbit_, method='newton-cg', verbose=True)
    # t1 = time.time()
    # print('newton-cg off', (1-(result0.orbit.residual()/orbit_.residual()))/(t1-t0), result0.orbit.residual()<outer_tol)
    #
    # t0 = time.time()
    # result0 = converge(orbit_, method='l-bfgs-b', verbose=True)
    # t1 = time.time()
    # print('l-bfgs-b on', (1-(result0.orbit.residual()/orbit_.residual()))/(t1-t0), result0.orbit.residual()<outer_tol)
    #
    # t0 = time.time()
    # result0 = converge(orbit_, method='tnc', verbose=True)
    # t1 = time.time()
    # print('tnc on', (1-(result0.orbit.residual()/orbit_.residual()))/(t1-t0), result0.orbit.residual()<outer_tol)
    #
    t0 = time.time()
    result0 = converge(orbit_, method='gradient', verbose=True)
    t1 = time.time()
    print('adj on', (1-(result0.orbit.residual()/orbit_.residual()))/(t1-t0), result0.orbit.residual()<outer_tol)

    return None


if __name__=='__main__':
    main()
