from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    td[0].plot()

    td[1].plot()

    td[2].plot()

    orbit_ = read_h5('../../data/examples/OrbitKS_L26p744_T54p129.h5')
    orbit_.plot()
    t0 = time.time()
    for i in range(10000):
        test = orbit_.copy()
    t1 = time.time()
    print(t1-t0)
    # orbit_ = rediscretize(orbit_, new_shape=(64,64)).convert(to='modes')
    J = orbit_.jacobian()
    rmatvec_direct = np.dot(J.transpose(), orbit_.state.ravel()).ravel()
    matvec_direct = np.dot(J, orbit_.state_vector().ravel()).ravel()
    test = matvec_direct - orbit_.matvec(orbit_).state.ravel()
    test2 = rmatvec_direct - orbit_.rmatvec(orbit_).state_vector().ravel()
    orbit_.L += 0.02
    t0 = time.time()
    result = converge(orbit_, method='gradient_descent', verbose=True, orbit_maxiter=10000, ftol=0, preconditioning=True)
    t1 = time.time()
    print(t1-t0, (t1-t0)/10000)
    cdict = {1:'success', 0:'failure', 2:'success'}
    orbit_list = []
    res, tim = [], []
    max_iter = 3
    # wide_eqv = EquilibriumOrbitKS(seed=1, orbit_parameters=(0., 1000., 0.)).rescale(2)
    #
    # converge_result = converge(wide_eqv, verbose=True, preconditioning=True)
    # converge_result2 = converge(wide_eqv,  verbose=True, preconditioning=False)

    # final_result = converge(converge_result.orbit, method='l-bfgs-b', verbose=True, preconditioning=False)

    return None

if __name__=='__main__':
    sys.exit(main())