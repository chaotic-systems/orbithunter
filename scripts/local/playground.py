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
    t0 = time.time()
    kwargs = {'verbose':True}
    orbit_tol=10**-15
    orbit_n_iter = 0
    orbit_maxiter = 500
    ftol=1e-10
    td = tile_dictionary_ks(padded=False)
    symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    orbit_ = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).reshape(32, 32).convert(to='modes')
    residual = orbit_.residual()
    while residual > orbit_tol:
        step_size = 1
        orbit_n_iter += 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A, b = orbit_.jacobian(), -1*orbit_.spatiotemporal_mapping().state.ravel()
        inv_A = pinv(A)
        dx = orbit_.from_numpy_array(inv_A.dot(b))
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_residual = next_orbit.residual()
        while next_residual > residual and step_size > 10**-6:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.spatiotemporal_mapping()
            next_residual = next_mapping.residual(apply_mapping=False)

        else:
            # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
            if next_residual <= orbit_tol:
                break
            elif step_size <= 10**-6 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
                break
            elif orbit_n_iter >= orbit_maxiter:
                break
            else:
                while next_residual < residual:
                    orbit_ = next_orbit
                    residual = next_residual
                    mapping = next_mapping
                    b = -1*mapping.state.ravel()
                    dx = orbit_.from_numpy_array(inv_A.dot(b))
                    next_orbit = orbit_.increment(dx, step_size=step_size)
                    next_residual = next_orbit.residual()
                    print(step_size)
    #         if kwargs.get('verbose', False):
    #             print(int(np.log2(1/step_size)),end='')
    #             if np.mod(orbit_n_iter, 25) == 0:
    #                 print(' Residual={:.7f} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, 'lstsq'))
    #             sys.stdout.flush()
    t1=time.time()
    print('\n')
    print(t1-t0)

    return None

if __name__=='__main__':
    sys.exit(main())