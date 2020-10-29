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
    #
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
    mapping = orbit_.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)
    step_size = 0.01
    # #
    # while residual > orbit_tol:
    #     orbit_n_iter += 1
    #     # Solve A dx = b <--> J dx = - f, for dx.
    #     A, b = orbit_.jacobian(), -1*mapping.state.ravel()
    #     inv_A = pinv(A)
    #     dx = orbit_.from_numpy_array(inv_A.dot(b))
    #     next_orbit = orbit_.increment(dx, step_size=step_size)
    #     next_mapping = next_orbit.spatiotemporal_mapping()
    #     next_residual = next_mapping.residual(apply_mapping=False)
    #     while next_residual > residual and step_size > 10**-6:
    #         # Continues until either step is too small or residual decreases
    #         step_size /= 2.0
    #         next_orbit = orbit_.increment(dx, step_size=step_size)
    #         next_mapping = next_orbit.spatiotemporal_mapping()
    #         next_residual = next_mapping.residual(apply_mapping=False)
    #     else:
    #         n_inner = 0
    #         b = -1 * next_mapping.state.ravel()
    #         # x_1 = x_0 + dx_1 + dx_2 + dx_3 ...
    #         dx_inner = orbit_.from_numpy_array(inv_A.dot(b))
    #         inner_orbit_ = next_orbit.increment(dx_inner, step_size=step_size)
    #         inner_mapping = inner_orbit_.spatiotemporal_mapping()
    #         inner_residual = inner_mapping.residual(apply_mapping=False)
    #         while inner_residual <= next_residual:
    #             n_inner+=1
    #             next_orbit = inner_orbit_
    #             next_mapping = inner_mapping
    #             next_residual = inner_residual
    #             b = -1 * next_mapping.state.ravel()
    #             # x_1 = x_0 + dx_1 + dx_2 + dx_3 ...
    #             dx_inner = orbit_.from_numpy_array(inv_A.dot(b))
    #             inner_orbit_ = next_orbit.increment(dx_inner, step_size=step_size)
    #             inner_mapping = inner_orbit_.spatiotemporal_mapping()
    #             inner_residual = inner_mapping.residual(apply_mapping=False)
    #         else:
    #             orbit_ = next_orbit
    #             mapping = next_mapping
    #             residual = next_residual
    #             print(n_inner, residual, step_size)

    # #
    # t1 = time.time()
    # print('took {} seconds'.format(t1-t0))
    ########################################
    t0 = time.time()
    kwargs = {'verbose':True}
    orbit_tol=10**-15
    orbit_n_iter = 0
    orbit_maxiter = 500
    ftol=1e-10
    td = tile_dictionary_ks(padded=False)
    symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    orbit_ = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).reshape(32, 32).convert(to='modes')
    mapping = orbit_.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)
    step_size = 0.01

    # while residual > orbit_tol:
    #     orbit_n_iter += 1
    #     # Solve A dx = b <--> J dx = - f, for dx.
    #     A, b = orbit_.jacobian(), -1.0*mapping.state.ravel()
    #     inv_A = pinv(A)
    #     dx = orbit_.from_numpy_array(inv_A.dot(b))
    #     next_orbit = orbit_.increment(dx, step_size=step_size)
    #     next_mapping = next_orbit.spatiotemporal_mapping()
    #     next_residual = next_mapping.residual(apply_mapping=False)
    #     while next_residual > residual and step_size > 10**-6:
    #         # Continues until either step is too small or residual decreases
    #         step_size /= 2.0
    #         next_orbit = orbit_.increment(dx, step_size=step_size)
    #         next_mapping = next_orbit.spatiotemporal_mapping()
    #         next_residual = next_mapping.residual(apply_mapping=False)
    #     else:
    #         n_inner = 0
    #         if step_size < 10**-6:
    #             print('step too small')
    #             break
    #         while next_residual < residual:
    #             n_inner+=1
    #             orbit_ = next_orbit
    #             mapping = next_mapping
    #             residual = next_residual
    #             b = -1 * mapping.state.ravel()
    #             dx = orbit_.from_numpy_array(inv_A.dot(b))
    #             next_orbit = orbit_.increment(dx, step_size=step_size)
    #             next_mapping = next_orbit.spatiotemporal_mapping()
    #             next_residual = next_mapping.residual(apply_mapping=False)
    #         print(n_inner, residual, step_size)
    # t1 = time.time()
    # print('took {} seconds'.format(t1-t0))
    #
    #


    ########################################
    t0 = time.time()
    kwargs = {'verbose':True}
    orbit_tol=10**-15
    orbit_n_iter = 0
    orbit_maxiter = 500
    ftol=1e-10
    td = tile_dictionary_ks(padded=False)
    symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    orbit_ = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).reshape(32, 32).convert(to='modes')
    mapping = orbit_.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)
    step_size = 0.01

    while residual > orbit_tol:
        orbit_n_iter += 1
        # Solve A dx = b <--> J dx = - f, for dx.
        J, JT, f = orbit_.jacobian(), orbit_.jacobian_transpose(), mapping.state.ravel()
        A = JT.dot(J)
        b = -1*JT.dot(f)
        inv_A = pinv(A)
        dx = orbit_.from_numpy_array(inv_A.dot(b))
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.spatiotemporal_mapping()
        next_residual = next_mapping.residual(apply_mapping=False)
        while next_residual > residual and step_size > 10**-6:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.spatiotemporal_mapping()
            next_residual = next_mapping.residual(apply_mapping=False)
        else:
            n_inner = 0
            if step_size < 10**-6:
                print('step too small')
                break
            while next_residual < residual:
                n_inner+=1
                orbit_ = next_orbit
                mapping = next_mapping
                residual = next_residual
                f = mapping.state.ravel()
                b = -1*JT.dot(f)
                dx = orbit_.from_numpy_array(inv_A.dot(b))
                next_orbit = orbit_.increment(dx, step_size=step_size)
                next_mapping = next_orbit.spatiotemporal_mapping()
                next_residual = next_mapping.residual(apply_mapping=False)
    #         print(np.abs(dx.state).min())
            print(n_inner, residual, step_size)
    t1 = time.time()
    print('took {} seconds'.format(t1-t0))



    # from orbithunter.gluing import tile_dictionary_ks
    # td = rediscretize_tiling_dictionary(tile_dictionary_ks(padded=False), new_shape=(512,512))
    # # td[0].plot()
    # # td[1].plot()
    # # td[2].plot()
    #
    # symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    # orbit_ = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).convert(to='modes').reshape()
    # # orbit_.plot()
    #
    # result = converge(orbit_, verbose=True, method='hybrid', comp_time='long')
    # result.orbit.plot()
    #
    # from orbithunter.gluing import tile_dictionary_ks
    # t0 = time.time()
    # kwargs = {'verbose':True}
    # orbit_tol=10**-15
    # orbit_n_iter = 0
    # orbit_maxiter = 500
    # ftol=1e-10
    # td = tile_dictionary_ks(padded=False)
    # symbol_array = np.array([[0, 1], [2, 2], [1, 0]])
    # orbit_ = tile(symbol_array, td, OrbitKS, stripwise=False).rescale(3).reshape(32, 32).convert(to='modes')
    # test0 = orbit_.convert(to='modes')
    # # test0.to_h5(directory='../../data/local/convergence_test/disc_testing/')
    # # test = converge(orbit_, method='hybrid', hybrid_tol=(1e-7, 1e-7), prec_param_power=(1, 1))
    # t0 = time.time()
    # kwargs = {'verbose':True}
    # orbit_tol=10**-15
    # orbit_n_iter = 0
    # orbit_maxiter = 500
    # ftol=1e-10
    # mapping = orbit_.spatiotemporal_mapping()
    # residual = mapping.residual(apply_mapping=False)
    # step_size = 0.01
    #
    # while residual > orbit_tol:
    #     orbit_n_iter += 1
    #     # Solve A dx = b <--> J dx = - f, for dx.
    #     A, b = orbit_.jacobian(), -1*mapping.state.ravel()
    #     inv_A = pinv(A)
    #     dx = orbit_.from_numpy_array(inv_A.dot(b))
    #     next_orbit = orbit_.increment(dx, step_size=step_size)
    #     next_mapping = next_orbit.spatiotemporal_mapping()
    #     next_residual = next_orbit.residual()
    #     while next_residual > residual and step_size > 10**-6:
    #         # Continues until either step is too small or residual decreases
    #         step_size /= 2.0
    #         next_orbit = orbit_.increment(dx, step_size=step_size)
    #         next_mapping = next_orbit.spatiotemporal_mapping()
    #         next_residual = next_mapping.residual(apply_mapping=False)
    #     else:
    #         n_inner = 0
    #         while next_residual < residual:
    #             n_inner+=1
    #             orbit_ = next_orbit
    #             mapping = next_mapping
    #             b = -1 * mapping.state.ravel()
    #             dx = orbit_.from_numpy_array(inv_A.dot(b))
    #             next_orbit = next_orbit.increment(dx, step_size=step_size)
    #             next_mapping = next_orbit.spatiotemporal_mapping()
    #             next_residual = next_mapping.residual(apply_mapping=False)

    return None

if __name__=='__main__':
    sys.exit(main())