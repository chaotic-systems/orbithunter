from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import glob


def main(*args,**kwargs):


    '''
    Things to include
    If number of continuation steps is too large, warn and quit
    if the predicted time to finish is too large, warn and quit
    continuation in N,M,T,L.
    Choose deltaN,deltaM,deltaT,deltaL
    persistent flag; ability to reduce deltaX if there is failure.
    
    parse filename or directory name info. ''for x in 
    '''


    search_directory = '../data/test_data/*.h5'
    for orbit_h5 in glob.glob(search_directory, recursive=True):
        orbit_ = read_h5(orbit_h5, directory='')
        # orbit_.constraints = {'T': True, 'L': False}
        # converge_result = dimension_continuation(orbit_, (orbit_.T+0.001, orbit_.L - 0.0001), precision='low', verbose=True)
        # print(converge_result.orbit.orbit_parameters,  (orbit_.T+0.001, orbit_.L - 0.0001), converge_result.orbit.residual())
        disc_converge_result = discretization_continuation(orbit_, (orbit_.N - 3, orbit_.M + 2), step_sizes=(4, -4), verbose=True)
        print(disc_converge_result.orbit.field_shape, disc_converge_result.orbit.residual())


        # new_orbit_ = rediscretize(orbit_, new_shape=(orbit_.N, orbit_.M + 2))
        # converge_result = converge(new_orbit_,  method='gradient_descent', orbit_tol=10**-15, verbose=True)
        # print(converge_result.orbit_.residual(), converge_result.exit_code)
        break


    return None


if __name__=='__main__':
    sys.exit(main())


