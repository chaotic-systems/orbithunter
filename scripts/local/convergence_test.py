import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np

def main(*args, **kwargs):
    search_directory = '../data/test_data/*.h5'
    for orbit_h5 in glob.glob(search_directory, recursive=True):
        orbit_ = read_h5(orbit_h5, directory='')
        # orbit_.plot(fundamental_domain=False)
        OrbitKS(orbit_parameters=(512, 512, 0)).plot(fundamental_domain=False)

        new_orbit_ = rediscretize(orbit_, new_shape=(orbit_.N, orbit_.M + 2))
        t0 = time.time()
        # converge_result = converge(new_orbit_,  method='gradient_descent', orbit_tol=10**-15,
        #                            verbose=True, orbit_maxiter=1000)
        t1 = time.time()
        # print(converge_result.orbit.residual(), converge_result.exit_code, t1-t0, ((t1-t0)/converge_result.n_iter),
        #       ((t1-t0)/converge_result.n_iter)/np.product(new_orbit_.field_shape))

    return None

if __name__=="__main__":
    sys.exit(main())
