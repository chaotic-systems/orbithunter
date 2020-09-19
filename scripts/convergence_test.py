import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *

def main(*args, **kwargs):
    search_directory = '../data/test_data/*.h5'
    for orbit_h5 in glob.glob(search_directory, recursive=True):
        orbit_ = read_h5(orbit_h5, directory='')
        new_orbit_ = rediscretize(orbit_, new_shape=(orbit_.N, orbit_.M + 2))
        converge_result = converge(new_orbit_,  orbit_tol=10**-15, verbose=True)
        print(converge_result.orbit.residual(), converge_result.exit_code)
    return None

if __name__=="__main__":
    sys.exit(main())
