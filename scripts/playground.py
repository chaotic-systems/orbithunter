from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, method='adj', **kwargs):
    test = read_h5('OrbitKS_L37p297_T79p778.h5', directory='local')
    test2 = rediscretize(test, newN=64, newM=32)

    cdict = {1:'success', 0:'failure', 2:'success'}
    orbit_list = []
    res, tim = [], []
    max_iter = 3

    t0 = time.time()
    result = converge(test2, method='lsqr',verbose=True)
    t1 = time.time()
    print(' exit code ', result.exit_code,', residual ', result.orbit.residual(),', time ', t1-t0)
    orbit_list.append(result.orbit)
    res.append(result.orbit.residual())
    tim.append(t1-t0)
    result.orbit.plot()

    res = np.array(res).reshape(-1,1)
    tim = np.array(tim).reshape(-1,1)

    return None


if __name__=='__main__':
    main()
