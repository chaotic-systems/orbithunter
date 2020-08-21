from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time

def main(*args, method='adj', **kwargs):
    sr_orbit = read_h5("C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\none_000_021.h5")
    orbit = OrbitKS(state=sr_orbit.state, state_type=sr_orbit.state_type, T=sr_orbit.T,
                    L=sr_orbit.L+0.01).convert(to='modes')
    self = orbit.copy()
    other = orbit.spatiotemporal_mapping()
    t0=time.time()
    result = converge(orbit, method='adj', atol=10**-6, verbose=True)
    t1=time.time()
    print('adj desc time', t1-t0, 'per step', (t1-t0) / (16384.))
    # test = ShiftReflectionOrbitKS()
    # defect = read_h5("C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\rpo_L13p02_T15.h5")
    # test = defect.residual()
    # x = defect.convert(to='modes')
    # y = x.rotate(distance=5)
    # test2 = y.residual()
    # result = converge(defect, method='lstsq', fixedparams=(False, False, False), verbose=True)
    # t, code = result.orbit, result.exit_code
    return None


if __name__=='__main__':
    main()
