from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time

def main(*args, method='adj', **kwargs):
    orbit = read_h5('OrbitKS_L42p007_T33p116.h5', directory='local', state_type='modes')
    # orbit2 = change_orbit_type(orbit, OrbitKS)
    # rpo_orbit = read_h5('RelativeOrbitKS_L22p007_T53p924.h5', directory='local', state_type='modes')
    # anti_orbit = read_h5('anti_L16p87_T24p39.h5', directory='data', state_type='modes')
    # sr_orbit = read_h5('ppo_L34p01_T64p70.h5', directory='data', state_type='modes')
    # eqva_orbit = read_h5('eqva_L11p436.h5', directory='data', state_type='modes')
    # reqva_orbit = read_h5('reqva_L27p39.h5', directory='data', state_type='modes')
    orbit_list = [orbit]#, rpo_orbit] #, sr_orbit, anti_orbit, eqva_orbit, reqva_orbit]
    for o in orbit_list:
        o.L += 0.01
        max_iter = 32768
        t0 = time.time()
        result = converge(o, method='adj', atol=10**-6, verbose=True, max_iter=max_iter)
        t1 = time.time()
        print('Total descent time for', str(o), t1-t0, 'per step', (t1-t0) / (max_iter))
    return None


if __name__=='__main__':
    main()
