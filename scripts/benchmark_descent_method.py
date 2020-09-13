from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np

def main(*args, method='adj', **kwargs):
    # orbit = read_h5('OrbitKS_L42p007_T33p116.h5', directory='local', state_type='modes')
    # rpo_orbit = read_h5('RelativeOrbitKS_L22p007_T53p924.h5', format='orbithunter_old', directory='local', state_type='modes')
    # anti_orbit = read_h5('AntisymmetricOrbitKS_L16p87_T24p39.h5', format='kstori',directory='local', state_type='modes')
    # sr_orbit = read_h5('ShiftReflectionOrbitKS_L34p018_T129p401.h5', format='orbithunter_old', directory='local', state_type='modes')
    # eqva_orbit = read_h5('EquilibriumOrbitKS_L11p436.h5', format='kstori', directory='local', state_type='modes')
    # reqva_orbit = read_h5('reqva_L27p39.h5', format='kstori',  directory='local', state_type='modes')
    # # orbit_list = [orbit, rpo_orbit, sr_orbit, anti_orbit, eqva_orbit, reqva_orbit]
    # orbit_list = [reqva_orbit]
    # for o in orbit_list:
    #     o.L += 0.01
    #     result = converge(o, method='lstsq', atol=10**-14, verbose=True)
    #     result.orbit.plot(save=True, show=False, directory='local')
    OrbitKS(state=np.ones([512, 512]), state_type='field', L=512, T=512).plot(padding=False, save=True)
    return None


if __name__=='__main__':
    main()
