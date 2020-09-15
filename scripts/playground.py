from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    # test = RelativeOrbitKS(T=44, L=44, state_type='s_modes', randomkwarg='im nothing')
    tmp = read_h5('RelativeOrbitKS_L21p956_T69p994.h5', directory='local')
    tmp2 = rediscretize(tmp, new_shape=(64, 64))

    # directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')

    # m = read_h5('merger_high_res.h5')
    # # w = read_h5('wiggle_high_res.h5')
    # # s = read_h5('streak_high_res.h5')

    # test = RelativeOrbitKS(T=44, L=44, constraints={'T': True, 'L': True, 'S': True}, seed=0)
    for orbit_name in ['RelativeEquilibriumOrbitKS_L28p810_T21p142.h5', 'EquilibriumOrbitKS_L11p39.h5',
                       'ShiftReflectionOrbitKS_L28p847_T97p242.h5', 'RelativeOrbitKS_L27p526_T68p730.h5',
                       'AntisymmetricOrbitKS_L34p913_T51p540.h5', 'OrbitKS_L31p245_T46p754.h5']:
        self = read_h5(orbit_name, directory='../data/test_data/')
        self = rediscretize(self, new_shape=(self.N, self.M+2))
        t0 = time.time()
        # print(orbit_name, self.residual())
        # self.plot(fundamental_domain=False)
        print(orbit_name, self.residual())
        r = converge(self,  method='lstsq', orbit_tol=1e-12, verbose=True)
        print(orbit_name, r.orbit.residual())
    return None


if __name__=='__main__':
    sys.exit(main())