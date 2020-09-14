from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    # test = RelativeOrbitKS(T=44, L=44, state_type='s_modes', randomkwarg='im nothing')
    # tmp = read_h5('RelativeOrbitKS_L21p956_T69p994.h5', directory='local')
    # directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')

    # m = read_h5('merger_high_res.h5')
    # # w = read_h5('wiggle_high_res.h5')
    # # s = read_h5('streak_high_res.h5')

    test = RelativeOrbitKS(T=44, L=44, constraints={'T': True, 'L': True, 'S': True}, seed=0)
    for orbit_name in ['RelativeEquilibriumOrbitKS_L28p810_T21p142.h5', 'EquilibriumOrbitKS_L11p39.h5',
                       'ShiftReflectionOrbitKS_L28p847_T97p242.h5', 'RelativeOrbitKS_L27p526_T68p730.h5',
                       'AntisymmetricOrbitKS_L34p913_T51p540.h5', 'OrbitKS_L31p245_T46p754.h5']:
        self = read_h5(orbit_name, directory='../data/test_data/')
        self = rediscretize(self, new_M=self.M-2)
        t0 = time.time()
        print(orbit_name)
        self.plot(fundamental_domain=False)
        r = converge(self,  method='lstsq', orbit_tol=1e-12, verbose=True)
        r.orbit.plot(fundamental_domain=False)
        t1 = time.time()
        print(orbit_name, t1-t0, r.orbit.residual())

    #
    # o = read_h5('RelativeOrbitKS_L13p026_T15p855.h5', directory='orbithunter')
    # for m in np.arange(o.N, 66, 2):
    #     o = rediscretize(o, new_N=m)
    #     print(o.residual(), m,m,m,m,m,m)
    #     r = converge(o, method='lstsq', orbit_tol=10**-12, verbose=True)
    #     o = r.orbit.copy()
    #
    # for m in np.arange(o.M, 50, 2):
    #     o = rediscretize(o, new_M=m)
    #     print(o.residual(), m,m,m,m,m,m)
    #     r = converge(o, method='lstsq', orbit_tol=10**-12, verbose=True)
    #     o = r.orbit.copy()
    #
    # merger = o.convert(to='field')
    # padded_merger = np.concatenate((np.zeros([merger.N, 8]), merger.state, np.zeros([merger.N, 8])), axis=1)
    # pmerger = merger.__class__(state=padded_merger, state_type='field', parameters=merger.parameters)
    # pmerger.to_h5(directory='../data/tiles')
    #
    # padded_merger = np.concatenate((np.zeros([merger.N, 8]),
    #                                 merger.change_reference_frame(to='physical').state,
    #                                 np.zeros([merger.N, 8])), axis=1)
    # pmerger = merger.__class__(state=padded_merger, state_type='field', parameters=merger.parameters)
    # pmerger.to_h5(filename='RelativeOrbitKS_L13p026_T15p856_fdomain.h5', directory='../data/tiles')
    #
    # o = read_h5('EquilibriumOrbitKS_L6p39.h5', directory='orbithunter')
    # for m in np.arange(o.M-2, 22, -2):
    #
    #     o = rediscretize(o, new_M=m)
    #
    #     print(o.residual())
    #     r = converge(o, method='lstsq', orbit_tol=10**-12, verbose=True)
    #     print(r.orbit.residual())
    #     o = r.orbit.copy()
    #
    # streak = o.convert(to='field')
    #
    # padded_streak = np.concatenate((np.zeros([streak.N, 20]), streak.state, np.zeros([streak.N, 20])), axis=1)
    # pstreak = streak.__class__(state=padded_streak, state_type='field', parameters=streak.parameters)
    # pstreak.to_h5(directory='../data/tiles/')

    return None


if __name__=='__main__':
    sys.exit(main())