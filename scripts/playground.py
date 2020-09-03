from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, method='adj', **kwargs):
    # orbit = read_h5("C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\none_000_021.h5", data_format='kstori', state_type='field')
    # orbit = rediscretize(orbit, parameter_based=True, resolution='fine')
    N = 256
    M = 256
    T = 80
    L = 22
    n, m = int(N // 2) - 1, int(M // 2) - 1
    sms = int(L / (2*pi*np.sqrt(2)))
    tms = int(T/10)

    sigma_time = 10
    sigma_space = 2

    np.random.seed(0)
    tester = OrbitKS(state=4*np.random.randn(N,M), state_type='field', T=T, L=L).convert(to='modes')

    space = np.abs((tester.L / (2 * pi)) * tester.elementwise_dxn(tester.parameters))
    time = np.abs((tester.T / (2 * pi)) * tester.elementwise_dtn(tester.parameters))

    mollifier = np.exp(-(space-sms)**2/(2*sigma_space**2)-(time-tms)**2/(2*sigma_time**2))

    modes = 3*np.multiply(mollifier, tester.state)
    test_orbit = OrbitKS(state=modes, state_type='modes', T=tester.T, L=tester.L)

    test_orbit.plot()
    return None


if __name__=='__main__':
    main()
