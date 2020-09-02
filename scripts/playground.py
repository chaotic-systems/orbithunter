from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time

def main(*args, method='adj', **kwargs):
    orbit = read_h5("C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\none_000_021.h5", data_format='kstori', state_type='field')
    orbit = rediscretize(orbit, parameter_based=True, resolution='fine')
    # orbit.plot()

    orbit = OrbitKS(state=orbit.state[-1,:].reshape(1,-1), state_type=orbit.state_type, T=orbit.T, L=orbit.L)

    # -1*(orbit.elementwise_dxn(orbit.parameters, power=2)+orbit.elementwise_dxn(orbit.parameters, power=4))

    tmp = orbit.convert(to='modes').convert(to='field').state
    return None


if __name__=='__main__':
    main()
