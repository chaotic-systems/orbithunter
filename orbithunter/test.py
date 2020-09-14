from __future__ import print_function, division
from .orbit import *
import numpy as np
import pytest
import unittest


class OrbitTest(unittest.TestCase):

    # unittest.TestCase.assertEqual(x, y) (checking if x == y)
    # assertRaises
    # setUp method
    # coverage report
    # fixtures
    # from mock import Mock
    # mock objects
    # test doubles
    # dependency injection
    # addCleanup
    # doctest
    # nose, py.test better test runners
    # ddt data-driven tests
    # coverage
    # selenium in browser testing
    # jenkins, travis
    # TDD tests before code
    # BDD describe external behavior
    # integration tests 
    def FFT_test(self):
        pass

    def convergence_test(self):
        pass

    def orbit_init_test(self):
        orbit_ = OrbitKS(T=10, L=10, constraints={'T', False})
        relative_orbit_ = RelativeOrbitKS(T=5., L=20000., S=1,  frame='physical', seed=1)
        shift_reflection_orbit = ShiftReflectionOrbitKS(T=10, L=10, constraints={'T', False})
        antisymmetric_orbit = AntisymmetricOrbitKS(T=10, L=10, constraints={'T', False})
        relative_equilibrium_orbit = RelativeEquilibriumOrbitKS(T=10, L=10, constraints={'T', False})
        equilibrium_orbit = EquilibriumOrbitKS(T=10, L=10, constraints={'T', False})

def FFT_testing_suite(tori_tuple):
    uu,N,M,T,L,S = tori_tuple
    u0 = np.reshape(uu,[N*M,1])

    stppo1 = ppo.fft_(u0,N,M)
    stppo2 = np.dot(ppodm.FFT_matrix(N,M),u0)
    strpo1 = rpo.fft_(u0,N,M)
    strpo2 = np.dot(rpodm.FFT_matrix(N,M),u0)
    stanti1 = anti.fft_(u0,N,M)
    stanti2 = np.dot(antidm.FFT_matrix(N,M),u0)

    print('Verifying the integrity of forward FFTs and DFT matrices')
    size0 = int(np.size(stppo1)-np.size(stppo2))
    size1 = int(np.size(strpo1)-np.size(strpo2))
    size2 = int(np.size(stanti1)-np.size(stanti2))
    if not size0 and not size1 and not size2:
        print('All transforms successfully return the same number of variables')
    else:
        print('There is an error in the dimension of the output of the FFTs of one of the isotropy subgroups')
        if size0 > 0:
            print('ERROR: PPO Transforms')
        if size0 > 0:
            print('ERROR: RPO Transforms')
        if size0 > 0:
            print('ERROR: Antisymmetric Transforms')
        return False

    testppo = np.linalg.norm(stppo1-stppo2)
    testrpo = np.linalg.norm(strpo1-strpo2)
    testanti = np.linalg.norm(stanti2-stanti2)

    if testrpo < 10**-14 or testppo < 10**-14 or testanti < 10**-14:
        print('All Forward transforms were successful')
    else:
        print('There is an error some of the forward FFTs')
        if testppo > 10**-14:
            print('ERROR: PPO Transforms')
        if testppo > 10**-14:
            print('ERROR: RPO Transforms')
        if testanti > 10**-14:
            print('ERROR: Antisymmetric Transforms')
        return False


    u1 = ppo.ifft_(stppo1,N,M)
    u2 = ppo.ifft_(stppo2,N,M)
    u3 = np.dot(ppodm.IFFT_matrix(N,M),stppo1)
    u4 = np.dot(ppodm.IFFT_matrix(N,M),stppo2)

    u5 = rpo.ifft_(strpo1,N,M)
    u6 = rpo.ifft_(strpo2,N,M)
    u7 = np.dot(rpodm.IFFT_matrix(N,M),strpo1)
    u8 = np.dot(rpodm.IFFT_matrix(N,M),strpo2)

    u9 = anti.ifft_(stanti1,N,M)
    u10 = anti.ifft_(stanti2,N,M)
    u11 = np.dot(antidm.IFFT_matrix(N,M),stanti1)
    u12 = np.dot(antidm.IFFT_matrix(N,M),stanti2)

    normvec = np.zeros([12,1])
    normvec[0] = np.linalg.norm(u1)
    normvec[1] = np.linalg.norm(u2)
    normvec[2] = np.linalg.norm(u3)
    normvec[3] = np.linalg.norm(u4)
    normvec[4] = np.linalg.norm(u5)
    normvec[5] = np.linalg.norm(u6)
    normvec[6] = np.linalg.norm(u7)
    normvec[7] = np.linalg.norm(u8)
    normvec[8] = np.linalg.norm(u9)
    normvec[9] = np.linalg.norm(u10)
    normvec[10] = np.linalg.norm(u11)
    normvec[11] = np.linalg.norm(u12)

    normppo = normvec[:4]
    normrpo = normvec[4:8]
    normanti = normvec[8:]
    for n in range(0,4):
        print(n)
        testppo = np.linalg.norm(normppo-np.roll(normppo,n))
        testrpo = np.linalg.norm(normrpo-np.roll(normrpo,n))
        testanti = np.linalg.norm(normanti-np.roll(normanti,n))
        if testppo > 10**-13:
            print('PPO inverse FFTs failed check definitions')
            return False
        elif testrpo > 10**-13:
            print('RPO inverse FFTs failed check definitions')
            return False
        elif testanti > 10**-13:
            print('ANTI inverse FFTs failed check definitions')
            return False

    print('All tests have passed')
    return True


def test_integration_tolerance(Orbit_converged,Orbit_init,**kwargs):
    uC,NC,MC,TC,LC,SC = Orbit_converged
    uI,NI,MI,TI,LI,SI = Orbit_init
    symmetry=kwargs.get('symmetry','rpo')
    residual = ks.compute_residual_fromtuple(Orbit_converged,symmetry=symmetry)
    NMresidual = NC*MC*residual
    Orbit_integrated = ETDRK4.ETDRK4_timeseries(Orbit_converged,symmetry=symmetry)
    Orbit_comparison = disc.rediscretize(Orbit_integrated,new_N=NC,new_M=MC)
    u_integrated = Orbit_comparison[0]

    converged_integrated_norm = np.linalg.norm(uC.flatten()-u_integrated.flatten())
    init_integrated_norm = np.linalg.norm(uI.flatten()-u_integrated.flatten())
    init_converged_norm = np.linalg.norm(uI.flatten()-uC.flatten())

    print('The L_2 difference between init and integrated is', init_integrated_norm)
    print('The L_2 difference between init and converged is', init_converged_norm)
    print('The L_2 difference between converged and integrated is', converged_integrated_norm,'when the residual is,',residual,NMresidual)

    ksplot.plot_spatiotemporal_field(Orbit_init,symmetry='none',display_flag=True,filename='init4.png')
    ksplot.plot_spatiotemporal_field(Orbit_converged,symmetry='none',display_flag=True,filename='converged4.png')
    ksplot.plot_spatiotemporal_field(Orbit_integrated,symmetry='none',display_flag=True,filename='integrated4.png')

    return None






