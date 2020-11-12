from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import glob
from scipy.linalg import pinv
def main(*args, **kwargs):
    #
    # streak = td[0]
    # merger = td[1]
    # wiggle = td[2]
    # x = discretization_continuation(merger.reshape(16, 16).transform(to='modes'), (24, 24),
    #                                 step_sizes=(2,2), cycle=True, verbose=True, method='lstsq')

        # figs = '../../data/local/thesis/figures/'
    # defect_fam = '../../data/local/thesis/families/defect/'
    # tmp = read_h5('ppo_L21p74_T95p23_ppo_L21p94_T94p98_time.h5',
    #               directory='../../data/local/thesis/gluing', data_format='kstori')
    # test = discretization_continuation(tmp, (96, 32), verbose=True, method='hybrid')
    # # read_h5('RelativeOrbitKS_L12p846_T40p688.h5', directory=defect_fam).plot(filename='left_defect.pdf', directory=figs,
    # #                                                                fundamental_domain=True)
    # # read_h5('RelativeOrbitKS_T17p659_L13p016.h5', directory=defect_fam).plot(filename='middle_defect.pdf', directory=figs,
    # #                                                                fundamental_domain=True)
    # read_h5('RelativeOrbitKS_T8p663_L13p100.h5', directory=defect_fam).plot(filename='right_defect.pdf', directory=figs,
    #                                                                fundamental_domain=True)
    # read_h5('RelativeOrbitKS_T8p458_L13p103.h5', directory='../../data/local/continuation/merge_tile/').plot(filename='reqv_defect.pdf', directory=figs,
    #                                                                fundamental_domain=True)
    # test = AntisymmetricOrbitKS(seed=5, parameters=(100,22,0)).reshape(64, 64).rescale(5)
    # result = kse_integrate(test, verbose=True, integration_time=2000, step_size=0.2)
    # result.plot(fundamental_domain=False)
    # mode_spectrum = 'time_truncated'
    # s = 0
    # mag = 5
    # a = AntisymmetricOrbitKS(seed=s, T_min=40, T_max=80, L_min=22,
    #                          L_max=22, tscale=0, nonzero_parameters=True, spectrum=mode_spectrum).rescale(mag)
    # a = a.reshape(32, 16)
    # a.plot(fundamental_domain=False)
    # plt.matshow(np.log10(np.abs(a.state)))
    # plt.colorbar()
    # plt.show()
    # datadir = '../../data/examples/gluing/'
    # o0=read_h5('OrbitKS_T41p382_L38p050.h5', directory=datadir).reshape(512,512)
    # o1=read_h5('OrbitKS_T46p754_L31p245.h5', directory=datadir).reshape(512,512)
    # o2=read_h5('OrbitKS_T43p819_L34p784.h5', directory=datadir).reshape(512,512)
    # o3=read_h5('OrbitKS_T36p167_L25p720.h5', directory=datadir).reshape(512,512)
    # random_orbit_td = {0:o0,1:o1,2:o2,3:o3}
    # np.random.seed(0)
    # symbol_array = (4*np.random.rand(10, 10)).astype(int)
    # random_orbit_tiling = tile(symbol_array, random_orbit_td, OrbitKS,stripwise=False).reshape()
    # random_orbit_tiling_result = converge(random_orbit_tiling, method='gmres',
    #                                   scipy_kwargs={'tol':1e-1, 'maxiter':1, 'restart':1},
    #                                   verbose=True)
    # plt.matshow(np.log10(np.abs(example_orbit.transform(to='modes').state)))
    # plt.colorbar()
    # plt.show()
    #
    # tile_dictionary = tile_dictionary_ks(padding=False, comoving=False)
    # tile_dictionary[0].plot()
    # tile_dictionary[1].plot()
    # tile_dictionary[2].plot()
    # symbols = np.array([[0,1]])
    # sym_str = to_symbol_string(symbols)
    # orbit_ = tile(symbols, tile_dictionary, OrbitKS, stripwise=True, gluing_order=(1,0))
    # orbit_.plot(fundamental_domain=False)
    # orbit_.plot()
    #
    # sym_str = to_symbol_string(symbols)
    # orbit_ = orbit_.reshape(32, 32)#.rescale(3.2)
    # if kwargs.get('verbose'):
    #     print('Beginning search for {}'.format(symbols))
    # fname_init = ''.join([str(OrbitKS.__name__), '_', sym_str, '_initial.h5'])
    # orbit_.to_h5(filename=fname_init,
    #              directory='../../data/local/testing/symbolic/',
    #              verbose=True, include_residual=True)
    # orbit_.plot(filename=fname_init,
    #             show=True, save=True, verbose=True,
    #             directory='../../data/local/testing/symbolic/')
    # orbit_ = converge(orbit_,  verbose=True, preconditioning=False,
    #                      method='adj', tol=1e-6, comp_time='thorough').orbit
    #
    # fname_adjoint = ''.join([str(OrbitKS.__name__), '_', sym_str, '_adjoint.h5'])
    # orbit_.to_h5(filename=fname_adjoint,
    #                    directory='../../data/local/testing/symbolic/',
    #                    verbose=True, include_residual=True)
    # orbit_.plot(filename=fname_adjoint,
    #                   show=True, save=True, verbose=True,
    #                   directory='../../data/local/testing/symbolic/')
    # hunt_result = converge(orbit_, method='lstsq', verbose=True,
    #                        comp_time='thorough', tol=1e-6)
    # # write_symbolic_log(symbols, hunt_result, 'symbolic_log_padded_time.csv',
    # #                    padding=padding, comoving=comoving)
    #
    # if hunt_result.status == -1:
    #     fname = ''.join([str(OrbitKS.__name__), '_', sym_str, '_converged.h5'])
    #     hunt_result.orbit.to_h5(filename=fname,
    #                             directory='../../data/local/testing/symbolic/',
    #                             verbose=True, include_residual=True)
    #     hunt_result.orbit.plot(filename=fname,
    #                            show=True, save=True, verbose=True,
    #                            directory='../../data/local/testing/symbolic/')
    # else:
    #     fname = ''.join([str(OrbitKS.__name__), '_', sym_str, '_unconverged.h5'])
    #     hunt_result.orbit.to_h5(filename=fname,
    #                             directory='../../data/local/testing/symbolic/',
    #                             verbose=True, include_residual=True)
    #     hunt_result.orbit.plot(filename=fname,
    #                            show=True, save=True, verbose=True,
    #                            directory='../../data/local/testing/symbolic/')
    return None

if __name__=='__main__':
    sys.exit(main())