from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import glob

def main(*args, **kwargs):
    test=converge(OrbitKS(seed=5, parameters=(100, 22, 0.), N=32, M=32, tscale=1, xscale=3).rescale(3.5),method='lstsq')

    # test = np.ones([32,32])
    # testtest = test.reshape(-1, 32)
    # example_orbit = OrbitKS(seed=5, parameters=(100, 100,0.)).rescale(3.5)
    # test = example_orbit.reshape((16, 16))
    # m = read_h5('C:/Users/Matt/Desktop/gudorf/KS/python/data_and_figures/tiles/defects/defect1/final_tile/RelativeOrbitKS_T15p855_L13p026.h5', directory='', basis='field')
    # cont_m = dimension_continuation(rediscretize(m, new_shape=(32, 32)), m.L-1, axis=1, method='lstsq', step_size=0.01, verbose=True).orbit
    # test = OrbitKS(seed=5, parameters=(32., 100,0.), N=32, M=256).rescale(4)
    # orbit_ = read_h5('OrbitKS_T54p129_L26p744.h5', directory='../../data/examples/').rotate(2, axis=0).rotate(0.1, axis=1)
    # orbit_ = read_h5('C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\full_L26.7_T54.h5',directory='', data_format='kstori', basis='field')
    # test = RelativeOrbitKS(parameters=(100,100, 0), zero_shift=True).change_reference_frame(to='physical')
    # orbit_ = rediscretize(orbit_.convert(to='field'), new_shape=(512, 512))
    # # orbit_.plot()
    # clipped_orbit = clip(orbit_, ((0., 15), (None, 1.9091883092036783)), clipping_class=OrbitKS)
    # test_shift = calculate_spatial_shift(clipped_orbit.convert(to='s_modes').state,
    #                                      clipped_orbit.L, n_modes=clipped_orbit.m)
    # clipped_orbit.plot()
    # clipped_orbit = rediscretize(clipped_orbit, new_shape=(16,20))
    # clipping_step_1_result = converge(clipped_orbit, precision='machine', method='lstsq', ftol=0, verbose=True)
    # clipping_step_2_orbit = clip(clipping_step_1_result.orbit, ((None, None), (0,2)))
    # clipping_step_2_result = converge(clipping_step_2_orbit)
    # clipping_orbit_2 = clipping_step_2_result.orbit
    # antisymmetric_clipping = convert_class(clipping_orbit_2, AntisymmetricOrbitKS)
    # antisymmetric_clipping.plot()
    # orbit_.plot()
    # orbit_.rotate(0.1, axis=1).rotate(2.5, axis=0).plot()
    # orbit_ = OrbitKS(seed=0, N=64, M=32, tscale=2, tvar=2, xvar=1, xscale=5, parameters=(200, 44, 0), spectrum='gaussian').rescale(5)
    # wtf_big = rediscretize(wtf, new_shape=(1, wtf.M+2))
    # res = converge(orbit_, method='lstsq', verbose=True)
    # res.orbit.plot()
    # orbit_ = read_h5('C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\ppo_L21p79_T96p17.h5',
    #                  directory='', data_format='kstori', basis='field')
    # orbit_ = convert_class(read_h5('C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\ppo_L21p79_T96p17.h5',
    #                  directory='', data_format='kstori', basis='field'), OrbitKS)
    # print(orbit_.residual())
    # orbit_ = converge(orbit_, method='lstsq', precision='machine', verbose=True).orbit
    #
    # orbit_ = orbit_.convert(to='field')
    # np.random.seed(0)
    # noise = np.random.randn(*orbit_.state.shape)
    #
    # orbit_.state = orbit_.state + np.multiply(noise, orbit_.state)
    # print(orbit_.residual())
    # res = converge(1.5*orbit_, method='lstsq', precision='machine', verbose=True)
    # print(orbit_.residual())
    test = 0
    # orbit_ = OrbitKS(seed=0, N=64, M=32, tscale=5, tvar=10, xvar=5,
    #                  xscale=4, parameters=(100, 44, 0), spectrum='linear').rescale(3.25)
    # orbit_.plot()
    # conv_orbits = []
    # conv_seeds = []
    # test = OrbitKS()
    # testdx2 = test.elementwise_dxn(test.dx_parameters, power=2)
    # for s in range(100):
    #     orbit_ = OrbitKS(seed=s, N=32, M=32).rescale(3.25)
    #     print('#',end='')
    #     conv_result = converge(orbit_, preconditioning=False)
    #     print('#',end='')
    #     conv_result.orbit.plot(save=False, show=True)
    # np.random.seed(0)
    # o = OrbitKS(parameters=(512, 512, 0))
    # o.plot()
    # tester = converge(o, method='l-bfgs-b')
    # tester.orbit.plot()
    # hey=0
    #
    # N = 256
    # M = 256
    # T = 512
    # L = 512
    # n, m = int(N // 2) - 1, int(M // 2) - 1
    # sms = int(L / (2*pi*np.sqrt(2)))
    # tms = int(np.sqrt(T/10))
    #
    # sigma_time = 20
    # sigma_space = 15
    #
    # np.random.seed(0)
    # tester = OrbitKS(state=4*np.random.randn(N, M), basis='field', parameters=(512, 512, 0)).convert(to='modes')
    # tester.state = np.random.randn(*tester.state.shape)
    # space = np.abs((tester.L / (2 * pi)) * tester.elementwise_dxn(tester.dx_parameters))
    # time = np.abs((tester.T / (2 * pi)) * tester.elementwise_dtn(tester.dt_parameters))
    #
    # mollifier = np.exp(-(space-sms)**2/(2*sigma_space**2)-(time-tms)**2/(2*sigma_time**2))
    #
    # modes = np.multiply(mollifier, tester.state)
    # test_orbit = OrbitKS(state=modes, state_type='modes', parameters=(512,512,0))
    #
    #
    # good_large_guess = OrbitKS(state=np.sign(test_orbit.convert(to='field').state) * np.abs(test_orbit.convert(to='field').state)**(1./3.),
    #                            basis='field', parameters=(512, 512, 0)).rescale(2.5, method='absolute')
    # good_large_guess.plot(padding=True)
    # print(good_large_guess.residual())
    # tester = converge(good_large_guess, method='l-bfgs-b', preconditioning=False)
    # tester.orbit.plot(padding=True)
    # print(tester.orbit.residual())
    # test=0
    # orbit_ = read_h5('RelativeOrbitKS_T90p673_L35p230.h5', directory='../../data/local/RelativeOrbitKS/')
    # orbit_.convert(to='field').plot(padding=False, fundamental_domain=False)
    # orbit_.convert(to='field').plot(padding=False, fundamental_domain=False)
    # print(orbit_.change_reference_frame(to='physical').frame)
    # phys_ref_frame = orbit_.change_reference_frame(to='physical')
    # print(phys_ref_frame.convert(to='field', inplace=True).frame)
    # phys_ref_frame.plot(padding=False)
    # orbit_.change_reference_frame(to='physical').convert(to='field').plot(padding=True, fundamental_domain=True)
    # orbit_.plot(padding=False)
    # orbit_ = read_h5('OrbitKS_T602p288_L1758p826.h5', directory='../../data/local/recent/')
    # testtest = orbit_.verify_integrity()
    # what = 0
    # test_param_grad = OrbitKS(seed=1, parameters=(66, 22, 0), N=32, M=32).rescale(2)
    # # test_param_grad.plot()
    # test_result = converge(test_param_grad, preconditioning=True, verbose=True)
    # # test_result.orbit.plot(save=True, filename='new_adjoint_prec.h5', directory='../../figs/local/recent/')
    # test_result.orbit.plot(save=True, filename='old_adjoint_prec.h5', directory='../../figs/local/recent/')
    #
    # # test_result = converge(test_result.orbit.rescale(magnitude=0.33, method='power'), method='hybrid')
    # # plt.matshow(np.log10(np.abs(test_result.orbit.state)))
    # # plt.colorbar()
    # # plt.show()
    # print(test_result.orbit.residual(), test_result.orbit.parameters, test_result.n_iter)
    # test_result.orbit.plot(figsize=(5, 5))
    # tmp = OrbitKS(seed=0, parameters=(512, 512), M=1024)
    # tmp.plot()
    # int1 = kse_integrate(tmp, verbose=True, integration_time=1000, step_size=0.25)
    # orbit_ = read_h5('ppo_L21p90_T96p68.h5', directory="C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\", data_format='kstori')
    # result = discretization_continuation(orbit_, new_shape=(32, 32), verbose=True, plot_intermediates=True, ftol=1e-8, precision='very_high', method='lstsq')
    # from orbithunter.gluing import tile_dictionary_ks
    # td = tile_dictionary_ks(padded=True)
    # tile_dictionary = rediscretize_tiling_dictionary(tile_dictionary_ks(padded=True), new_shape=(16, 16))
    # glue_shape = (3, 3)
    # symbol_arrays = generate_symbol_arrays(tile_dictionary, glue_shape=glue_shape, unique=True)
    #
    # for sa in symbol_arrays:
    #     tiled_initial_condition = tile(sa, tile_dictionary, OrbitKS)
    #     orbit_complex()
    #     result = converge(tiled_initial_condition, method='hybrid', precision='very_high', verbose=True)
    #
    # test = list(glob.glob('C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\trawl\\rpo\\data\\**\\*.h5', recursive=True))
    # for orbit_h5 in test:
    #     try:
    #         orbit_ = read_h5(orbit_h5, data_format='kstori', check=True)
    #     except KeyError:
    #         try:
    #             orbit_ = read_h5(orbit_h5, data_format='orbithunter_old', check=True)
    #         except KeyError:
    #             orbit_ = read_h5(orbit_h5,  check=True)
    #     print(orbit_, orbit_.field_shape)
    return None

if __name__=='__main__':
    sys.exit(main())