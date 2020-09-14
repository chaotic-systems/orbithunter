from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time
import numpy as np
from math import pi

def main(*args, **kwargs):
    test = '0112012_1201112_1211020_1002112_1101112_2022212_2110121__1220101_0202020_2202202_0122101_1221122_0211010_0120011__1000202_2022101_0112021_1110121_1210202_2022212_1022112__0121012_0211221_2011022_1200011_2020102_1110101_0202100__1011010_1000022_0201220_1012100_0201022_1211212_1212002__2100021_2222112_2110010_0102221_2110110_1100121_0002012__2001220_2222122_1222020_1002200_2001101_2122110_1011022__2211201_1220221_2212200_0222020_0200010_1210100_2110110__0200101_0120101_1200201_2012122_1101202_1002000_1120212__0102211_2212122_1000122_2010210_2111012_2212002_1201002__2022120_2020022_2022222_2111012_1220012_1001001_2000121__2221122_0212100_2201002_0221001_2111012_0210112_1121002__0002220_2100202_1121100_2112012_2120022_2112101_0010202__1020000_1222100_2202020_0212201_2212011_0022121_0221211__1200120_1100122_2011211_0100001_1100022_1120121_0012200__1201012_0221201_2021112_0012211_1102211_2100212_1101220__0000120_2002112_1012220_1201020_2011020_1020011_1110221__2200101_1211022_1112010_2021222_2022222_0222211_0122000__2210200_1200120_0122110_2222101_1210100_2110001_1200111__0102010_0021101_0102001_0120210_0021021_0102011_1122012__1212112_1101211_1011011_2122120_0112012_0020201_2000012__1112011_0100020_0201200_0011122_0210100_2020222_0211110__2110220_0220122_2202110_0020020_0021010_0122101_1002120__1202101_2022221_0001020_0022021_0201010_0100021_1211011__2201202_1221221_0112021_2200201_0101220_1110102_1121001__0200222_2000110_1200110_2210222_0222102_2002221_1000111__0201222_1200112_2021212_2022001_1222210_0202011_2120202__2020210_0122220_1222210_2202000_1212022_2221021_1222101__1001020_0101220_1212210_2011111_0021102_1111210_2011222__0011102_0221221_2200102_0000211_1101101_2111220_1121202__0011202_1222220_2121000_2002201_1222210_1121121_0222012__1200020_1002221_0021201_2122201_0220221_1022101_2012012__1222211_1111212_1102221_1021220_0110111_0221212_1211000__1200012_1022011_1122211_0211022_2000111_0202211_0002022__1010101_2112200_0000021_2012021_1012120_2201010_2122111__0010221_0122002_0021210_2221021_1221022_0111111_1120200__1011120_0202120_2220011_1100021_0110220_1011122_0001010__1211112_0211001_0112200_2010002_2001012_2221212_2210102__1021022_1000211_0222100_2021012_2111021_0011010_1221022__0111200_0210001_2220000_2101021_0201212_0010211_0211200__1120002_0201012_1110100_0221102_2002202_2221111_0120222__0012211_1022101_0201111_2222002_1010122_1011101_2001112__1020000_0101012_2120010_0122000_2112021_0222001_2122002__2101022_2120201_1122201_0022221_1010122_2221110_0221202__1211102_0011111_0011010_2100221_2020200_0112221_0111122__1121211_0202012_1120222_2012010_2112111_2210200_0200011__1201022_2211202_1222111_1011002_1121001_2211102_2021120__0102022_1221011_2010220_0002021_1100220_1112212_1112202__1201010_0100101_0100201_1012201_2220121_1110102_1210210'
    # test = RelativeOrbitKS(T=44, L=44, state_type='s_modes', randomkwarg='im nothing')
    # tmp = read_h5('RelativeOrbitKS_L21p956_T69p994.h5', directory='local')
    # directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')

    # m = read_h5('merger_high_res.h5')
    # # w = read_h5('wiggle_high_res.h5')
    # # s = read_h5('streak_high_res.h5')
    #
    # m = converge(rediscretize(m, new_M=16,new_N=16), method='lgmres', verbose=True, options={'disp': True})
    # s = converge(rediscretize(s, new_M=24), method='lstsq', verbose=True).orbit


    # o = read_h5('AntisymmetricOrbitKS_L17p590_T17p146.h5', directory='orbithunter')
    # for m in np.arange(o.N, 66, 2):
    #     o = rediscretize(o, new_N=m)
    #     print(o.residual(), m,m,m,m,m,m)
    #     r = converge(o, method='lstsq', orbit_tol=10**-12, verbose=True)
    #     o = r.orbit.copy()
    #
    # for m in np.arange(o.M, 66, 2):
    #     o = rediscretize(o, new_M=m)
    #     print(o.residual(), m,m,m,m,m,m)
    #     r = converge(o, method='lstsq', orbit_tol=10**-12, verbose=True)
    #     o = r.orbit.copy()
    #
    # wiggle = o.convert(to='field')
    # wiggle.to_h5(directory='../data/tiles/')

    #
    # RelativeOrbitKS_L27p526_T68p730
    # RelativeEquilibriumOrbitKS_L28p810_T21p142
    # EquilibriumOrbitKS_L11p39
    # AntisymmetricOrbitKS_L34p913_T51p540
    # OrbitKS_L31p245_T46p754
    #     self = read_h5('RelativeEquilibriumOrbitKS_L28p810_T21p142.h5', directory='../data/test_data/')
    #     other = read_h5('RelativeEquilibriumOrbitKS_L28p810_T21p142.h5', directory='../data/test_data/', class_name='RelativeOrbitKS')
    #     st = self.convert(to='s_modes').change_reference_frame(to='physical').state
    #     calculate_spatial_shift(st, self.L, n_modes=5)

    # for orbit_name in ['RelativeEquilibriumOrbitKS_L28p810_T21p142.h5', 'EquilibriumOrbitKS_L11p39.h5',
    #                    'ShiftReflectionOrbitKS_L28p847_T97p242.h5', 'RelativeOrbitKS_L27p526_T68p730.h5',
    #                    'AntisymmetricOrbitKS_L34p913_T51p540.h5', 'OrbitKS_L31p245_T46p754.h5']:
    tmp = id(test)
    orbit_name = 'AntisymmetricOrbitKS_L34p913_T51p540.h5'
    self = read_h5(orbit_name, directory='../data/test_data/')
    self = rediscretize(self, new_M=self.M+2)
    t0 = time.time()
    print(orbit_name)
    r = converge(self,  method='lstsq', orbit_tol=1e-12, verbose=True)
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