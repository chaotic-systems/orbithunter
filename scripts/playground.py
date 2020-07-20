from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time

def main(*args, method='adj', **kwargs):

    test = ShiftReflectionOrbitKS()
    defect = read_h5("C:\\Users\\Matt\\Desktop\\orbithunter_test_data\\rpo_L13p02_T15.h5")
    test = defect.residual()
    x = defect.convert(to='modes')
    y = x.rotate(distance=5)
    test2 = y.residual()
    result = converge(defect, method='lstsq', fixedparams=(False, False, False), verbose=True)
    t, code = result.orbit, result.exit_code
    return None


if __name__=='__main__':
    main()
