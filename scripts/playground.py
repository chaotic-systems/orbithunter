from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import time

def main(*args, method='adj', **kwargs):
    orbit = read_h5("C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\trawl\\wegolow\\eqva\\data\\eqva_L11.1.h5",
                    state_type='modes', data_format='kstori')
    orbit.plot()
    return None


if __name__=='__main__':
    main()
