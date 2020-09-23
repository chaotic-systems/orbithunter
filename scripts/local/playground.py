from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
def main(*args, **kwargs):


    test = read_h5('OrbitKS_L26p931_T41p266.h5', data_format='orbithunter_old')

    return None

if __name__=='__main__':
    sys.exit(main())