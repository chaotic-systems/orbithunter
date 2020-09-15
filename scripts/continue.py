from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *

def main(filename, *args,**kwargs):


    '''
    Things to include
    If number of continuation steps is too large, warn and quit
    if the predicted time to finish is too large, warn and quit
    continuation in N,M,T,L.
    Choose deltaN,deltaM,deltaT,deltaL
    persistent flag; ability to reduce deltaX if there is failure.
    
    parse filename or directory name info. ''for x in 
    '''
    # if os.path.isfile(path_to_data) and path_to_data.endswith(".h5"):



    # return Orbit



if __name__=='__main__':
    sys.exit(main())


