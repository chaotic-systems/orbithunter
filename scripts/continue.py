from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from torihunter import *
import numpy as np




def main(filename):

    for files in os.listdir(parent_folder):
        folder = os.path.join(os.path.abspath(os.path.join(parent_folder, files,'data')),'')
        if os.path.isdir(folder):
            print('accessing',folder)
            for data_file in os.listdir(folder):
                if data_file.endswith(".h5"):
                    basename = data_file.split('.h5')[0]
                    print(basename)
                    torus = torus_io.import_torus(''.join([folder,data_file]))
                    U,N,M,T,L,S = torus


    return None

from __future__ import print_function, division
import numpy as np
import os

def main(path_to_data,*args,**kwargs):
    symmetry = kwargs.get('symmetry','rpo')
    dimension = kwargs.get('dimension',1)
    deltamodifier = kwargs.get('deltamodifier',1)
    if dimension:
        fixedL = True
        fixedT = False
    else:
        fixedT = True
        fixedL = False

    '''
    Things to include
    If number of continuation steps is too large, warn and quit
    if the predicted time to finish is too large, warn and quit
    continuation in N,M,T,L.
    Choose deltaN,deltaM,deltaT,deltaL
    persistent flag; ability to reduce deltaX if there is failure.
    
    parse filename or directory name info. ''for x in 
    '''
    if os.path.isfile(path_to_data) and path_to_data.endswith(".h5"):
        torus = torus_io.import_torus(path_to_data)
        u,n,m,t,l,s = torus
        # if args:
        #     minD,maxD = args
        # else:
        #     if dimension:
        #         minD,maxD = l-l/8.,l+l/8.
        #     else:
        #         minD,maxD = t-t/4.,t+t/4.
        # if increase:
        #     continuation_loop(torus,maxD,instance_folder,symmetry=symmetry,dimension=dimension,save=save,deltamodifier=deltamodifier)
        # if decrease:
        #     continuation_loop(torus,minD,instance_folder,symmetry=symmetry,dimension=dimension,save=save,deltamodifier=deltamodifier)

    elif os.path.isdir(path_to_data):
        for dirs,subdirs,files in os.walk(path_to_data):
            for d,s,file in dirs,subdirs,files:
                if file.endswith(".h5"):
                    basename = file.split('.h5')[0]
                    print('Starting numerical continuation on', basename)
                    import_filepath = os.path.abspath(os.path.join(d,s,file))
                    torus = torus_io.import_torus(import_filepath)
                    u,n,m,t,l,s = torus

                    if args:
                        minD,maxD = args
                    else:
                        if dimension:
                            minD,maxD = l-l/8.,l+l/8.
                        else:
                            minD,maxD = t-t/4.,t+t/4.
                    if increase:
                        continuation_loop(torus,maxD,instance_folder,symmetry=symmetry,dimension=dimension,save=save,deltamodifier=deltamodifier)
                    if decrease:
                        continuation_loop(torus,minD,instance_folder,symmetry=symmetry,dimension=dimension,save=save,deltamodifier=deltamodifier)

    if dimension:
        torus = (U,N,M,T,D,S)
    else:
        torus = (U,N,M,D,L,S)
    torus,delta = continuation_initial_condition(torus,Dlim,delta,retcode=retcode,dimension=dimension,deltamodifier=deltamodifier)
    print_large_message(instance_folder,delta,N,M,T,L,S,symmetry,fixedL,fixedT)
    while retcode==1:
        torus,retcode,residual = ks.find_torus(torus,symmetry=symmetry,fixedL=fixedL,fixedT=fixedT)
        torus,retcode,_ = ksdm.find_torus(torus,symmetry=symmetry,fixedL=fixedL,fixedT=fixedT)
        if retcode:
            previous_torus = torus
            torus_io.export_data_and_fig(torus,instance_folder,decimals=5,symmetry=symmetry)
            torus,delta = continuation_step(torus,Dlim,delta,retcode=retcode,dimension=dimension)

    return torus


if __name__=='__main__':
    sys.exit(main())


if __name__=='__main__':
    sys.exit(main())



