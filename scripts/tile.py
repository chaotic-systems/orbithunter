from __future__ import print_function, division, absolute_import
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from torihunter import *
import numpy as np
import itertools



def main(*args, **kwargs):
    symmetry=kwargs.get('block_symmetry','none')

    for period in range(1,4):
        for speriod in range(1,4):
            retcode = 0
            all_symbolic_combinations_list = list(itertools.product(['0','1','2'],repeat=period*speriod))
            for symbolic_instance in all_symbolic_combinations_list:
                symbolic_instance = list(symbolic_instance)
                # symbolic_instance = ['2','0','1']
                if list(symbolic_instance).count(symbolic_instance[0]) == len(symbolic_instance):
                    print('Symbolic block is all of the same tile, skipping',symbolic_instance)
                else:
                    symbolic_block = np.reshape(symbolic_instance,[period,speriod])

                    trinary_symbolic_repr = [[symbol for symbol in row] for row in symbolic_block[:]]
                    trinary_aliases = []
                    for symbol_row in trinary_symbolic_repr:
                        trinary_row_aliases = []
                        for x in range(len(symbol_row)):
                            symbol_row.append(symbol_row.pop(0))
                            trinary_row_aliases.append(''.join(symbol_row))
                        trinary_aliases.append(trinary_row_aliases)
                    all_aliases = list(set(list(itertools.product(*trinary_aliases))))
                    block_aliases = ['_'.join([symmetry,'_'.join(list(alias))]) for alias in all_aliases]

                    if fh.check_tiling_log(block_aliases,folder_containing_logfile):
                        print('Attempt already made, skipping',block_aliases[0])
                        pass
                    else:
                        print('Attempting to converge', symbolic_block, 'assuming symmetry',symmetry)
                        block_Orbit_high_res = ksinit.symbolic_initial_condition(symbolic_instance,period,speriod,padded=True)
                        blockU,blockN,blockM,blockT,blockL,blockS = block_Orbit_high_res
                        block_pathnames = fh.create_save_data_pathnames(block_Orbit_high_res,directory_structure,custom_name=block_aliases[0],TL=False,NM=False)
                        minN,minM = np.min([16,2**(int(np.log2(max([blockT,2])))+1)]), np.max([16,2**(int(np.log2(blockL)))])
                        for newN,newM in ((n,m) for n in np.arange(minN,129,16) for m in np.arange(minM,65,16)):
                            block_Orbit = disc.rediscretize(block_Orbit_high_res,newN=newN,newM=newM)
                            # ksplot.plot_spatiotemporal_field(block_Orbit,symmetry=symmetry,filename=block_pathnames.initialpng)
                            block_Orbit_adjoint,retcode,res = ks.find_Orbit(block_Orbit,symmetry=symmetry)
                            converged_Orbit,retcode,res = ksdm.find_Orbit(block_Orbit_adjoint,symmetry=symmetry)
                            print('T,L,S',converged_Orbit[-3],converged_Orbit[-2],converged_Orbit[-1])

                            if retcode:
                                print('Solution Converged: Saving to ', block_pathnames.h5)
                                ksplot.plot_spatiotemporal_field(block_Orbit,symmetry=symmetry,filename=block_pathnames.initialpng)
                                ksplot.plot_spatiotemporal_field(block_Orbit_adjoint,symmetry=symmetry,filename=block_pathnames.adjointpng)
                                ksplot.plot_spatiotemporal_field(converged_Orbit,symmetry=symmetry,filename=block_pathnames.finalpng)
                                Orbit_io.export_Orbit(converged_Orbit,block_pathnames.h5,symmetry=symmetry)
                                break
                            else:
                                print('Failure. Trying new discretization.')
                        if retcode:
                            fh.log_tiling_attempt(block_aliases[0],'SUCCESS',folder_containing_logfile)
                        else:
                            fh.log_tiling_attempt(block_aliases[0],'FAILURE',folder_containing_logfile)
    return None




if __name__=="__main__":
    sys.exit(main())


