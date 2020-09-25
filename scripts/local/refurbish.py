import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import numpy as np
import pandas as pd

def main(*args, **kwargs):
    figs_only = False
    overwrite = False
    same_name = True
    directory_root = 'C:\\Users\\Matt\\Desktop\\data_and_figures\\'
    fail_counter = 0
    search = ['C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\trawl\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\GuBuCv17\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\initial_conditions\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\tiles\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\blocks\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\continuation\\**\\*.h5',
              'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\gluing\\rpo\\data\\**\\*.h5']
    # search = ['C:\\Users\\matt\\Desktop\\orbithunter\\data\\**\\*.h5']
    log_path = os.path.abspath(os.path.join(__file__ + '..\\..\\..\\..\\data\\refurbish_log.csv'))
    entry = True
    for search_directory in search:
        for orbit_h5 in glob.glob(search_directory, recursive=True):
            if orbit_h5.upper().count('FAIL') != 0 or orbit_h5.upper().count('OTHER') != 0 \
                    or orbit_h5.upper().count('BLOCKS2'):
                pass
            else:
                print(orbit_h5)
                try:
                    orbit_ = read_h5(orbit_h5, data_format='kstori', check=True)
                    # orbit_.to_h5(orbit_h5, directory='')
                except KeyError:
                    try:
                        orbit_ = read_h5(orbit_h5, data_format='orbithunter_old', check=True)
                        # orbit_.to_h5(orbit_h5, directory='')
                    except KeyError:
                        orbit_ = read_h5(orbit_h5,  check=True)
                        # orbit_.to_h5(orbit_h5, directory='')
                # continue
                directory = orbit_h5.split(os.path.basename(orbit_h5))[0]
                full_save_name = os.path.join(directory, orbit_.parameter_dependent_filename())
                branch = directory.split('\\data_and_figures\\')[-1]
                dat_directory = os.path.abspath(os.path.join(directory_root, branch.split('\\data\\')[0],
                                                             '.\\data\\', branch.split('\\data\\')[-1]))
                fig_directory = os.path.abspath(os.path.join(directory_root, branch.split('\\data\\')[0],
                                                             '.\\figs\\', branch.split('\\data\\')[-1]))

                if orbit_.N * orbit_.M > 64 * 128:
                    orbit_ = rediscretize(orbit_)

                if (not os.path.isfile(full_save_name)) or (overwrite is True):
                    if figs_only:
                        orbit_.plot(show=False, directory=fig_directory)
                        continue
                    print('\n'+'Refurbishing '+repr(orbit_)+'\n')

                if search_directory.count('initial') > 0:
                    orbit_.to_h5(directory=directory, verbose=True)
                    orbit_.plot(show=False, save=True, directory=fig_directory, verbose=True)
                else:
                    if entry and not os.path.isfile(log_path):
                        refurbish_log_ = pd.Series(orbit_h5).to_frame(name='filename')
                    elif entry:
                        refurbish_log_ = pd.read_csv(log_path, index_col=0)

                    if not overwrite and orbit_h5 in np.array(refurbish_log_.values).tolist():
                        continue
                    else:
                        converge_result = converge(orbit_, precision='machine', method='hybrid', verbose=True)
                        # Want lower residual if possibile, but the tolerance sent to converge is the termination tolerance not
                        # minimum.
                        if (converge_result.orbit.residual() < np.product(orbit_.field_shape) * 1e-10):
                            redone_orbit_h5 = (orbit_h5.split(os.path.basename(orbit_h5))[0]
                                               + converge_result.orbit.parameter_dependent_filename())
                            converge_result.orbit.to_h5(redone_orbit_h5, directory='', verbose=True)
                            converge_result.orbit.plot(filename=redone_orbit_h5, directory='', show=False,
                                                       save=True, verbose=True)
                        else:
                            # orbit_.to_h5(directory=dat_directory, verbose=True)
                            # orbit_.plot(show=False, save=True, directory=fig_directory, verbose=True)
                            orbit_.to_h5(orbit_h5, directory='', verbose=True)
                            orbit_.plot(filename=orbit_h5, show=False, save=True, directory='', verbose=True)
                            fail_counter += 1
                        refurbish_log_ = pd.concat((refurbish_log_,
                                                    pd.Series(orbit_h5).to_frame(name='filename')), axis=0)
                        refurbish_log_.reset_index(drop=True).drop_duplicates().to_csv(log_path)
                    entry = False
    print('There were {} failures to converge'.format(fail_counter))
    return None

if __name__=="__main__":
    sys.exit(main())