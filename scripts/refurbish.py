import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *

def main(*args, **kwargs):
    figs_only = False
    overwrite = False
    directory_root = 'C:\\Users\\Matt\\Desktop\\data_and_figures\\'
    fail_counter = 0
    search_directory = 'C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\**\\*.h5'
    for orbit_h5 in glob.glob(search_directory, recursive=True):
        if orbit_h5.upper().count('FAIL') != 0 or orbit_h5.upper().count('OTHER') != 0 \
                or orbit_h5.upper().count('BLOCKS2'):
            pass
        else:
            orbit_ = read_h5(orbit_h5, data_format='kstori')
            sub_directory = os.path.split(orbit_h5.split('data_and_figures')[-1])[0]
            directory = directory_root+sub_directory+'\\'
            full_save_name = os.path.join(directory, orbit_.parameter_dependent_filename())
            if orbit_.N * orbit_.M > 64*128:
                orbit_ = rediscretize(orbit_, parameter_based=True)

            if (not os.path.isfile(full_save_name)) or (overwrite is True):
                if figs_only:
                    orbit_.plot(show=False, directory=directory)
                    continue
                print('Starting optimization on', repr(orbit_))
                print(orbit_h5)
                converge_result = converge(orbit_, max_iter=10000, verbose=True)
                if converge_result.exit_code == 1:
                    converge_result.orbit.to_h5(directory=directory, verbose=True)
                    # converge_result.orbit.plot(show=False, save=True, directory=directory)
                else:
                    fail_counter += 1
            else:
                pass
    print('There were {} failures to converge'.format(fail_counter))
    return None

if __name__=="__main__":
    sys.exit(main())
