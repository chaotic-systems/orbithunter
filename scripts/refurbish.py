import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
import glob

def main(*args,**kwargs):
    figs_only = False
    overwrite = False
    directory_root = 'C:\\Users\\Matt\\Desktop\\data_and_figures\\'
    fail_counter = 0
    for orbit_h5 in glob.glob('C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\**\\*.h5',
                             recursive=True):
        if orbit_h5.upper().count('FAIL') != 0 or orbit_h5.upper().count('OTHER') != 0:
            pass
        else:
            orbit = read_h5(orbit_h5, data_format='kstori')
            sub_directory = os.path.split(orbit_h5.split('data_and_figures')[-1])[0]
            directory = directory_root+sub_directory+'\\'
            full_save_name = os.path.join(directory, orbit.parameter_dependent_filename())
            if (not os.path.isfile(full_save_name)) or (overwrite is True):
                if figs_only:
                    orbit.plot(show=False, directory=directory)
                converge_result = converge(orbit, atol=orbit.M*orbit.N*10**-15)
                if converge_result.exit_code == 1:
                    converge_result.orbit.to_h5(directory=directory, verbose=True)
                    converge_result.orbit.plot(show=False, save=True, directory=directory)
                else:
                    fail_counter += 1
            else:
                pass
    print('There were {} failures to converge'.format(fail_counter))
    return None
if __name__=="__main__":
    sys.exit(main())
