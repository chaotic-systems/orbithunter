import os
import sys
import glob
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../..')))
from orbithunter import *
import numpy as np
import pandas as pd

def main(*args, **kwargs):
    overwrite = False
    directory_root = 'C:/Users/Matt/Desktop/orbithunter/data/local/'
    fail_counter = 0
    search = ['C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/blocks/**/*.h5',
              'C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/trawl/**/*.h5',
              'C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/GuBuCv17/**/*.h5',
              'C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/tiles/**/*.h5',
              'C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/continuation/**/*.h5',
              'C:/Users/matt/Desktop/gudorf/KS/python/data_and_figures/gluing/**/*.h5',
              'C:/Users/matt/Desktop/orbithunter/data/**/*.h5'
              ]
    log_path = os.path.abspath(os.path.join(__file__ + '../../../../data/logs/refurbish_log.csv'))
    entry = True
    for search_directory in search:
        for orbit_h5 in glob.glob(search_directory, recursive=True):
            if orbit_h5.upper().count('FAIL') != 0 or orbit_h5.upper().count('OTHER') != 0 \
                    or orbit_h5.upper().count('BLOCKS2') or orbit_h5.upper().count('LOCAL\\TILES'):
                pass
            else:
                try:
                    orbit_ = read_h5(orbit_h5, data_format='kstori', check=True)
                except KeyError:
                    try:
                        orbit_ = read_h5(orbit_h5, data_format='orbithunter_old', check=True)
                    except KeyError:
                        orbit_ = read_h5(orbit_h5,  check=True)
                directory = orbit_h5.split(os.path.basename(orbit_h5))[0]
                if orbit_.N * orbit_.M > 64 * 128:
                    orbit_ = orbit_.reshape()
                if search_directory.count('initial') > 0:
                    orbit_.to_h5(directory=directory, verbose=True)
                    orbit_.plot(show=False, save=True, directory=directory, verbose=True)
                else:

                    if entry and not os.path.isfile(log_path):
                        refurbish_log_ = pd.Series(orbit_h5).to_frame(name='filename')
                    elif entry:
                        refurbish_log_ = pd.read_csv(log_path, index_col=0)

                    if orbit_h5 in np.array(refurbish_log_.applymap(os.path.abspath).values).ravel().tolist():
                        if not overwrite:
                            continue

                    if not overwrite and os.path.abspath(orbit_h5) in np.array(refurbish_log_.applymap(
                            os.path.abspath).values).ravel().tolist():
                        print(orbit_h5, ' already logged.')
                        continue
                    else:
                        print('\n' + 'Refurbishing ' + orbit_h5)
                        converge_result = converge(orbit_, ftol=10**-15, precision='machine', method='hybrid', verbose=True)
                        # Want lower residual if possibile, but the tolerance sent to converge is the termination tolerance not
                        # minimum.
                        if (converge_result.orbit.residual() < np.product(orbit_.field_shape) * 1e-10) \
                                and converge_result.exit_code != 4:
                            dat_directory = os.path.abspath(os.path.join(directory_root.split('/data/')[0],
                                                                         './data/local/', str(converge_result.orbit),
                                                                         './'))
                            # fig_directory = os.path.abspath(os.path.join(directory_root.split('/data/')[0],
                            #                                              './figs/local/',str(converge_result.orbit),
                            #                                              './'))
                            redone_orbit_h5 = os.path.abspath(os.path.join(dat_directory,
                                                              './',converge_result.orbit.parameter_dependent_filename()))
                            converge_result.orbit.to_h5(directory=dat_directory, verbose=True)
                            converge_result.orbit.plot(directory=dat_directory, show=False,
                                                       save=True, verbose=True)
                            refurbish_log_ = pd.concat((refurbish_log_,
                                                        pd.Series([redone_orbit_h5, orbit_h5]).to_frame(name='filename')), axis=0)
                        else:

                            dat_directory = os.path.abspath(os.path.join(directory_root.split('/data/')[0],
                                                                         './data/local/unconverged/',str(converge_result.orbit),
                                                                         './'))
                            orbit_.to_h5(orbit_h5, directory=dat_directory, verbose=True)
                            orbit_.plot(filename=orbit_h5, show=False, save=True, directory=dat_directory, verbose=True)
                            fail_counter += 1

                        refurbish_log_ = pd.concat((refurbish_log_,
                                                    pd.Series(orbit_h5).to_frame(name='filename')), axis=0)
                        refurbish_log_.reset_index(drop=True).drop_duplicates().to_csv(log_path)
                    entry = False
    print('There were {} failures to converge'.format(fail_counter))
    return None

if __name__=="__main__":
    sys.exit(main())