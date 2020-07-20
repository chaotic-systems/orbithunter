import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import orbithunter as th

def main(*args,**kwargs):
    # Re-do all previous calculations; relatively but not extraordinarily time consuming.
    # Walk through the parent folder to find all data.
    overwrite=False
    for dirpath, dirname, filenames in os.walk('C:\\Users\\matt\\Desktop\\gudorf\\KS\\python\\data_and_figures\\'):
        if dirpath.count('fail') != 0:
            # Skip over "fail" folders, which contain only placeholder files to avoid redundant bad calculations.
            continue
        else:
            for f in filenames:
                if f.split('_')[0] in ['Torus', 'AntisymmetricTorus',
                                       'ShiftReflectionTorus', 'RelativeTorus', 'EquilibriumTorus']:
                    continue
                cls = th.parse_class(f)
                if cls is not None and f.endswith('.h5'):
                    if dirpath.count('block') != 0:
                        # Different naming format for tiles/blocks
                        x = th.read_h5(f, directory=dirpath)
                        x.convert(to='modes')
                        basename = '_'.join((f.split('.')[0]).split('_')[1:])
                        h5name = '_'.join([x.__class__.__name__, basename + '.h5'])
                        pngname ='_'.join([x.__class__.__name__, basename + '.png'])
                        savename = os.path.join(dirpath, h5name)
                    else:
                        # Normal naming format.
                        x = th.read_h5(f, directory=dirpath)
                        # x.convert(to='modes')
                        h5name=x.parameter_dependent_filename('.h5')
                        pngname=x.parameter_dependent_filename('.png')
                        savename = os.path.join(dirpath, h5name)

                    if os.path.isfile(savename) and overwrite == False:
                        # If the target filename already exists then skip a redundant calculation.
                        continue
                    else:
                        # x = rediscretize(x, parameter_based=True)
                        print(f, [x.N, x.M])
                        sys.stdout.flush()
                        result = th.converge(x, verbose=True)
                        if (result.exit_code == 1) | (result.exit_code == 3):
                            result.orbit.to_h5(filename=h5name, directory=dirpath)
                            figdirpath = os.path.abspath(os.path.join(dirpath, '../figs/'))
                            result.orbit.plot(show=False, save=True, filename=pngname,
                                              verbose=True, directory=figdirpath)
                        else:
                            sys.stdout.flush()
    return None
if __name__=="__main__":
    sys.exit(main())
