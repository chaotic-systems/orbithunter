import os,sys
import numpy as np
import time
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
from orbithunter import *
from joblib import Parallel, delayed
from argparse import ArgumentParser, ArgumentTypeError, ArgumentDefaultsHelpFormatter


def str2bool(val):
    if isinstance(val, bool):
       return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def hunt(x, verbose=True):
    if verbose:
        print('Beginning search for {}'.format(repr(x)))
    result = converge(x, verbose=verbose)
    if result.exit_code:
        result.orbit.to_h5(directory='default',)
        result.orbit.plot(show=False, save=True, )

    return None


def main(*args, method='hybrid', **kwargs):
    parser = ArgumentParser('hunt', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n_jobs', default=1, type=int, help='Number of concurrently runnings jobs; -1 == using all '
                                                             'cores. See joblib\'s Parallel class for more details.')
    parser.add_argument('-cls', default='Orbit', help='Orbit class defined by which symmetries are enforced:'
                                                            'Options include: "Orbit", "ShiftReflectionOrbit", '
                                                            '"RelativeOrbit, "AntisymmetricOrbit", "EquilibriumOrbit,'
                                                                'default=Orbit (no symmetries)')
    parser.add_argument('-T_min', default=20, type=float, help='Minimum time-period value to search over')
    parser.add_argument('-T_max', default=200,type=float, help='Maximum time-period value to search over')
    parser.add_argument('-L_min', default=16, type=float, help='Minimum space-period value to search over')
    parser.add_argument('-L_max', default=64, type=float, help='Maximum space-period value to search over')
    parser.add_argument('-n_trials', default=1, type=int, help='Number of attempts to converge')
    parser.add_argument('-solver', default='hybrid', type=str, help='Solver to use: "descent", "lstsq", "hybrid".')
    parser.add_argument('-verbose', default=False, type=str2bool, help='Whether or not to print stats, not recommended'
                                                                       'for n_jobs != 1.')

    args = parser.parse_args()

    n_jobs = int(args.n_jobs)
    cls, _ = parse_class(args.cls)
    n_trials = int(args.n_trials)
    verbose = args.verbose
    # verbose = False

    T_min, T_max = float(args.T_min), float(args.T_max)
    L_min, L_max = float(args.L_min), float(args.L_max)

    trange = (T_max-T_min)*np.random.rand(n_trials) + T_min
    lrange = (L_max-L_min)*np.random.rand(n_trials) + L_min
    domains = zip(trange, lrange)
    t = time.time()
    # for T, L in domains:
    #     hunt(cls(T=T, L=L), verbose=verbose)
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(delayed(hunt)(cls(T=T, L=L), verbose=verbose) for (T, L) in domains)

    print('{} trials took {} to complete with {} jobs'.format(n_trials, time.time()-t, n_jobs))
    return None


if __name__=='__main__':
    sys.exit(main())
