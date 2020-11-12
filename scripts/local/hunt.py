import os,sys
import numpy as np
import time
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../../../')))
from orbithunter import *
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
    # result = converge(converge(x, verbose=True, method='adj').orbit, method='lstsq', verbose=True, min_step=0.0001)
    # result = converge(converge(x, verbose=True, method='adj').orbit, method='lstsq', verbose=True)
    result = converge(x, method='hybrid', verbose=True,
                      comp_time='excessive',
                      preconditioning=True, pexp=(1, 4))

    if result.orbit.residual() <= result.tol:
        fname_init = ''.join([result.orbit.parameter_dependent_filename(extension=''), '_initial.h5'])
        x.to_h5(filename=fname_init, verbose=True,
                           directory='../../data/local/hunt/')
        x.plot(filename=fname_init, show=False, save=True, verbose=True,
                          directory='../../data/local/hunt/')
        result.orbit.to_h5(verbose=True,
                           directory='../../data/local/hunt/')
        result.orbit.plot(show=False, save=True, verbose=True,
                          directory='../../data/local/hunt/')

    return None


def main():
    parser = ArgumentParser('hunt', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of concurrently runnings jobs; -1 == using all '
                                                             'cores. See joblib\'s Parallel class for more details.')
    parser.add_argument('--cls', default='OrbitKS', help='Orbit class defined by which symmetries are enforced:'
                                                            'Options include: "OrbitKS", "ShiftReflectionOrbitKS", '
                                                            '"RelativeOrbitKS, "AntisymmetricOrbitKS",'
                                                            '"EquilibriumOrbitKS,''default=OrbitKS (no symmetries)')
    parser.add_argument('--T_min', default=20, type=float, help='Smallest possible time-period value')
    parser.add_argument('--T_max', default=200,type=float, help='Largest possible time-period value')
    parser.add_argument('--L_min', default=16, type=float, help='Smallest possible space-period value generated')
    parser.add_argument('--L_max', default=64, type=float, help='Largest possible space-period value generated')
    parser.add_argument('--field_magnitude', default=5, type=float, help='Value of L_infinite norm to initialize with.')
    parser.add_argument('--n_trials', default=1, type=int, help='Number of initial conditions to try')
    parser.add_argument('--solver', default='hybrid', type=str, help='Solver to use')
    parser.add_argument('--seed_min', default=0, type=int,
                        help='Starting value for random seeds for reproducibility')
    parser.add_argument('--verbose', default=True, type=str2bool, help='Whether or not to print stats progress')
    parser.add_argument('--spectrum', default='gaussian', type=str, help='The spectrum modulation to apply to the '
                                                                         'randomly initialized modes.')
    parser.add_argument('--constrain', default=-1, type=int, help='Dimension to constrain, provided as the array axis.')
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    cls = parse_class(args.cls)
    n_trials = int(args.n_trials)
    verbose = args.verbose
    solve = args.solver
    seed_min = args.seed_min
    mode_spectrum = args.spectrum
    seeds = np.arange(int(seed_min), int(seed_min+n_trials))
    T_min, T_max = float(args.T_min), float(args.T_max)
    L_min, L_max = float(args.L_min), float(args.L_max)
    mag = args.field_magnitude
    t = time.time()
    for s in seeds:
        orbit_ = cls(seed=s, T_min=T_min, T_max=T_max, L_min=L_min, L_max=L_max, nonzero_parameters=True,
                     spectrum=mode_spectrum).rescale(mag)
        hunt(orbit_)

    print('{} trials took {} to complete with {} jobs'.format(n_trials, time.time()-t, n_jobs))
    return None


if __name__=='__main__':
    sys.exit(main())
