from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from scipy.linalg import lstsq
from scipy.optimize import minimize, root
import sys
import numpy as np

__all__ = ['converge']


class OrbitResult(dict):
    """ Represents the result of applying converge; format copied from SciPy scipy.optimize.OptimizeResult.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    exit_code : int
        Integer which tracks the type of exit from whichever numerical algorithm was applied.
        See Notes for more details.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.

    The descriptions for each value of exit_code are as follows. In order, the codes [0,1,2,3,4,5] == 0:
    0 : Failed to converge
    1 : Converged
    2:
        print('\nFailed to converge. Maximum number of iterations reached.'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 3:
        print('\nConverged to an errant equilibrium'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 4:
        print('\nConverged to the trivial u(x,t)=0 solution')
    elif exit_code == 5:
        print('\n Relative periodic orbit converged to periodic orbit with no shift.')

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def converge(orbit, *args, method='hybrid', **kwargs):
    orbit.convert(to='modes', inplace=True)
    if kwargs.get('verbose', False):
        print('Starting {} numerical method. Initial residual {}'.format(method, orbit.residual()))

    if method == 'hybrid':
        adjoint_orbit, exit_code = _adjoint_descent(orbit, **kwargs)
        result_orbit, exit_code = _gauss_newton(adjoint_orbit, **kwargs)
    elif method == 'adj':
        result_orbit, exit_code = _adjoint_descent(orbit,  **kwargs)
    elif method == 'lstsq':
        result_orbit, exit_code = _gauss_newton(orbit, **kwargs)
    elif method in ['lsqr', 'lsmr']:
        result_orbit, exit_code = _scipy_sparse_linalg_solver_wrapper(orbit, *args, method=method, **kwargs)
    elif method in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b',
                    'tnc', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact']:
        result_orbit, exit_code = _scipy_optimize_minimize_wrapper(orbit, method=method, **kwargs)

    elif method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing',
                    'diagbroyden', 'excitingmixing', 'krylov','df-sane']:
        result_orbit, exit_code = _scipy_optimize_root_wrapper(orbit, method=method, **kwargs)
    else:
        raise ValueError('Unknown solver %s' % method)

    if kwargs.get('verbose', False):
        print_exit_messages(result_orbit, exit_code)

    return OrbitResult(orbit=result_orbit, exit_code=exit_code)


def _adjoint_descent(orbit, parameter_constraints=(False, False, False), **kwargs):
    # Specific modifying exponent for changes in period, domain_size
    # Absolute tolerance for the descent method.
    atol = 10**-6
    max_iter = kwargs.get('max_iter', 32*orbit.N*orbit.M)
    preconditioning = kwargs.get('preconditioning', True)

    step_size = 1
    n_iter = 0
    # By default assume failure
    exit_code = 0

    mapping = orbit.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)

    while residual > atol and n_iter < max_iter:
        # Calculate the step
        dx = orbit.rmatvec(mapping, parameter_constraints=parameter_constraints,
                           preconditioning=preconditioning)
        # Apply the step
        next_orbit = orbit.increment(dx, stepsize=-1.0*step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.spatiotemporal_mapping()
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(apply_mapping=False)
        while next_residual >= residual:
            # reduce the step size until minimum is reached or residual decreases.
            step_size = 0.5*step_size
            next_orbit = orbit.increment(dx, stepsize=-1.0*step_size)
            next_mapping = next_orbit.spatiotemporal_mapping()
            next_residual = next_mapping.residual(apply_mapping=False)
            if step_size <= 10**-8:
                return orbit, exit_code
        else:
            # Update and restart loop if residual successfully decreases.
            orbit, mapping, residual = next_orbit, next_mapping, next_residual
            n_iter += 1
            if kwargs.get('verbose', False):
                if np.mod(n_iter, 2500) == 0:
                    print('Step number {} residual {}'.format(n_iter, orbit.residual()))
                elif np.mod(n_iter, 100) == 0:
                    print('.', end='')
                sys.stdout.flush()

    if orbit.residual() <= atol:
        orbit, exit_code = orbit.status()
    elif n_iter == max_iter:
        exit_code = 2

    return orbit, exit_code


def _gauss_newton(orbit, max_iter=500, parameter_constraints=(False, False, False), max_damp_factor=9, **kwargs):
    orbit.convert(inplace=True, to='modes')
    preconditioning = kwargs.get('preconditioning', False)
    atol = kwargs.get('atol', orbit.N * orbit.M * 10**-15)

    n_iter = 0
    exit_code = 0
    damp_factor = 0

    residual = orbit.residual()
    while residual > atol and n_iter < max_iter:
        damp_factor = 0
        n_iter += 1
        A = orbit.jacobian(parameter_constraints=parameter_constraints, preconditioning=preconditioning)
        b = -1.0 * orbit.spatiotemporal_mapping(preconditioning=preconditioning).state.ravel()
        dorbit = _state_vector_to_orbit(orbit, lstsq(A, b.reshape(-1, 1))[0],
                                        parameter_constraints=parameter_constraints)
        
        # To avoid redundant function calls, store optimization variables using
        # clunky notation.
        next_orbit = orbit.increment(dorbit, stepsize=2**-damp_factor)
        next_residual = next_orbit.residual()
        while next_residual > residual:
            # Continues until either step is too small or residual is decreases
            damp_factor += 1
            next_orbit = orbit.increment(dorbit, stepsize=2**-damp_factor)
            next_residual = next_orbit.residual()
            if damp_factor > max_damp_factor:
                return orbit, exit_code
        else:
            # Executed when step decreases residual and is not too short
            orbit = next_orbit
            residual = next_residual

            if kwargs.get('verbose', False):
                print(damp_factor, end='')
                if np.mod(n_iter, 50) == 0:
                    print('step ', n_iter,' residual ', orbit.residual())
                sys.stdout.flush()

    if orbit.residual() <= atol:
        orbit, exit_code = orbit.status()
    elif n_iter == max_iter:
        exit_code = 2
    elif damp_factor > max_damp_factor:
        exit_code = 0
    else:
        exit_code = 0

    return orbit, exit_code


def _state_vector_to_orbit(orbit, correction_vector, parameter_constraints=(False, False, False)):
    """ Utility to convert from numpy array to orbithunter format
    :param orbit:
    :param correction_vector:
    :param parameter_constraints:
    :return:
    """
    correction_vector = correction_vector.reshape(-1, 1)
    mode_shape, mode_size = orbit.state.shape, orbit.state.size
    d_modes = correction_vector[:mode_size]
    # slice the changes to parameters from vector
    d_params = correction_vector[mode_size:].ravel()

    for i, constrained in enumerate(parameter_constraints):
        if constrained or (i == len(d_params)):
            d_params = np.insert(d_params, i, 0)
    dT, dL, dS = d_params
    correction_orbit = orbit.__class__(state=np.reshape(d_modes, mode_shape), state_type='modes',
                                       T=dT, L=dL, S=dS)
    return correction_orbit


def _scipy_sparse_linalg_solver_wrapper(orbit, max_damp_factor=8, atol=1e-06, btol=1e-06,
                                        method='lsqr', maxiter=None, conlim=1e+08,
                                        show=False, calc_var=False, **kwargs):

    linear_operator_shape = (orbit.state.size, orbit.state_vector().size)
    istop = 1

    # Return codes that represent good results from the SciPy least-squares solvers.
    good_codes = [0, 1, 2, 4, 5]
    while (orbit.residual() > atol) and (istop in good_codes):
        # The operator depends on the current state; A=A(orbit)
        mv = _matvec_wrapper(orbit, preconditioning=False)
        rmv = _rmatvec_wrapper(orbit, preconditioning=False)
        orbit_linear_operator = LinearOperator(linear_operator_shape, mv, rmatvec=rmv, dtype=float)
        b = -1.0 * orbit.spatiotemporal_mapping().state.reshape(-1, 1)
        damp_factor = 0
        if method == 'lsmr':
            result_tuple = lsmr(orbit_linear_operator, b, atol=atol, btol=btol,
                                conlim=conlim, maxiter=maxiter, show=show)
        elif method == 'lsqr':
            # Depends heavily on scaling of the problem.
            result_tuple = lsqr(orbit_linear_operator, b, atol=atol, btol=btol, conlim=conlim,
                                iter_lim=maxiter, show=show, calc_var=calc_var)
        else:
            raise ValueError('Unknown solver %s' % method)

        x = result_tuple[0]
        istop = result_tuple[1]
        cond = result_tuple[6]
        dorbit = _state_vector_to_orbit(orbit, x, **kwargs)
        next_orbit = orbit.increment(dorbit)
        while next_orbit.residual() > orbit.residual():
            damp_factor += 1
            next_orbit = orbit.increment(dorbit, stepsize=2**-damp_factor)
            if damp_factor > max_damp_factor:
                return orbit, False
        orbit = next_orbit
        print(damp_factor, end='')

    return orbit, True


def _scipy_optimize_minimize_wrapper(orbit, method=None, bounds=None,
                                     tol=None, callback=None, options=None, **kwargs):
    if method in ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc']:
        jac_ = _cost_function_jac
    else:
        jac_ = None

    result = minimize(_cost_function, orbit.state_vector(), args=(orbit,),
                      method=method, jac=jac_, bounds=bounds, tol=tol,
                      callback=callback, options=options)
    x, n_int, exit_code = result.x, result.nit, result.success
    result_orbit = _state_vector_to_orbit(orbit, x)
    return result_orbit, exit_code


def _scipy_optimize_root_wrapper(orbit, method=None, bounds=None,
                                 tol=None, callback=None, options=None):

    result = minimize(_cost_function, orbit.state_vector(), args=(orbit,),
                      method=method, jac=_cost_function_jac, bounds=bounds, tol=tol,
                      callback=callback, options=options)
    x, n_int, exit_code = result.x, result.nit, result.exit_code
    result_orbit = _state_vector_to_orbit(orbit, x)
    return result_orbit, exit_code


def _matvec_wrapper(orbit, **kwargs):

    def mv_func(v):
        # _state_vector_to_orbit turns state vector into class object.
        v_orbit = _state_vector_to_orbit(orbit, v, **kwargs)
        return orbit.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

    return mv_func


def _rmatvec_wrapper(orbit, **kwargs):

    def rmv_func(v):
            # _process_newton_step turns state vector into class object.
            v_orbit = _state_vector_to_orbit(orbit, v, **kwargs)
            rmatvec_result = orbit.rmatvec(v_orbit, **kwargs).state_vector().reshape(-1, 1)
            return rmatvec_result

    return rmv_func


def _cost_function(x, *args):
    '''
    :param x0: (n,) numpy array
    :param args: time discretization, space discretization, subClass from orbit.py
    :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
    '''

    '''
    Note that passing Class as a function avoids dangerous statements using eval()
    '''
    orbit = args[0]
    x_orbit = _state_vector_to_orbit(orbit, x)
    return x_orbit.residual()


def _cost_function_jac(x, *args):
    """ The jacobian of the cost function (scalar) can be expressed as a vector product

    Parameters
    ----------
    x
    args

    Returns
    -------

    Notes
    -----
    The gradient of 1/2 F^2 = J^T F, rmatvec is a function which does this matrix vector product
    Will always use preconditioned version by default, not sure if wise.
    """
    orbit = args[0]
    x_orbit = _state_vector_to_orbit(orbit, x)
    return x_orbit.rmatvec(x_orbit.spatiotemporal_mapping()).state_vector().ravel()


def print_exit_messages(orbit, exit_code):
    messages = []
    if exit_code == 0:
        print('\nFailed to converge. Exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 1:
        print('\nConverged. exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 2:
        print('\nFailed to converge. Maximum number of iterations reached.'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 3:
        print('\nConverged to an errant equilibrium'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 4:
        print('\nConverged to the trivial u(x,t)=0 solution')
    elif exit_code == 5:
        print('\n Relative periodic orbit converged to periodic orbit with no shift.')
    return None



