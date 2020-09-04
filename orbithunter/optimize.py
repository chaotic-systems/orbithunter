from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from scipy.linalg import lstsq
from scipy.optimize import minimize, root, newton_krylov
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


def converge(orbit_, *args, method='hybrid', **kwargs):
    orbit_.convert(to='modes', inplace=True)
    if kwargs.get('verbose', False):
        print('Starting {} numerical method. Initial residual {}'.format(method, orbit_.residual()))

    if method == 'hybrid':
        adjoint_orbit, nit_a, _ = _adjoint_descent(orbit_, **kwargs)
        result_orbit, nit_gn, exit_code = _gauss_newton(adjoint_orbit, **kwargs)
        nit = (nit_a, nit_gn)
    elif method == 'adj':
        result_orbit, nit, exit_code = _adjoint_descent(orbit_, **kwargs)
    elif method == 'lstsq':
        result_orbit, nit, exit_code = _gauss_newton(orbit_, **kwargs)
    elif method == 'gmres':
        result_orbit, nit, exit_code = _gmres(orbit_, **kwargs)
    elif method == 'newton_krylov':
        result_orbit, nit, exit_code = _scipy_newton_krylov_wrapper(orbit_, **kwargs)
    elif method in ['lsqr', 'lsmr']:
        result_orbit, nit, exit_code = _scipy_sparse_linalg_solver_wrapper(orbit_, *args, method=method, **kwargs)
    elif method in ['cg', 'l-bfgs-b']:
        result_orbit, nit, exit_code = _scipy_optimize_minimize_wrapper(orbit_, method=method, **kwargs)
    elif method in ['lm', 'linearmixing', 'excitingmixing']:
        result_orbit, nit, exit_code = _scipy_optimize_root_wrapper(orbit_, method=method, **kwargs)

    else:
        raise ValueError('Unknown solver %s' % method)

    if kwargs.get('verbose', False):
        print_exit_messages(result_orbit, exit_code)

    return OrbitResult(orbit=result_orbit, exit_code=exit_code, nit=nit)


def _adjoint_descent(orbit_, **kwargs):
    # Specific modifying exponent for changes in period, domain_size
    # Absolute tolerance for the descent method.
    atol = 10 ** -6
    max_iter = kwargs.get('max_iter', 16 * orbit_.N * orbit_.M)
    preconditioning = kwargs.get('preconditioning', True)

    step_size = 1
    n_iter = 0
    # By default assume failure
    exit_code = 0

    mapping = orbit_.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)

    while residual > atol and n_iter < max_iter:
        # Calculate the step
        dx = orbit_.rmatvec(mapping, **kwargs)
        # Apply the step
        next_orbit = orbit_.increment(dx, stepsize=-1.0 * step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.spatiotemporal_mapping()
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(apply_mapping=False)

        while next_residual >= residual:
            # reduce the step size until minimum is reached or residual decreases.
            step_size = 0.5 * step_size
            next_orbit = orbit_.increment(dx, stepsize=-1.0 * step_size)
            next_mapping = next_orbit.spatiotemporal_mapping()
            next_residual = next_mapping.residual(apply_mapping=False)
            if step_size <= 10 ** -8:
                return orbit_, n_iter, exit_code
        else:
            # Update and restart loop if residual successfully decreases.
            orbit_, mapping, residual = next_orbit, next_mapping, next_residual
            n_iter += 1
            if kwargs.get('verbose', False):
                if np.mod(n_iter, 2500) == 0:
                    print('Step number {} residual {}'.format(n_iter, orbit_.residual()))
                elif np.mod(n_iter, 100) == 0:
                    print('.', end='')
                sys.stdout.flush()

    if orbit_.residual() <= atol:
        orbit, exit_code = orbit_.status()
    elif n_iter == max_iter:
        exit_code = 2

    return orbit_, n_iter, exit_code


def _gauss_newton(orbit_, max_iter=500, max_damp_factor=9, **kwargs):
    # use .get() so orbit_ can be referenced.
    atol = kwargs.get('atol', orbit_.N * orbit_.M * 10 ** -15)

    n_iter = 0
    exit_code = 0
    damp_factor = 0

    residual = orbit_.residual()
    while residual > atol and n_iter < max_iter:
        damp_factor = 0
        n_iter += 1
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * orbit_.spatiotemporal_mapping(**kwargs).state.ravel()
        dorbit = orbit_.from_numpy_array(lstsq(A, b.reshape(-1, 1))[0], **kwargs)

        # To avoid redundant function calls, store optimization variables using
        # clunky notation.
        next_orbit = orbit_.increment(dorbit, stepsize=2 ** -damp_factor)
        next_residual = next_orbit.residual()
        while next_residual > residual:
            # Continues until either step is too small or residual is decreases
            damp_factor += 1
            next_orbit = orbit_.increment(dorbit, stepsize=2 ** -damp_factor)
            next_residual = next_orbit.residual()
            if damp_factor > max_damp_factor:
                return orbit_, n_iter, exit_code
        else:
            # Executed when step decreases residual and is not too short
            orbit_ = next_orbit
            residual = next_residual

            if kwargs.get('verbose', False):
                print(damp_factor, end='')
                if np.mod(n_iter, 50) == 0:
                    print('step ', n_iter, ' residual ', orbit_.residual())
                sys.stdout.flush()

    if orbit_.residual() <= atol:
        orbit, exit_code = orbit_.status()
    elif n_iter == max_iter:
        exit_code = 2
    elif damp_factor > max_damp_factor:
        exit_code = 0
    else:
        exit_code = 0

    return orbit_, n_iter, exit_code


def _gmres(orbit_, max_iter=500, parameter_constraints=(False, False, False), max_damp_factor=9, **kwargs):
    # A, b, x0=None, tol=1e-05, restart=None, maxiter=None, M=None, callback=None, atol=None, callback_type=None
    return None, None, None


def _scipy_sparse_linalg_solver_wrapper(orbit_, max_damp_factor=8, atol=1e-06, btol=1e-06,
                                        method='lsqr', maxiter=None, conlim=1e+08,
                                        show=False, calc_var=False, **kwargs):
    linear_operator_shape = (orbit_.state.size, orbit_.state_vector().size)
    istop = 1
    exit_code = 0
    itn = 0
    # Return codes that represent good results from the SciPy least-squares solvers.
    good_codes = [0, 1, 2, 4, 5]
    while (orbit_.residual() > atol) and (istop in good_codes):
        # The operator depends on the current state; A=A(orbit)
        def rmv_func(v):
            # _process_newton_step turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v, **kwargs)
            v_orbit.T, v_orbit.L, v_orbit.S = orbit_.T, orbit_.L, orbit_.S
            rmatvec_result = orbit_.rmatvec(v_orbit, **kwargs).state_vector().reshape(-1, 1)
            return rmatvec_result

        def mv_func(v):
            # _state_vector_to_orbit turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v, **kwargs)
            v_orbit.T, v_orbit.L, v_orbit.S = orbit_.T, orbit_.L, orbit_.S
            return orbit_.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

        orbit_linear_operator = LinearOperator(linear_operator_shape, mv_func, rmatvec=rmv_func, dtype=float)
        b = -1.0 * orbit_.spatiotemporal_mapping().state.reshape(-1, 1)
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
        itn += result_tuple[2]

        dorbit = orbit_.from_numpy_array(x, **kwargs)
        next_orbit = orbit_.increment(dorbit)
        while next_orbit.residual() > orbit_.residual():
            damp_factor += 1
            next_orbit = orbit_.increment(dorbit, stepsize=2 ** -damp_factor)
            if damp_factor > max_damp_factor:
                return orbit_, itn, exit_code
        else:
            orbit_ = next_orbit
            if kwargs.get('verbose', False):
                print(damp_factor, end='')
    exit_code = 1
    return orbit_, itn, exit_code


def _scipy_optimize_minimize_wrapper(orbit_, method=None, bounds=None,
                                     tol=None, callback=None, options=None, **kwargs):
    
    def _cost_function_scipy_minimize(x):
        '''
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        '''
    
        '''
        Note that passing Class as a function avoids dangerous statements using eval()
        '''
        x_orbit = orbit_.from_numpy_array(x)
        return x_orbit.residual()
    
    
    def _cost_function_jac_scipy_minimize(x):
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

        x_orbit = orbit_.from_numpy_array(x)
        return x_orbit.rmatvec(x_orbit.spatiotemporal_mapping()).state_vector().ravel()
    
    if method in ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc']:
        jac_ = _cost_function_jac_scipy_minimize
    else:
        jac_ = None

    result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                      method=method, jac=jac_, bounds=bounds, tol=tol,
                      callback=callback, options=options)
    result_orbit = orbit_.from_numpy_array(result.x)
    return result_orbit, result.nit, result.success


def _scipy_optimize_root_wrapper(orbit_, method=None, tol=None, callback=None, options=None, **kwargs):
    # I really don't like this workaround but scipy.optimize.root can't handle unexpected arguments.
    parameter_constraints = kwargs.get('parameter_constraints', (False, False, False))

    # define the functions using orbit instance within scope instead of passing orbit 
    # instance as arg to scipy functions. 

    def _cost_function_scipy_root(x):
        '''
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        '''
    
        '''
        Note that passing Class as a function avoids dangerous statements using eval()
        '''
        x_orbit = orbit_.from_numpy_array(x, parameter_constraints=parameter_constraints)
        n_params = x_orbit.state_vector().size - x_orbit.state.size
        # Root requires input shape = output shape.
        return np.concatenate((x_orbit.spatiotemporal_mapping().state.ravel(), np.zeros(n_params)), axis=0)


    def _cost_function_jac_scipy_root(x):
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

        x_orbit = orbit_.from_numpy_array(x, parameter_constraints=parameter_constraints)
        return np.concatenate((x_orbit.jacobian(), np.zeros([x_orbit.state_vector().size - x_orbit.state.size,
                                                             x_orbit.state_vector().size])), axis=0)

    result = root(_cost_function_scipy_root, orbit_.state_vector(),
                  method=method, jac=_cost_function_jac_scipy_root, tol=tol,
                  callback=callback, options=options)
    result_orbit = orbit_.from_numpy_array(result.x)
    return result_orbit, 0, result.success


def _scipy_newton_krylov_wrapper(orbit_, **kwargs):
    """

    Parameters
    ----------
    orbit
    kwargs

    Returns
    -------

    Notes
    -----
    There are advantages and disadvantages of using this over scipy.optimize.root(method='krylov'). The biggest
    disadvantage (in my eyes) is the inability to pass *args to the solver. This means that the cost function cannot
    be written using instance methods + conversion to and from numpy arrays. Instead, the cost function will be returned
    by another wrapper which can be passed the instance.

    """
    nk_kwarg_list = ['iter', 'rdiff', 'method', 'inner_maxiter', 'inner_M', 'outer_k',
                     'verbose', 'maxiter', 'f_tol', 'f_rtol', 'x_tol', 'x_rtol',
                     'tol_norm', 'line_search', 'callback']

    nkkwargs = {nk_key: kwargs.get(nk_key, None) for nk_key in nk_kwarg_list}

    def _cost_func_newton_krylov(x):
        # _state_vector_to_orbit turns state vector into class instance of the same type as orbit_.
        x_orbit = orbit_.from_numpy_array(orbit_, x, **kwargs)
        # Root requires input shape = output shape.
        n_params = orbit_.state_vector().size - orbit_.state.ravel().size
        return np.concatenate((x_orbit.spatiotemporal_mapping().state.ravel(), np.zeros(n_params)))

    scipy_optimize_result = newton_krylov(_cost_func_newton_krylov, orbit_.state_vector(), **nkkwargs)
    result_orbit = orbit_.from_numpy_array(scipy_optimize_result.x)
    return result_orbit, 0, scipy_optimize_result.success


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
