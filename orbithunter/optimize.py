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


def converge(orbit_, **kwargs):
    """ Main optimization function for orbithunter

    Parameters
    ----------
    orbit_
    method
    kwargs

    Returns
    -------

    Notes
    -----
    Passing orbit_tol and orbit_maxiter in conjuction with method=='hybrid' will cause the _gradient_descent and
    _lstsq to have the same tolerance and number of steps, which is not optimal. Therefore it is recommended to
    either use the keyword arguments 'precision' and 'max_iter' to get default templates, or call converge twice,
    once with method == 'gradient_descent' and once more with method == 'lstsq', passing unique
    orbit_tol and orbit_maxiter to each call.
    """
    orbit_.convert(to='modes', inplace=True)
    method = kwargs.get('method', 'gradient_descent')
    if not orbit_.residual() < kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs)):
        if method == 'hybrid':
            kwargs.pop('orbit_maxiter', None)
            kwargs.pop('orbit_tol', None)
            # There is often a desire to have different tolerances,
            descent_tol, lstsq_tol = kwargs.get('hybrid_tol', (_default_orbit_tol(orbit_, **kwargs),
                                                               _default_orbit_tol(orbit_, **kwargs)))
            descent_iter, lstsq_iter = kwargs.get('hybrid_maxiter', (_default_orbit_maxiter(orbit_, 'gradient_descent'),
                                                                     _default_orbit_maxiter(orbit_, 'lstsq')))
            if descent_tol is None:
                descent_tol = _default_orbit_tol(orbit_, **kwargs)
            if lstsq_tol is None:
                lstsq_tol = _default_orbit_tol(orbit_, **kwargs)
            gradient_orbit, n_iter_a, _ = _gradient_descent(orbit_,  orbit_tol=descent_tol,
                                                            orbit_maxiter=descent_iter, **kwargs)
            result_orbit, n_iter_gn, exit_code = _lstsq(gradient_orbit,  orbit_tol=lstsq_tol,
                                                        orbit_maxiter=lstsq_iter, **kwargs)
            n_iter = (n_iter_a, n_iter_gn)
        elif method == 'gradient_descent':
            result_orbit, n_iter, exit_code = _gradient_descent(orbit_, **kwargs)
        elif method == 'lstsq':
            result_orbit, n_iter, exit_code = _lstsq(orbit_, **kwargs)
        elif method in ['lsqr', 'lsmr']:
            result_orbit, n_iter, exit_code = _scipy_sparse_linalg_solver_wrapper(orbit_, **kwargs)
        elif method in ['cg', 'newton-cg', 'l-bfgs-b', 'tnc']:
            result_orbit, n_iter, exit_code = _scipy_optimize_minimize_wrapper(orbit_,  **kwargs)
        elif method in ['lm', 'lgmres', 'gmres', 'minres']:
            result_orbit, n_iter, exit_code = _scipy_optimize_root_wrapper(orbit_, **kwargs)
        else:
            raise ValueError('Unknown or unsupported solver %s' % method)

        if kwargs.get('verbose', False):
            _print_exit_messages(result_orbit, exit_code)

        return OrbitResult(orbit=result_orbit, exit_code=exit_code, n_iter=n_iter)
    else:
        return OrbitResult(orbit=orbit_, exit_code=1, n_iter=0)


def _gradient_descent(orbit_, **kwargs):
    """
    Parameters
    ----------
    orbit_
    preconditioning
    kwargs

    Returns
    -------

    Notes
    -----
    Preconditioning left out of **kwargs because of its special usage

    """
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, 'gradient_descent'))
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-13)
    verbose = kwargs.get('verbose', False)

    n_iter = 0
    step_size = 1
    # By default assume failure
    exit_code = 0

    if verbose:
        print('Starting gradient descent. Initial residual={}, target={}, max_iter={}'.format(orbit_.residual(),
                                                                                              orbit_tol,
                                                                                              orbit_maxiter))
    mapping = orbit_.spatiotemporal_mapping(**kwargs)
    residual = mapping.residual(apply_mapping=False)
    while residual > orbit_tol:
        # Calculate the step
        gradient = orbit_.cost_function_gradient(mapping, **kwargs)
        # Apply the step
        next_orbit = orbit_.increment(gradient, step_size=-1.0*step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.spatiotemporal_mapping(**kwargs)
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(apply_mapping=False)
        while next_residual >= residual and step_size > 10**-8:
            # reduce the step size until minimum is reached or residual decreases.
            step_size /= 2
            next_orbit = orbit_.increment(gradient, step_size=-1.0*step_size)
            next_mapping = next_orbit.spatiotemporal_mapping(**kwargs)
            next_residual = next_mapping.residual(apply_mapping=False)

        if next_residual <= orbit_tol:
            return next_orbit, n_iter, 1
        elif step_size <= 10**-9 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            return orbit_, n_iter, 0
        elif n_iter > orbit_maxiter:
            return next_orbit, n_iter, 2
        else:
            # Update and restart loop if residual successfully decreases.
            orbit_, mapping, residual = next_orbit, next_mapping, next_residual
            n_iter += 1
            if verbose:
                if np.mod(n_iter, 5000) == 0:
                    print('\n Residual={:.7f} after {} gradient descent steps. Parameters:{}'.format(
                        orbit_.residual(), n_iter, orbit_.parameters))
                elif np.mod(n_iter, 100) == 0:
                    print('#', end='')

    return orbit_, n_iter, 1


def _lstsq(orbit_, **kwargs):
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, **kwargs))
    verbose = kwargs.get('verbose', False)
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-13)
    n_iter = 0
    residual = orbit_.residual()

    if verbose:
        print('\nStarting lstsq. Initial residual={}, target={}'.format(orbit_.residual(), orbit_tol))

    while residual > orbit_tol:
        step_size = 1
        n_iter += 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * orbit_.spatiotemporal_mapping(**kwargs).state.ravel()
        dx = orbit_.from_numpy_array(lstsq(A, b.reshape(-1, 1))[0], **kwargs)

        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_residual = next_orbit.residual()
        while next_residual > residual and step_size > 10**-8:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_residual = next_orbit.residual()

        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        if next_residual <= orbit_tol:
            return next_orbit, n_iter, 1
        elif step_size <= 10**-8 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            return next_orbit, n_iter, 0
        elif n_iter >= orbit_maxiter:
            return orbit_, n_iter, 2
        else:
            orbit_ = next_orbit
            residual = next_residual

        if kwargs.get('verbose', False):
            print('#', end='')
            if np.mod(n_iter, max([1, (orbit_maxiter // 10)])) == 0:
                print(' Residual={:.7f} after {} {} iterations'.format(orbit_.residual(), n_iter, 'lstsq'))
            sys.stdout.flush()
    return orbit_, n_iter, 1


def _scipy_sparse_linalg_solver_wrapper(orbit_, damp=0.0, atol=1e-6, btol=1e-6,
                                        method='lsqr', maxiter=None, conlim=1e+08,
                                        show=False, calc_var=False, **kwargs):

    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, method, **kwargs))
    max_damp_factor = kwargs.get('orbit_damp_max', 8)
    preconditioning = kwargs.get('preconditioning', True)

    linear_operator_shape = (orbit_.state.size, orbit_.state_vector().size)
    istop = 1
    n_iter = 0
    orbit_n_iter = 0
    # Return codes that represent good results from the SciPy least-squares solvers.
    good_codes = [0, 1, 2, 4, 5]
    residual = orbit_.residual()

    while (residual > orbit_tol) and (istop in good_codes) and orbit_n_iter < orbit_maxiter:
        orbit_n_iter += 1

        # The operator depends on the current state; A=A(orbit)
        def rmv_func(v):
            # _process_newton_step turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v, parameters=orbit_.parameters)
            if preconditioning:
                # To be consistent, if left preconditioning in matvec, need to right precondition rmatvec
                # This means that only the modes should be preconditioned; the work around for this is to constrain
                # the parameters, as v_orbit is just a placeholder anyway.
                v_orbit.constraints = {k: True for k in orbit_.constraints.keys()}
                return orbit_.rmatvec(v_orbit.precondition(orbit_.preconditioning_parameters),
                                      **kwargs).state_vector().reshape(-1, 1)
            else:
                return orbit_.rmatvec(v_orbit, **kwargs).state_vector().reshape(-1, 1)

        def mv_func(v):
            # _state_vector_to_orbit turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v)
            if preconditioning:
                return orbit_.matvec(v_orbit, **kwargs).precondition(orbit_.preconditioning_parameters).state.reshape(-1, 1)
            else:
                return orbit_.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

        orbit_linear_operator = LinearOperator(linear_operator_shape, mv_func, rmatvec=rmv_func, dtype=float)
        b = -1.0 * orbit_.spatiotemporal_mapping(**kwargs).state.reshape(-1, 1)
        damp_factor = 0

        if method == 'lsmr':
            result_tuple = lsmr(orbit_linear_operator, b, damp=damp, atol=atol, btol=btol,
                                conlim=conlim, maxiter=maxiter, show=show)
        elif method == 'lsqr':
            # Depends heavily on scaling of the problem.
            result_tuple = lsqr(orbit_linear_operator, b, damp=damp, atol=atol, btol=btol, conlim=conlim,
                                iter_lim=maxiter, show=show, calc_var=calc_var)
        else:
            raise ValueError('Unknown solver %s' % method)

        x = result_tuple[0]
        istop = result_tuple[1]
        n_iter += result_tuple[2]

        dorbit = orbit_.from_numpy_array(x, **kwargs)
        next_orbit = orbit_.increment(dorbit)
        next_residual = next_orbit.residual()
        while next_residual > residual:
            damp_factor += 1
            next_orbit = orbit_.increment(dorbit, step_size=2 ** -damp_factor)
            next_residual = next_orbit.residual()
            if damp_factor > max_damp_factor:
                return orbit_, n_iter, 0
        else:
            print(next_residual)
            orbit_ = next_orbit
            residual = next_residual
            if kwargs.get('verbose', False):
                # print(damp_factor, end='')
                if np.mod(orbit_n_iter, (orbit_maxiter // 10)) == 0:
                    print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))

    if n_iter == orbit_maxiter:
        exit_code = 2
    else:
        exit_code = 0

    return orbit_, n_iter, exit_code


def _scipy_optimize_minimize_wrapper(orbit_, method=None, bounds=None,
                                     tol=None, callback=None, options=None, **kwargs):

    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, method, **kwargs))
    verbose = kwargs.get('verbose', False)

    # The performance of the different methods depends on preconditioning differently/
    if method in ['newton-cg', 'tnc']:
        # This is a work-around to have different defaults for the different methods.
        preconditioning = kwargs.get('preconditioning', False)
    elif method == 'l-bfgs-b':
        # This is a work-around to have different defaults for the different methods.
        preconditioning = kwargs.get('preconditioning', True)
    elif method == 'cg':
        # This is a work-around to have different defaults for the different methods.
        preconditioning = kwargs.get('preconditioning', True)
        if options is None:
            options = {'gtol': 1e-3}
        elif isinstance(options, dict):
            options['gtol'] = options.get('gtol', 1e-3)

    def _cost_function_scipy_minimize(x):
        '''
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        '''

        '''
        Note that passing Class as a function avoids dangerous statements using eval()
        '''
        x_orbit = orbit_.from_numpy_array(x, **kwargs)
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
        if preconditioning:
            return (x_orbit.rmatvec(x_orbit.spatiotemporal_mapping(), **kwargs)
                    ).precondition(x_orbit.preconditioning_parameters).state_vector().ravel()
        else:
            return x_orbit.rmatvec(x_orbit.spatiotemporal_mapping(), **kwargs).state_vector().ravel()

    orbit_n_iter = 0
    success = True

    while (orbit_.residual() > orbit_tol) and (orbit_n_iter < orbit_maxiter) and success:
        orbit_n_iter += 1
        result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                          method=method, jac=_cost_function_jac_scipy_minimize, bounds=bounds, tol=tol,
                          callback=callback, options=options)
        orbit_ = orbit_.from_numpy_array(result.x)
        success = result.success
        if verbose:
            if np.mod(orbit_n_iter, orbit_maxiter // 10) == 0:
                print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))

    return orbit_, orbit_n_iter, success


def _scipy_optimize_root_wrapper(orbit_, method=None, tol=None, callback=None, options=None, **kwargs):
    """ Wrapper for scipy.optimize.root methods

    Parameters
    ----------
    orbit_
    method
    tol
    callback
    options
    kwargs

    Returns
    -------

    Notes
    -----
    Does not allow for preconditioning currently. Only supports the following methods: 'lm', 'lgmres', 'gmres', 'minres'

    """
    # define the functions using orbit instance within scope instead of passing orbit
    # instance as arg to scipy functions.

    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_,  **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, method, **kwargs))
    verbose = kwargs.get('verbose', False)

    def _cost_function_scipy_root(x):
        '''
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        '''

        '''
        Note that passing Class as a function avoids dangerous statements using eval()
        '''
        x_orbit = orbit_.from_numpy_array(x, **kwargs)
        n_params = x_orbit.state_vector().size - x_orbit.state.size
        # Root requires input shape = output shape.
        return np.concatenate((x_orbit.spatiotemporal_mapping(**kwargs).state.ravel(),
                               np.zeros(n_params)), axis=0)

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

        x_orbit = orbit_.from_numpy_array(x, **kwargs)
        n_params = x_orbit.state_vector().size - x_orbit.state.size
        return np.concatenate((x_orbit.jacobian(**kwargs),
                               np.zeros([n_params, x_orbit.state_vector().size])), axis=0)

    # If not providing jacobian numerical approximation is used which can be beneficial.
    if method == 'lm' and kwargs.get('jacobian_on', True):
        jac_ = _cost_function_jac_scipy_root
    elif method in ['lgmres', 'gmres', 'minres']:
        jac_ = None
        # These methods are actually inner loop methods of 'krylov' method; need to be passed into options dict.
        #
        if options is None:
            options = {'jac_options': {'method': method}}
        elif isinstance(options, dict):
            options['jac_options'] = {'method': method}
        method = 'krylov'
    else:
        jac_ = None

    orbit_n_iter = 0
    success = True

    while (orbit_.residual() > orbit_tol) and (orbit_n_iter < orbit_maxiter) and success:
        orbit_n_iter += 1
        result = root(_cost_function_scipy_root, orbit_.state_vector(),
                      method=method, jac=jac_, tol=tol,
                      callback=callback, options=options)
        orbit_ = orbit_.from_numpy_array(result.x, **kwargs)
        success = result.success
        if verbose:
            if np.mod(orbit_n_iter, (kwargs.get('orbit_maxiter', 20) // 10)) == 0:
                print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))
    return orbit_, orbit_n_iter, success


def _print_exit_messages(orbit, exit_code):
    if exit_code == 0:
        print('\nInsufficient residual decrease. Exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 1:
        print('\nTolerance threshold met. Exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 2:
        print('\nFailed to converge. Maximum number of iterations reached.'
              ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 3:
        print('\nConverged to an equilibrium'
              ' exiting with residual {}'.format(orbit.residual()))
    elif exit_code == 4:
        print('\nConverged to the trivial u(x,t)=0 solution')
    elif exit_code == 5:
        print('\n Relative periodic orbit converged to periodic orbit with essentially zero shift.')
    elif exit_code == 6:
        print('\n Relative periodic orbit has lower residual when flipping sign of shift.'
              ' Look for a negative sign error')
    return None


def _default_orbit_tol(orbit_, **kwargs):
    """ Wrapper to allow str keyword argument to be used to choose orbit tolerance dependent on method

    Parameters
    ----------
    orbit_
    kwargs

    Returns
    -------

    Notes
    -----
    The purposes of this function are to not have conflicting types for orbit_tol kwarg and improved convenience
    in specifying the default tolerance per method.

    """
    precision_level = kwargs.get('precision', 'default')
    # Introduction of ugly conditional statements for convenience
    if precision_level == 'machine':
        default_tol = np.product(orbit_.field_shape) * 10**-15
    elif precision_level == 'high':
        default_tol = np.product(orbit_.field_shape) * 10**-12
    elif precision_level == 'medium' or precision_level == 'default':
        default_tol = np.product(orbit_.field_shape) * 10**-9
    elif precision_level == 'low':
        default_tol = np.product(orbit_.field_shape) * 10**-6
    elif precision_level == 'minimal':
        default_tol = np.product(orbit_.field_shape) * 10**-3
    else:
        raise ValueError('If a custom tolerance is desired, use ''orbit_tol'' key word instead.')

    return default_tol


def _default_orbit_maxiter(orbit_, method, **kwargs):
    """ Wrapper to allow str keyword argument to be used to choose number of iterations dependent on method

    Parameters
    ----------
    orbit_
    kwargs

    Returns
    -------

    Notes
    -----
    The purposes of this function are to not have conflicting types for orbit_tol kwarg and improved convenience
    in specifying the default tolerance per method.

    """
    max_iter = kwargs.get('max_iter', 'default')
    if method in ['gradient_descent', 'cg', 'newton-cg', 'l-bfgs-b', 'tnc', 'lgmres', 'gmres', 'minres', 'krylov']:
        # Introduction of ugly conditional statements for convenience
        if max_iter == 'long':
            default_max_iter = 1024 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif max_iter == 'medium' or max_iter == 'default':
            default_max_iter = 512 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif max_iter == 'short':
            default_max_iter = 128 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif max_iter == 'minimal':
            default_max_iter = 32 * int(np.sqrt(np.product(orbit_.field_shape)))
        else:
            raise ValueError('If a custom number of iterations is desired, use ''orbit_maxiter'' key word instead.')
    else:
        # Introduction of ugly conditional statements for convenience
        if max_iter == 'long':
            default_max_iter = 500
        elif max_iter == 'medium' or max_iter == 'default':
            default_max_iter = 250
        elif max_iter == 'short':
            default_max_iter = 50
        elif max_iter == 'minimal':
            default_max_iter = 10
        else:
            raise ValueError('If a custom number of iterations is desired, use ''orbit_maxiter'' key word instead.')
    return default_max_iter

