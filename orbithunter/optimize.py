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


def converge(orbit_, method='hybrid', **kwargs):
    orbit_.convert(to='modes', inplace=True)
    if not (orbit_.residual() < kwargs.get('orbit_tol', orbit_.M * orbit_.N * 10**-6)):

        if kwargs.get('verbose', False):
            print('Starting {} numerical method. Initial residual {}'.format(method, orbit_.residual()))

        if method == 'hybrid':
            adj_maxiter, newton_maxiter = kwargs.pop('orbit_maxiter', (16 * orbit_.N * orbit_.M, 500))
            gradient_orbit, n_iter_a, _ = _gradient_descent(orbit_, orbit_maxiter=adj_maxiter, **kwargs)
            result_orbit, n_iter_gn, exit_code = _gauss_newton(gradient_orbit, orbit_maxiter=newton_maxiter, **kwargs)
            n_iter = (n_iter_a, n_iter_gn)
        elif method in ['grad', 'gradient', 'gd', 'steepest']:
            result_orbit, n_iter, exit_code = _gradient_descent(orbit_, **kwargs)
        elif method == 'lstsq':
            result_orbit, n_iter, exit_code = _gauss_newton(orbit_, **kwargs)
        elif method in ['lsqr', 'lsmr']:
            result_orbit, n_iter, exit_code = _scipy_sparse_linalg_solver_wrapper(orbit_, method=method, **kwargs)
        elif method in ['cg', 'newton-cg', 'l-bfgs-b', 'tnc']:
            result_orbit, n_iter, exit_code = _scipy_optimize_minimize_wrapper(orbit_, method=method, **kwargs)
        elif method in ['lm', 'lgmres', 'gmres', 'minres']:
            result_orbit, n_iter, exit_code = _scipy_optimize_root_wrapper(orbit_, method=method, **kwargs)
        else:
            raise ValueError('Unknown or unsupported solver %s' % method)

        if kwargs.get('verbose', False):
            print_exit_messages(result_orbit, exit_code)

        return OrbitResult(orbit=result_orbit, exit_code=exit_code, n_iter=n_iter)
    else:
        return OrbitResult(orbit=orbit_, exit_code=1, n_iter=0)


def _gradient_descent(orbit_, preconditioning=True, ftol=1e-8, **orbithunter_kwargs):
    """
    Parameters
    ----------
    orbit_
    preconditioning
    orbithunter_kwargs

    Returns
    -------

    Notes
    -----
    Preconditioning left out of **orbithunter_kwargs because of its special usage

    """

    # Specific modifying exponent for changes in period, domain_size
    # Absolute tolerance for the descent method.
    tol = orbithunter_kwargs.get('orbit_tol', orbit_.M * orbit_.N * 10**-6)
    maxiter = orbithunter_kwargs.get('orbit_maxiter', 32 * orbit_.N * orbit_.M)

    step_size = 1
    n_iter = 0
    # By default assume failure
    exit_code = 0

    mapping = orbit_.spatiotemporal_mapping(**orbithunter_kwargs)
    residual = mapping.residual(apply_mapping=False)
    while residual > tol and n_iter < maxiter:
        # Calculate the step
        if preconditioning:
            dx = orbit_.rmatvec(mapping, **orbithunter_kwargs).precondition(orbit_.parameters)
        else:
            dx = orbit_.rmatvec(mapping, **orbithunter_kwargs)

        # Apply the step
        next_orbit = orbit_.increment(dx, stepsize=-1.0 * step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.spatiotemporal_mapping(**orbithunter_kwargs)
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(apply_mapping=False)

        while next_residual >= residual:
            # reduce the step size until minimum is reached or residual decreases.
            step_size = 0.5 * step_size
            next_orbit = orbit_.increment(dx, stepsize=-1.0 * step_size)
            next_mapping = next_orbit.spatiotemporal_mapping(**orbithunter_kwargs)
            next_residual = next_mapping.residual(apply_mapping=False)
            if step_size <= 10 ** -8:
                return orbit_, n_iter, exit_code
        else:
            fval = (residual - next_residual) / max([residual, next_residual, 1])
            if fval < ftol:
                return orbit_, n_iter, exit_code
            # Update and restart loop if residual successfully decreases.
            orbit_, mapping, residual = next_orbit, next_mapping, next_residual
            n_iter += 1

            if orbithunter_kwargs.get('verbose', False):

                if np.mod(n_iter, (maxiter // 10)) == 0:
                    print(' Residual={} after {} {} iterations'.format(orbit_.residual(), n_iter, 'gradient descent'))
                elif np.mod(n_iter, (maxiter // 100)) == 0:
                    print('#', end='')

    if orbit_.residual() <= tol:
        orbit, exit_code = orbit_.status()
    elif n_iter == maxiter:
        exit_code = 2

    return orbit_, n_iter, exit_code


def _gauss_newton(orbit_, orbit_maxiter=500, max_damp_factor=9, **orbithunter_kwargs):
    tol = orbithunter_kwargs.get('orbit_tol', orbit_.N * orbit_.M * 10 ** -6)
    n_iter = 0
    residual = orbit_.residual()

    while residual > tol and n_iter < orbit_maxiter:
        damp_factor = 0
        n_iter += 1
        A = orbit_.jacobian(**orbithunter_kwargs)
        b = -1.0 * orbit_.spatiotemporal_mapping(**orbithunter_kwargs).state.ravel()
        dorbit = orbit_.from_numpy_array(lstsq(A, b.reshape(-1, 1))[0], **orbithunter_kwargs)

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
                return orbit_, n_iter, 0
        else:
            # Executed when step decreases residual and is not too short
            orbit_ = next_orbit
            residual = next_residual

            if orbithunter_kwargs.get('verbose', False):
                print(damp_factor, end='')
                if np.mod(n_iter, (orbit_maxiter // 10)) == 0:
                    print('Residual={} after {} {} iterations'.format(orbit_.residual(), n_iter, 'lstsq'))
                sys.stdout.flush()

    if orbit_.residual() <= tol:
        orbit, exit_code = orbit_.status()
    elif n_iter == orbit_maxiter:
        exit_code = 2
    else:
        exit_code = 0

    return orbit_, n_iter, exit_code


def _scipy_sparse_linalg_solver_wrapper(orbit_, damp=0.0, atol=1e-03, btol=1e-03,
                                        method='lsqr', maxiter=None, conlim=1e+08,
                                        show=False, calc_var=False, **orbithunter_kwargs):

    linear_operator_shape = (orbit_.state.size, orbit_.state_vector().size)
    istop = 1
    n_iter = 0
    orbit_n_iter = 0
    # Return codes that represent good results from the SciPy least-squares solvers.
    good_codes = [0, 1, 2, 4, 5]
    residual = orbit_.residual()

    orbit_tol = orbithunter_kwargs.get('orbit_tol', orbit_.M * orbit_.N * 10**-6)
    orbit_maxiter = orbithunter_kwargs.get('orbit_maxiter', 250)
    max_damp_factor = orbithunter_kwargs.get('orbit_damp_max', 8)
    preconditioning = orbithunter_kwargs.get('preconditioning', False)

    while (residual > orbit_tol) and (istop in good_codes) and orbit_n_iter < orbit_maxiter:
        orbit_n_iter += 1

        # The operator depends on the current state; A=A(orbit)
        def rmv_func(v):
            # _process_newton_step turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v, **orbithunter_kwargs)
            v_orbit.T, v_orbit.L, v_orbit.S = orbit_.T, orbit_.L, orbit_.S
            rmatvec_orbit = orbit_.rmatvec(v_orbit, **orbithunter_kwargs)
            if preconditioning:
                return rmatvec_orbit.precondition(orbit_.parameters).state_vector().reshape(-1, 1)
            else:
                return rmatvec_orbit.state_vector().reshape(-1, 1)

        def mv_func(v):
            # _state_vector_to_orbit turns state vector into class object.
            v_orbit = orbit_.from_numpy_array(v, **orbithunter_kwargs)
            v_orbit.T, v_orbit.L, v_orbit.S = orbit_.T, orbit_.L, orbit_.S
            matvec_orbit = orbit_.matvec(v_orbit, **orbithunter_kwargs)
            if preconditioning:
                return matvec_orbit.precondition(orbit_.parameters).state.reshape(-1, 1)
            else:
                return matvec_orbit.state.reshape(-1, 1)

        orbit_linear_operator = LinearOperator(linear_operator_shape, mv_func, rmatvec=rmv_func, dtype=float)
        b = -1.0 * orbit_.spatiotemporal_mapping(**orbithunter_kwargs).state.reshape(-1, 1)
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

        dorbit = orbit_.from_numpy_array(x, **orbithunter_kwargs)
        next_orbit = orbit_.increment(dorbit)
        next_residual = next_orbit.residual()
        while next_residual > residual:
            damp_factor += 1
            next_orbit = orbit_.increment(dorbit, stepsize=2 ** -damp_factor)
            next_residual = next_orbit.residual()
            if damp_factor > max_damp_factor:
                return orbit_, n_iter, 0
        else:
            orbit_ = next_orbit
            residual = next_residual
            if orbithunter_kwargs.get('verbose', False):
                if np.mod(orbit_n_iter, (orbit_maxiter // 10)) == 0:
                    print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))

    if orbit_.residual() <= orbit_tol:
        orbit_, exit_code = orbit_.status()
    elif n_iter == orbit_maxiter:
        exit_code = 2
    else:
        exit_code = 0

    return orbit_, n_iter, exit_code


def _scipy_optimize_minimize_wrapper(orbit_, method=None, bounds=None,
                                     tol=None, callback=None, options=None, **orbithunter_kwargs):

    # The performance of the different methods depends on preconditioning differently/
    if method in ['newton-cg', 'tnc']:
        # This is a work-around to have different defaults for the different methods.
        preconditioning = orbithunter_kwargs.get('preconditioning', False)
    elif method =='l-bfgs-b':
        # This is a work-around to have different defaults for the different methods.
        preconditioning = orbithunter_kwargs.get('preconditioning', True)
    elif method =='cg':
        # This is a work-around to have different defaults for the different methods.
        preconditioning = orbithunter_kwargs.get('preconditioning', True)
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
        x_orbit = orbit_.from_numpy_array(x, **orbithunter_kwargs)
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
            return (x_orbit.rmatvec(x_orbit.spatiotemporal_mapping(), **orbithunter_kwargs)
                    ).precondition(x_orbit.parameters).state_vector().ravel()
        else:
            return x_orbit.rmatvec(x_orbit.spatiotemporal_mapping(), **orbithunter_kwargs).state_vector().ravel()

    orbit_n_iter = 0
    success = True
    while ((orbit_.residual() > orbithunter_kwargs.get('orbit_tol', orbit_.M * orbit_.N * 10**-6))
           and (orbit_n_iter < orbithunter_kwargs.get('orbit_maxiter', 20))
           and success):
        orbit_n_iter += 1
        result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                          method=method, jac=_cost_function_jac_scipy_minimize, bounds=bounds, tol=tol,
                          callback=callback, options=options)
        orbit_ = orbit_.from_numpy_array(result.x)
        success = result.success
        if orbithunter_kwargs.get('verbose', False):
            if np.mod(orbit_n_iter, (orbithunter_kwargs.get('orbit_maxiter', 20) // 10)) == 0:
                print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))

    return orbit_, orbit_n_iter, success


def _scipy_optimize_root_wrapper(orbit_, method=None, tol=None, callback=None, options=None, **orbithunter_kwargs):
    """ Wrapper for scipy.optimize.root methods

    Parameters
    ----------
    orbit_
    method
    tol
    callback
    options
    orbithunter_kwargs

    Returns
    -------

    Notes
    -----
    Does not allow for preconditioning currently. Only supports the following methods: 'lm', 'lgmres', 'gmres', 'minres'

    """
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
        x_orbit = orbit_.from_numpy_array(x, **orbithunter_kwargs)
        n_params = x_orbit.state_vector().size - x_orbit.state.size
        # Root requires input shape = output shape.
        return np.concatenate((x_orbit.spatiotemporal_mapping(**orbithunter_kwargs).state.ravel(),
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

        x_orbit = orbit_.from_numpy_array(x, **orbithunter_kwargs)
        n_params = x_orbit.state_vector().size - x_orbit.state.size
        return np.concatenate((x_orbit.jacobian(**orbithunter_kwargs),
                               np.zeros([n_params, x_orbit.state_vector().size])), axis=0)

    # If not providing jacobian numerical approximation is used which can be beneficial.
    if method == 'lm' and orbithunter_kwargs.get('jacobian_on', True):
        jac_ = _cost_function_jac_scipy_root
    elif method in ['lgmres', 'gmres', 'minres']:
        jac_ = None
        # These methods are actually inner loop methods of 'krylov' method; need to be passed into options dict.
        #
        if options is None:
            options = {'jac_options': {'method': method}}
            method = 'krylov'
        elif isinstance(options, dict):
            options['method'] = method
            options['jac_options'] = {'method': method}
    else:
        jac_ = None

    orbit_n_iter = 0
    success = True
    while ((orbit_.residual() > orbithunter_kwargs.get('orbit_tol', orbit_.M * orbit_.N * 10**-6))
           and (orbit_n_iter < orbithunter_kwargs.get('orbit_maxiter', 20))
           and success):
        orbit_n_iter += 1
        result = root(_cost_function_scipy_root, orbit_.state_vector(),
                      method=method, jac=jac_, tol=tol,
                      callback=callback, options=options)
        orbit_ = orbit_.from_numpy_array(result.x, **orbithunter_kwargs)
        success = result.success
        if orbithunter_kwargs.get('verbose', False):
            if np.mod(orbit_n_iter, (orbithunter_kwargs.get('orbit_maxiter', 20) // 10)) == 0:
                print('Residual={} after {} {} iterations'.format(orbit_.residual(), orbit_n_iter, method))
    return orbit_, orbit_n_iter, success


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
