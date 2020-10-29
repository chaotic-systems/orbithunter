from scipy.linalg import lstsq, pinv
from scipy.optimize import minimize
from scipy.sparse.linalg import (LinearOperator, bicg, bicgstab, gmres, lgmres,
                                 cg, cgs, qmr, minres, lsqr, lsmr, gcrotmk)
import sys
import numpy as np

__all__ = ['converge']


class OrbitResult(dict):
    """ Represents the result of applying converge; format copied from SciPy scipy.optimize.OptimizeResult.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    status : int
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

    The descriptions for each value of status are as follows. In order, the codes [0,1,2,3,4,5] == 0:
    0 : Failed to converge
    1 : Converged
    2:
        print('\nFailed to converge. Maximum number of iterations reached.'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif status == 3:
        print('\nConverged to an errant equilibrium'
                     ' exiting with residual {}'.format(orbit.residual()))
    elif status == 4:
        print('\nConverged to the trivial u(x,t)=0 solution')
    elif status == 5:
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

    kwargs:
        method :

        orbit_maxiter :

        orbit_tol :

        hybrid_maxiter : tuple of two ints
            Only used if method == 'hybrid', contains the maximum number of iterations to be used in gradient
            descent and lstsq, respectively.

        hybrid_tol : tuple of floats
            Only used if method == 'hybrid', contains the tolerance threshold to be used in gradient
            descent and lstsq, respectively.

        scipy.optimize.minimize kwargs:
            There are too many to describe and they depend on the particular algorithm utilized, see scipy
            docs for more info. These pertain to numerical methods in ['cg', 'newton-cg', 'l-bfgs-b', 'tnc']

        scipy.sparse.linalg kwargs
            Additional arguments for scipy.sparse.linalg solvers method in ['lsqr', 'lsmr']

        scipy.optimize.root kwargs:
            Additional arguments for scipy.optimize.minimize solvers in ['lm', 'lgmres', 'gmres', 'minres'].
            'lm' should not be used for large problems.


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
    # Convert basis, creating a copy in the process.
    orbit_ = orbit_.convert(to='modes')
    if not orbit_.residual() < kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs)):
        if kwargs.get('method', '') == 'hybrid':
            # make sure that extraneous parameters are removed.
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
            gradient_orbit, gd_stats = _gradient_descent(orbit_, orbit_tol=descent_tol,
                                                         orbit_maxiter=descent_iter, **kwargs)
            result_orbit, lstsq_stats = _lstsq(gradient_orbit,  orbit_tol=lstsq_tol,
                                               orbit_maxiter=lstsq_iter, **kwargs)
            statistics = {key: (gd_stats[key], lstsq_stats[key])
                          for key in sorted({**gd_stats, **lstsq_stats}.keys())}
        elif kwargs.get('method', '') == 'lstsq':
            # solves Ax = b in least-squares manner
            result_orbit, statistics = _lstsq(orbit_, **kwargs)
        elif kwargs.get('method', '') in ['lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
                                          'cg', 'cgs', 'qmr', 'minres', 'gcrotmk']:
            # solves A^T A x = A^T b
            result_orbit, statistics = _scipy_sparse_linalg_solver_wrapper(orbit_, **kwargs)
        elif kwargs.get('method', '') in ['cg_min', 'newton-cg', 'l-bfgs-b', 'tnc', 'bfgs']:
            # minimizes cost functional 1/2 F^2
            if kwargs.get('method', '') == 'cg_min':
                kwargs['method'] = 'cg'
            result_orbit, statistics = _scipy_optimize_minimize_wrapper(orbit_,  **kwargs)
        else:
            result_orbit, statistics = _gradient_descent(orbit_, **kwargs)

        if kwargs.get('verbose', False):
            _print_exit_messages(result_orbit, statistics['status'])

        return OrbitResult(orbit=result_orbit, **statistics)
    else:
        return OrbitResult(orbit=orbit_, status=-1)


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

    if verbose:
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting gradient descent')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(orbit_tol))
        print('Maximum iteration number : {}'.format(orbit_maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')

    mapping = orbit_.spatiotemporal_mapping(**kwargs)
    residual = mapping.residual(apply_mapping=False)
    step_size = 1
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': orbit_maxiter, 'tol': orbit_tol, 'status':1}
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
        stats['nit'] += 1
        if step_size <= 10**-8 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            stats['status'] = 0
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        elif stats['nit'] > orbit_maxiter:
            stats['status'] = 2
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        else:

            # Update and restart loop if residual successfully decreases.
            if kwargs.get('log_residual', False):
                stats['residuals'].append(next_residual)
            orbit_, mapping, residual = next_orbit, next_mapping, next_residual
            if verbose:
                if np.mod(stats['nit'], 5000) == 0:
                    print('\n Residual={:.7f} after {} gradient descent steps. Parameters:{}'.format(
                        next_residual, stats['nit'], orbit_.parameters))
                elif np.mod(stats['nit'], 100) == 0:
                    print('#', end='')
    else:
        stats['status'] = 1
        stats['residuals'].append(orbit_.residual())
        return orbit_,  stats


def _newton_descent(orbit_, **kwargs):
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, **kwargs))
    step_size=kwargs.get('step_size', 0.001)
    verbose = kwargs.get('verbose', False)
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-13)
    residual = orbit_.residual()
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': orbit_maxiter, 'tol': orbit_tol, 'status': 1}
    if verbose:
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting lstsq optimization')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(orbit_tol))
        print('Maximum iteration number : {}'.format(orbit_maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')

    mapping = orbit_.spatiotemporal_mapping()
    residual = mapping.residual(apply_mapping=False)
    step_size = 0.001
    while residual > orbit_tol:
        # Solve A dx = b <--> J dx = - f, for dx.
        A, b = orbit_.jacobian(), -1*mapping.state.ravel()
        inv_A = pinv(A)
        dx = orbit_.from_numpy_array(inv_A.dot(b))
        # print('####################', dx.norm(), dx.L, dx.T)
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.spatiotemporal_mapping()
        next_residual = next_mapping.residual(apply_mapping=False)
        while next_residual > residual and step_size > 10**-6:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.spatiotemporal_mapping()
            next_residual = next_mapping.residual(apply_mapping=False)
        else:
            stats['nit'] += 1
            if kwargs.get('approximation', True):
                n_inner = 0
                while next_residual < residual:
                    n_inner += 1
                    orbit_ = next_orbit
                    mapping = next_mapping
                    residual = next_residual
                    b = -1 * mapping.state.ravel()
                    dx = orbit_.from_numpy_array(inv_A.dot(b))
                    next_orbit = orbit_.increment(dx, step_size=step_size)
                    next_mapping = next_orbit.spatiotemporal_mapping()
                    next_residual = next_mapping.residual(apply_mapping=False)
            else:
                orbit_ = next_orbit
                residual = next_residual
                mapping = next_orbit.spatiotemporal_mapping()

        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        if step_size <= 10**-5 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            stats['status'] = 0
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        elif stats['nit'] > orbit_maxiter:
            stats['status'] = 2
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        else:
            # Update and restart loop if residual successfully decreases.
            if kwargs.get('log_residual', False):
                stats['residuals'].append(next_residual)

            if kwargs.get('verbose', False):
                print('#', end='')
                if np.mod(stats['nit'], 25) == 0:
                    print(' Residual={:.7f} after {} {} iterations'.format(next_residual, stats['nit'], 'lstsq'))
                sys.stdout.flush()

            # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
            if step_size <= 10**-5 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
                stats['status'] = 0
                stats['residuals'].append(next_residual)
                return orbit_,  stats
            elif stats['nit'] > orbit_maxiter:
                stats['status'] = 2
                stats['residuals'].append(next_residual)
                return orbit_,  stats
            else:
                stats['nit'] += 1
                # Update and restart loop if residual successfully decreases.
                if kwargs.get('log_residual', False):
                    stats['residuals'].append(next_residual)
                if kwargs.get('verbose', False):
                    print('#', end='')
                    if np.mod(stats['nit'], 25) == 0:
                        print(' Residual={:.7f} after {} {} iterations'.format(next_residual, stats['nit'], 'lstsq'))
                    sys.stdout.flush()

    return orbit_, stats


def _lstsq(orbit_, **kwargs):
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, **kwargs))
    verbose = kwargs.get('verbose', False)
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-13)
    residual = orbit_.residual()
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': orbit_maxiter, 'tol': orbit_tol, 'status': 1}
    if verbose:
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting lstsq optimization')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(orbit_tol))
        print('Maximum iteration number : {}'.format(orbit_maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
    while residual > orbit_tol:
        step_size = 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * orbit_.spatiotemporal_mapping(**kwargs).state.ravel()
        dx = orbit_.from_numpy_array(lstsq(A, b.reshape(-1, 1))[0], **kwargs)

        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_residual = next_orbit.residual()
        while next_residual > residual and step_size > 10**-5:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_residual = next_orbit.residual()

        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        stats['nit'] += 1
        if step_size <= 10**-5 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            stats['status'] = 0
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        elif stats['nit'] > orbit_maxiter:
            stats['status'] = 2
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        else:
            # Update and restart loop if residual successfully decreases.
            if kwargs.get('log_residual', False):
                stats['residuals'].append(next_residual)
            orbit_ = next_orbit
            residual = next_residual
            if kwargs.get('verbose', False):
                print('#', end='')
                if np.mod(stats['nit'], 25) == 0:
                    print(' Residual={:.7f} after {} {} iterations'.format(next_residual, stats['nit'], 'lstsq'))
                sys.stdout.flush()
    else:
        stats['residuals'].append(orbit_.residual())
        return orbit_, stats


def _scipy_sparse_linalg_solver_wrapper(orbit_, method='minres', **kwargs):
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    orbit_maxiter = kwargs.get('orbit_maxiter', _default_orbit_maxiter(orbit_, method, **kwargs))
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-13)
    min_step_size = kwargs.get('min_step_size', 10**-8)
    residual = orbit_.residual()
    if kwargs.get('verbose', False):
        print('\n------------------------------------------------------------------------------------------------')
        print('Starting {} optimization'.format(method))
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(orbit_tol))
        print('Maximum iteration number : {}'.format(orbit_maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')

    stats = {'nit': 0, 'residuals': [residual], 'maxiter': orbit_maxiter, 'tol': orbit_tol, 'status': 1}
    while residual > orbit_tol:
        step_size = 1
        if method in ['lsmr', 'lsqr']:
            scipy_kwargs = kwargs.pop('scipy_kwargs', {'atol': 1e-6, 'btol': 1e-6})
            # Solving least-squares equations, A x = b

            def matvec_func(v):
                # _state_vector_to_orbit turns state vector into class object.
                v_orbit = orbit_.from_numpy_array(v)
                return orbit_.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

            def rmatvec_func(v):
                # _state_vector_to_orbit turns state vector into class object.
                v_orbit = orbit_.from_numpy_array(v, parameters=orbit_.parameters)
                return orbit_.rmatvec(v_orbit, **kwargs).state_vector().reshape(-1, 1)

            linear_operator_shape = (orbit_.state.size, orbit_.state_vector().size)
            A = LinearOperator(linear_operator_shape, matvec_func, rmatvec=rmatvec_func, dtype=float)
            b = -1.0 * orbit_.spatiotemporal_mapping().state.reshape(-1, 1)
            if method == 'lsmr':
                result_tuple = lsmr(A, b, **scipy_kwargs)
            elif method == 'lsqr':
                result_tuple = lsqr(A, b, **scipy_kwargs)

        else:
            scipy_kwargs = kwargs.pop('scipy_kwargs', {'tol': 1e-8})
            # Solving `normal equations, A^T A x = A^T b. A^T A is its own transpose hence matvec_func=rmatvec_func

            def matvec_func(v):
                # _state_vector_to_orbit turns state vector into class object.
                v_orbit = orbit_.from_numpy_array(v)
                return orbit_.rmatvec(orbit_.matvec(v_orbit, **kwargs), **kwargs).state_vector().reshape(-1, 1)

            # Currently only allows solving of the normal equations. A^T A x = A^T b
            linear_operator_shape = (orbit_.state_vector().size, orbit_.state_vector().size)
            ATA = LinearOperator(linear_operator_shape, matvec_func, rmatvec=matvec_func, dtype=float)
            ATb = orbit_.rmatvec(-1.0 * orbit_.spatiotemporal_mapping()).state_vector().reshape(-1, 1)

            if kwargs.get('preconditioning', False):
                def p_matvec_func(v):
                    # _state_vector_to_orbit turns state vector into class object.
                    v_orbit = orbit_.from_numpy_array(v)
                    return v_orbit.precondition_matvec(orbit_.preconditioning_parameters,
                                                       **kwargs).state_vector().reshape(-1, 1)

                def p_rmatvec_func(v):
                    # _state_vector_to_orbit turns state vector into class object.
                    v_orbit = orbit_.from_numpy_array(v)
                    return v_orbit.precondition_rmatvec(orbit_.preconditioning_parameters,
                                                        **kwargs).state_vector().reshape(-1, 1)
                scipy_kwargs['M'] = LinearOperator(ATA.shape, p_matvec_func, rmatvec=p_rmatvec_func, dtype=float)

            if method == 'minres':
                result_tuple = minres(ATA, ATb, **scipy_kwargs),
            elif method == 'bicg':
                result_tuple = bicg(ATA, ATb, **scipy_kwargs),
            elif method == 'bicgstab':
                result_tuple = bicgstab(ATA, ATb, **scipy_kwargs),
            elif method == 'gmres':
                result_tuple = gmres(ATA, ATb, **scipy_kwargs),
            elif method == 'lgmres':
                result_tuple = lgmres(ATA, ATb, **scipy_kwargs),
            elif method == 'cg':
                result_tuple = cg(ATA, ATb, **scipy_kwargs),
            elif method == 'cgs':
                result_tuple = cgs(ATA, ATb, **scipy_kwargs),
            elif method == 'qmr':
                result_tuple = qmr(ATA, ATb, **scipy_kwargs),
            elif method == 'gcrotmk':
                result_tuple = gcrotmk(ATA, ATb, **scipy_kwargs),
            else:
                raise ValueError('Unknown solver %s' % method)

        if len(result_tuple) == 1:
            # if passed as tuple, .reshape((a,b)), then need to unpack ((a, b)) into (a, b)
            x = tuple(*result_tuple)[0]
        else:
            x = result_tuple[0]

        dx = orbit_.from_numpy_array(x, **kwargs)
        next_orbit = orbit_.increment(dx)
        next_residual = next_orbit.residual()
        while next_residual > residual and step_size > min_step_size:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_residual = next_orbit.residual()

        stats['nit'] += 1
        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        if step_size <= 10**-5 or (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
            stats['status'] = 0
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        elif stats['nit'] > orbit_maxiter:
            stats['status'] = 2
            stats['residuals'].append(next_residual)
            return orbit_,  stats
        else:
            # Update and restart loop if residual successfully decreases.
            if kwargs.get('log_residual', False):
                stats['residuals'].append(next_residual)
            orbit_ = next_orbit
            residual = next_residual
            if kwargs.get('verbose', False):
                print(' Residual={} after {} cycles'.format(orbit_.residual(), method, step_size))
    else:
        stats['residuals'].append(orbit_.residual())
        return orbit_, stats


def _scipy_optimize_minimize_wrapper(orbit_, method='l-bfgs-b', **kwargs):
    orbit_tol = kwargs.get('orbit_tol', _default_orbit_tol(orbit_, **kwargs))
    residual = orbit_.residual()
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': 1, 'tol': orbit_tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting {} optimization'.format(method))
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(orbit_tol))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')

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
        return x_orbit.cost_function_gradient(x_orbit.spatiotemporal_mapping(), **kwargs).state_vector().ravel()

    scipy_kwargs = kwargs.get('scipy_kwargs', None)
    if isinstance(scipy_kwargs, dict):
        result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                          method=method, jac=_cost_function_jac_scipy_minimize, **scipy_kwargs)
    else:
        result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                          method=method, jac=_cost_function_jac_scipy_minimize)

    orbit_ = orbit_.from_numpy_array(result.x)
    stats['nit'] += 1
    if orbit_.residual() <= orbit_tol:
        stats['status'] = 1
    else:
        stats['status'] = 2

    stats['residuals'].append(orbit_.residual())
    return orbit_, stats


def _print_exit_messages(orbit, status):
    if isinstance(status, tuple):
        status = status[-1]

    if status == 0:
        print('\nInsufficient residual decrease. Exiting with residual {}'.format(orbit.residual()))
    elif status == 1:
        print('\nTolerance threshold met. Exiting with residual {}'.format(orbit.residual()))
    elif status == 2:
        print('\nFailed to converge. Maximum number of iterations reached.'
              ' exiting with residual {}'.format(orbit.residual()))
    elif status == 3:
        print('\nConverged to an equilibrium'
              ' exiting with residual {}'.format(orbit.residual()))
    elif status == 4:
        print('\nConverged to the trivial u(x,t)=0 solution')
    elif status == 5:
        print('\n Relative periodic orbit converged to periodic orbit with essentially zero shift.')
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
    comp_time = kwargs.get('comp_time', 'default')
    if method == 'gradient_descent':
        # Introduction of ugly conditional statements for convenience
        if comp_time == 'long':
            default_max_iter = 1024 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif comp_time == 'medium' or comp_time == 'default':
            default_max_iter = 512 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif comp_time == 'short':
            default_max_iter = 128 * int(np.sqrt(np.product(orbit_.field_shape)))
        elif comp_time == 'minimal':
            default_max_iter = 32 * int(np.sqrt(np.product(orbit_.field_shape)))
        else:
            raise ValueError('If a custom number of iterations is desired, use ''orbit_maxiter'' key word instead.')
    else:
        # Introduction of ugly conditional statements for convenience
        if comp_time == 'long':
            default_max_iter = 500
        elif comp_time == 'medium' or comp_time == 'default':
            default_max_iter = 250
        elif comp_time == 'short':
            default_max_iter = 50
        elif comp_time == 'minimal':
            default_max_iter = 10
        else:
            raise ValueError('If a custom number of iterations is desired, use ''orbit_maxiter'' key word instead.')
    return default_max_iter

