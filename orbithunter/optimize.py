from scipy.linalg import lstsq, pinv, solve
from scipy.optimize import minimize, root, newton_krylov, anderson
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


def converge(orbit_, method='adj', precision='default', comp_time='default', **kwargs):
    """ Main optimization function for orbithunter

    Parameters
    ----------
    orbit_ : Orbit
        The orbit instance serving as the initial condition for optimization.

    kwargs:
        method : str, optional
            Default is 'adj', representing the adjoint descent method, valid choices are:
            'adj', 'lstsq', 'newton_descent', 'lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
            'cg', 'cgs', 'qmr', 'minres', 'gcrotmk', 'hybr', 'lm','broyden1', 'broyden2', 'root_anderson',
            'linearmixing', 'diagbroyden', 'excitingmixing', 'root_krylov',' df-sane',
            'newton_krylov', 'anderson', 'cg_min', 'newton-cg', 'l-bfgs-b', 'tnc', 'bfgs'

        precision : str, optional
            Key word to choose an `orbithunter recommended' tolerance.  Choices include:
            'machine', 'high', 'medium' or 'default' (equivalent), 'low', 'minimal'

        comp_time: str, optional
            Key word to choose an `orbithunter recommended' number of iterations, dependent on the chosen method.
            Choices include : 'excessive', 'thorough', 'long' , 'medium' or 'default' (equivalent), 'short', 'minimal'.

        maxiter : int, optional
            The maximum number of steps; computation time can be highly dependent on this number i.e.
            maxiter=100 for adjoint descent and lstsq have very very different computational times.

        tol : float, optional
            The threshold for the residual function for an orbit approximation to be declared successful.

        scipy_kwargs: dict
            Additional arguments for SciPy solvers.
            There are too many to describe and they depend on the particular algorithm utilized, see scipy
            docs for more info. These pertain to numerical methods:

            See https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html for details
            For methods in the following list: ['lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
                                                'cg', 'cgs', 'qmr', 'minres', 'gcrotmk']

:
            https://docs.scipy.org/doc/scipy/reference/optimize.html
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize


            ['cg_min', 'newton-cg', 'l-bfgs-b', 'tnc', 'bfgs']:

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
            ['hybr', 'lm','broyden1', 'broyden2', 'root_anderson', 'linearmixing',
                        'diagbroyden', 'excitingmixing', 'root_krylov',' df-sane', 'newton_krylov', 'anderson']

        hybrid_maxiter : tuple of two ints, optional
            Only used if method == 'hybrid', contains the maximum number of iterations to be used in gradient
            descent and lstsq, respectively.

        hybrid_tol : tuple of floats, optional
            Only used if method == 'hybrid', contains the tolerance threshold to be used in gradient
            descent and lstsq, respectively.


    Returns
    -------
    OrbitResult :
        OrbitResult instance including optimization properties like exit code, residuals, tol, maxiter, etc. and
        the final resulting orbit approximation.
    Notes
    -----
    Passing tol and maxiter in conjuction with method=='hybrid' will cause the _adj and
    _lstsq to have the same tolerance and number of steps, which is not optimal. Therefore it is recommended to
    either use the keyword arguments 'precision' and 'max_iter' to get default templates, or call converge twice,
    once with method == 'adj' and once more with method == 'lstsq', passing unique
    tol and maxiter to each call.
    """
    # Sometimes it is desirable to provide exact numerical value for tolerance, other times an approximate guideline
    # `precision' is used. This may seem like bad design except the guidelines depend on the orbit_'s discretization
    # and so there is no way of including this in the explicit function kwargs as opposed to **kwargs.
    tol = kwargs.get('tol', _default_tol(orbit_, precision=precision))
    maxiter = kwargs.get('maxiter', _default_maxiter(orbit_, method=method, comp_time=comp_time))
    if not orbit_.residual() < tol:
        # unused if method=='hybrid', but saves us from rewriting it for each conditional.
        kwargs.pop('maxiter', None)
        kwargs.pop('tol', None)
        if method == 'hybrid':
            # make sure that extraneous parameters are removed.
            # There is often a desire to have different tolerances,
            descent_tol, lstsq_tol = kwargs.get('hybrid_tol', (tol, tol))
            descent_iter, lstsq_iter = kwargs.get('hybrid_maxiter',
                                                  (_default_maxiter(orbit_, method='adj', comp_time=comp_time),
                                                   _default_maxiter(orbit_, method='lstsq', comp_time=comp_time)))

            adjoint_orbit, adj_stats = _adjoint_descent(orbit_, descent_tol, descent_iter, **kwargs)
            result_orbit, lstsq_stats = _lstsq(adjoint_orbit,  lstsq_tol, lstsq_iter, **kwargs)

            statistics = {key: ((adj_stats[key], lstsq_stats[key]) if key != 'status' else lstsq_stats[key])
                          for key in sorted({**adj_stats, **lstsq_stats}.keys())}
        elif method == 'newton_descent':
            result_orbit, statistics = _newton_descent(orbit_, tol, maxiter, **kwargs)
        elif method == 'lstsq':
            # solves Ax = b in least-squares manner
            result_orbit, statistics = _lstsq(orbit_, tol, maxiter, **kwargs)
        elif method == 'solve':
            # solves Ax = b in least-squares manner
            result_orbit, statistics = _solve(orbit_, tol, maxiter, **kwargs)
        elif method in ['lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
                                          'cg', 'cgs', 'qmr', 'minres', 'gcrotmk']:
            # solves A^T A x = A^T b repeatedly
            result_orbit, statistics = _scipy_sparse_linalg_solver_wrapper(orbit_, tol, maxiter, method=method, **kwargs)
        elif method in ['cg_min', 'newton-cg', 'l-bfgs-b', 'tnc', 'bfgs']:
            # minimizes cost functional 1/2 F^2
            if method == 'cg_min':
                # had to use an alias because this is also defined for scipy.sparse.linalg
                method = 'cg'
            result_orbit, statistics = _scipy_optimize_minimize_wrapper(orbit_, tol, maxiter, method=method,  **kwargs)
        elif method in ['hybr', 'lm','broyden1', 'broyden2', 'root_anderson', 'linearmixing',
                        'diagbroyden', 'excitingmixing', 'root_krylov',' df-sane', 'newton_krylov', 'anderson']:
            result_orbit, statistics = _scipy_optimize_root_wrapper(orbit_, tol, maxiter, method=method,  **kwargs)
        else:
            result_orbit, statistics = _adjoint_descent(orbit_, tol, maxiter, **kwargs)

        if kwargs.get('verbose', False):
            _print_exit_messages(result_orbit, statistics['status'])
            sys.stdout.flush()
        return OrbitResult(orbit=result_orbit, **statistics)
    else:
        return OrbitResult(orbit=orbit_, nit=0, residuals=[orbit_.residual()], status=-1, maxiter=maxiter, tol=[tol])


def _adjoint_descent(orbit_, tol, maxiter, min_step=1e-6, **kwargs):
    """

    Parameters
    ----------
    orbit_ : Orbit (or subclass) instance
    tol : float
        Numerical tolerance for the residual.
    maxiter : int
        Maximum number of adjoint descent iterations
    min_step : float
        Minimum backtracking step size.
    kwargs : dict
        Keyword arguments relevant for differential algebraic equations and cost function gradient.

    Returns
    -------

    """
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    verbose = kwargs.get('verbose', False)
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting adjoint descent')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
        sys.stdout.flush()
    mapping = orbit_.dae(**kwargs)
    residual = mapping.residual(dae=False)
    step_size = 1
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    while residual > tol and stats['status'] == 1:
        # Calculate the step
        gradient = orbit_.cost_function_gradient(mapping, **kwargs)
        # Negative sign -> 'descent'
        next_orbit = orbit_.increment(gradient, step_size=-1.0*step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.dae(**kwargs)
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(dae=False)
        while next_residual >= residual and step_size > min_step:
            # reduce the step size until minimum is reached or residual decreases.
            step_size /= 2
            next_orbit = orbit_.increment(gradient, step_size=-1.0*step_size)
            next_mapping = next_orbit.dae(**kwargs)
            next_residual = next_mapping.residual(dae=False)
        else:
            orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                              step_size, min_step, residual, next_residual,
                                              'adj', verbose=verbose,
                                              residual_logging=kwargs.get('residual_logging', None))
            mapping = next_mapping
            residual = next_residual
    else:
        return orbit_,  stats


def _newton_descent(orbit_, tol, maxiter, min_step=1e-6, **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    min_step
    kwargs

    Returns
    -------

    """
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    step_size = kwargs.get('step_size', 0.001)
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    residual = orbit_.residual()
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting Newton descent optimization')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
        sys.stdout.flush()
    mapping = orbit_.dae(**kwargs)
    residual = mapping.residual(dae=False)
    while residual > tol and stats['status'] == 1:
        # Solve A dx = b <--> J dx = - f, for dx.
        A, b = orbit_.jacobian(), -1*mapping.state.ravel()
        inv_A = pinv(A)
        dx = orbit_.from_numpy_array(np.dot(inv_A, b))
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.dae(**kwargs)
        next_residual = next_mapping.residual(dae=False)
        # This modifies the step size if too large; i.e. its a very crude way of handling curvature.
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.dae(**kwargs)
            next_residual = next_mapping.residual(dae=False)
        else:
            inner_nit = 1
            if kwargs.get('approximation', True):
                # Re-use the same pseudoinverse for many inexact solutions to dx_n = - A^+(x) F(x + dx_{n-1})
                b = -1 * next_mapping.state.ravel()
                dx = orbit_.from_numpy_array(np.dot(inv_A, b))
                inner_orbit = next_orbit.increment(dx, step_size=step_size)
                inner_mapping = inner_orbit.dae(**kwargs)
                inner_residual = inner_mapping.residual(dae=False)
                while inner_residual < next_residual:
                    next_orbit = inner_orbit
                    next_mapping = inner_mapping
                    next_residual = inner_residual
                    b = -1 * next_mapping.state.ravel()
                    dx = orbit_.from_numpy_array(np.dot(inv_A, b))
                    inner_orbit = next_orbit.increment(dx, step_size=step_size)
                    inner_mapping = inner_orbit.dae(**kwargs)
                    inner_residual = inner_mapping.residual(dae=False)
                    inner_nit += 1
            # If the trigger that broke the while loop was step_size then
            # assume next_residual < residual was not met.
            orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                              step_size, min_step, residual, next_residual,
                                              'newton_descent',
                                              inner_nit=inner_nit,
                                              residual_logging=kwargs.get('residual_logging', False),
                                              verbose=kwargs.get('verbose', False))
            mapping = next_mapping
            residual = next_residual
    else:
        stats['residuals'].append(orbit_.residual())
        return orbit_, stats


def _lstsq(orbit_, tol, maxiter, min_step=1e-6,  **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    min_step
    kwargs

    Returns
    -------

    """
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    mapping = orbit_.dae(**kwargs)
    residual = mapping.residual(dae=False)
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting lstsq optimization')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
    while residual > tol and stats['status'] == 1:
        step_size = 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)
        dx = orbit_.from_numpy_array(lstsq(A, b)[0], **kwargs)
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.dae(**kwargs)
        next_residual = next_mapping.residual(dae=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.dae(**kwargs)
            next_residual = next_mapping.residual(dae=False)
        else:
            orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                              step_size, min_step, residual, next_residual,
                                              'lstsq',
                                              residual_logging=kwargs.get('residual_logging', False),
                                              verbose=kwargs.get('verbose', False))
            mapping = next_mapping
            residual = next_residual
    else:
        return orbit_, stats


def _solve(orbit_, tol, maxiter, min_step=1e-6,  **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    min_step
    kwargs

    Returns
    -------

    """
    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    mapping = orbit_.dae(**kwargs)
    residual = mapping.residual(dae=False)
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting lstsq optimization')
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
    while residual > tol and stats['status'] == 1:
        step_size = 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)
        dx = orbit_.from_numpy_array(solve(A, b)[0], **kwargs)
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.dae(**kwargs)
        next_residual = next_mapping.residual(dae=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.dae(**kwargs)
            next_residual = next_mapping.residual(dae=False)
        else:
            orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                              step_size, min_step, residual, next_residual,
                                              'lstsq',
                                              residual_logging=kwargs.get('residual_logging', False),
                                              verbose=kwargs.get('verbose', False))
            mapping = next_mapping
            residual = next_residual
    else:
        return orbit_, stats


def _scipy_sparse_linalg_solver_wrapper(orbit_, tol, maxiter, method='minres', min_step=1e-6, **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    method
    min_step
    kwargs

    Returns
    -------

    """
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    residual = orbit_.residual()
    if kwargs.get('verbose', False):
        print('\n------------------------------------------------------------------------------------------------')
        print('Starting {} optimization'.format(method))
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
        sys.stdout.flush()
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}

    while residual > tol and stats['status'] == 1:
        step_size = 1
        if method in ['lsmr', 'lsqr']:
            if stats['nit'] == 0:
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
            b = -1.0 * orbit_.dae(**kwargs).state.reshape(-1, 1)
            if method == 'lsmr':
                result_tuple = lsmr(A, b, **scipy_kwargs)
            else:
                result_tuple = lsqr(A, b, **scipy_kwargs)

        else:
            if stats['nit'] == 0:
                scipy_kwargs = kwargs.pop('scipy_kwargs', {'tol': 1e-3})

            # Solving `normal equations, A^T A x = A^T b. A^T A is its own transpose hence matvec_func=rmatvec_func
            def matvec_func(v):
                # _state_vector_to_orbit turns state vector into class object.
                v_orbit = orbit_.from_numpy_array(v)
                return orbit_.rmatvec(orbit_.matvec(v_orbit, **kwargs), **kwargs).state_vector().reshape(-1, 1)

            # Currently only allows solving of the normal equations. A^T A x = A^T b
            linear_operator_shape = (orbit_.state_vector().size, orbit_.state_vector().size)
            ATA = LinearOperator(linear_operator_shape, matvec_func, rmatvec=matvec_func, dtype=float)
            ATb = orbit_.rmatvec(-1.0 * orbit_.dae(**kwargs)).state_vector().reshape(-1, 1)

            ############################################################################################################

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
            x = tuple(*result_tuple)[0]
        else:
            x = result_tuple[0]

        dx = orbit_.from_numpy_array(x, **kwargs)
        next_orbit = orbit_.increment(dx)
        next_mapping = next_orbit.dae(**kwargs)
        next_residual = next_mapping.residual(dae=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.dae(**kwargs)
            next_residual = next_mapping.residual(dae=False)
        else:
            # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
            orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                              1, 0, residual, next_residual,
                                              method, residual_logging=kwargs.get('residual_logging', False),
                                              verbose=kwargs.get('verbose', False))
            residual = next_residual

    else:
        return orbit_, stats


def _scipy_optimize_minimize_wrapper(orbit_, tol, maxiter, method='l-bfgs-b',  **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    method
    kwargs

    Returns
    -------

    """
    residual = orbit_.residual()
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting {} optimization'.format(method))
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Maximum iteration number : {}'.format(maxiter))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
        sys.stdout.flush()

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
        return x_orbit.cost_function_gradient(x_orbit.dae(**kwargs), **kwargs).state_vector().ravel()

    scipy_kwargs = kwargs.get('scipy_kwargs', {'tol': tol})
    while residual > tol and stats['status'] == 1:
        result = minimize(_cost_function_scipy_minimize, orbit_.state_vector(),
                          method=method, jac=_cost_function_jac_scipy_minimize, **scipy_kwargs)
        scipy_kwargs['tol'] /= 10.

        next_orbit = orbit_.from_numpy_array(result.x)
        next_residual = next_orbit.residual()
        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                          1, 0, residual, next_residual,
                                          method, residual_logging=kwargs.get('residual_logging', False),
                                          verbose=kwargs.get('verbose', False))
        residual = next_residual
    else:
        return orbit_, stats


def _scipy_optimize_root_wrapper(orbit_, tol, maxiter, method='lgmres', **kwargs):
    """

    Parameters
    ----------
    orbit_
    tol
    maxiter
    method
    kwargs

    Returns
    -------

    """
    residual = orbit_.residual()
    ftol = kwargs.get('ftol', np.product(orbit_.shape) * 10**-10)
    stats = {'nit': 0, 'residuals': [residual], 'maxiter': maxiter, 'tol': tol, 'status': 1}
    if kwargs.get('verbose', False):
        print('\n-------------------------------------------------------------------------------------------------')
        print('Starting {} optimization'.format(method))
        print('Initial residual : {}'.format(orbit_.residual()))
        print('Target residual tolerance : {}'.format(tol))
        print('Initial guess : {}'.format(repr(orbit_)))
        print('-------------------------------------------------------------------------------------------------')
        sys.stdout.flush()

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
        xvec = x_orbit.dae(**kwargs).state_vector().ravel()
        xvec[-(x_orbit.state_vector().size - len(x_orbit.parameters)):] = 0
        return xvec

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

        x_orbit = orbit_.from_numpy_array(x)
        return x_orbit.cost_function_gradient(x_orbit.dae(**kwargs), **kwargs).state_vector().ravel()

    while residual > tol and stats['status'] == 1:
        if method == 'newton_krylov':
            scipy_kwargs = dict({'f_tol': 1e-6}, **kwargs.get('scipy_kwargs', {}))
            result_state_vector = newton_krylov(_cost_function_scipy_root, orbit_.state_vector(),
                                                **scipy_kwargs)
        elif method == 'anderson':
            scipy_kwargs = dict({'f_tol': 1e-6}, **kwargs.get('scipy_kwargs', {}))
            result_state_vector = anderson(_cost_function_scipy_root, orbit_.state_vector().ravel(),
                                           **scipy_kwargs)

        elif method in ['root_anderson', 'linearmixing', 'diagbroyden',
                        'excitingmixing', 'krylov',' df-sane']:
            scipy_kwargs = dict({'tol': 1e-6}, **kwargs.get('scipy_kwargs', {}))
            result_state_vector = root(_cost_function_scipy_root, orbit_.state_vector().ravel(),
                                       method=method, jac=_cost_function_jac_scipy_root,
                                       **scipy_kwargs)
        else:
            stats['residuals'].append(orbit_.residual())
            return orbit_, stats
        next_orbit = orbit_.from_numpy_array(result_state_vector)
        next_residual = next_orbit.residual()
        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        orbit_, stats = _parse_correction(orbit_, next_orbit, stats, tol, maxiter, ftol,
                                          1, 0, residual, next_residual,
                                          method, residual_logging=kwargs.get('residual_logging', False),
                                          verbose=kwargs.get('verbose', False))
        residual = next_residual
    else:
        stats['residuals'].append(orbit_.residual())
        return orbit_, stats


def _print_exit_messages(orbit, status):
    """

    Parameters
    ----------
    orbit
    status

    Returns
    -------

    """
    if isinstance(status, tuple):
        status = status[-1]
    if status == -1:
        print('\nConverged. Exiting with residual={}'.format(orbit.residual()))
    elif status == 0:
        print('\nStalled. Exiting with residual={}'.format(orbit.residual()))
    elif status == 2:
        print('\nMaximum number of iterations reached.'
              ' exiting with residual={}'.format(orbit.residual()))
    elif status == 3:
        print('\nInsufficient residual decrease, exiting with residual={}'.format(orbit.residual()))
    else:
        # A general catch all for custom status flag values
        print('\n Optimization termination with status {} and residual {}'.format(status, orbit.residual()))
    return None


def _default_tol(orbit_, precision='default'):
    """ Wrapper to allow str keyword argument to be used to choose orbit tolerance dependent on method

    Parameters
    ----------
    orbit_
    kwargs

    Returns
    -------

    Notes
    -----
    The purposes of this function are to not have conflicting types for tol kwarg and improved convenience
    in specifying the default tolerance per method.

    """
    # Introduction of ugly conditional statements for convenience
    if precision == 'machine':
        default_tol = np.product(orbit_.field_shape) * 10**-12
    elif precision == 'high':
        default_tol = np.product(orbit_.field_shape) * 10**-10
    elif precision == 'medium' or precision == 'default':
        default_tol = np.product(orbit_.field_shape) * 10**-7
    elif precision == 'low':
        default_tol = np.product(orbit_.field_shape) * 10**-4
    elif precision == 'minimal':
        default_tol = np.product(orbit_.field_shape) * 10**-1
    else:
        raise ValueError('If a custom tolerance is desired, use ''tol'' key word instead.')

    return default_tol


def _default_maxiter(orbit_, method='adj', comp_time='default'):
    """ Wrapper to allow str keyword argument to be used to choose number of iterations dependent on method

    Parameters
    ----------
    orbit_
    kwargs

    Returns
    -------

    Notes
    -----
    The purposes of this function are to not have conflicting types for tol kwarg and improved convenience
    in specifying the default tolerance per method.

    """
    if method in ['lstsq', 'newton_descent', 'lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
                  'cg', 'cgs', 'qmr', 'minres', 'gcrotmk']:
        # Introduction of ugly conditional statements for convenience
        if comp_time == 'excessive':
            default_max_iter = 5000
        elif comp_time == 'thorough':
            default_max_iter = 1000
        elif comp_time == 'long':
            default_max_iter = 500
        elif comp_time == 'medium' or comp_time == 'default':
            default_max_iter = 250
        elif comp_time == 'short':
            default_max_iter = 50
        elif comp_time == 'minimal':
            default_max_iter = 10
        else:
            raise ValueError('If a custom number of iterations is desired, use ''maxiter'' key word instead.')
    elif method in ['cg_min', 'newton-cg', 'bfgs', 'l-bfgs-b', 'tnc']:
        # The way that these methods work is that these aren't really iteration controls but rather tolerance controls.
        # if default_max_iter = 3 then this means the function scipy.optimize.minimize is called 3 times with
        # tolerance values tol, tol/10, tol/100. The reason for this is because the methods differ wildly
        # in what tolerance and iteration number mean.
        if comp_time == 'excessive':
            default_max_iter = 10
        elif comp_time == 'thorough':
            default_max_iter = 5
        elif comp_time == 'long':
            default_max_iter = 4
        elif comp_time == 'medium' or comp_time == 'default':
            default_max_iter = 3
        elif comp_time == 'short':
            default_max_iter = 2
        elif comp_time == 'minimal':
            default_max_iter = 1
        else:
            raise ValueError('If a custom number of iterations is desired, use ''maxiter'' key word instead.')
    else:
        # Introduction of ugly conditional statements for convenience
        if comp_time == 'excessive':
            default_max_iter = 512 * int(np.product(orbit_.field_shape))
        elif comp_time == 'thorough':
            default_max_iter = 128 * int(np.product(orbit_.field_shape))
        elif comp_time == 'long':
            default_max_iter = 32 * int(np.product(orbit_.field_shape))
        elif comp_time == 'medium' or comp_time == 'default':
            default_max_iter = 16 * int(np.product(orbit_.field_shape))
        elif comp_time == 'short':
            default_max_iter = 8 * int(np.product(orbit_.field_shape))
        elif comp_time == 'minimal':
            default_max_iter = int(np.product(orbit_.field_shape))
        else:
            raise ValueError('If a custom number of iterations is desired, use ''maxiter'' key word instead.')

    return default_max_iter


def _parse_correction(orbit_, next_orbit_, stats, tol, maxiter, ftol, step_size, min_step,
                      residual, next_residual, method, residual_logging=False, inner_nit=None, verbose=False):
    # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
    stats['nit'] += 1
    if next_residual <= tol:
        stats['status'] = -1
        stats['residuals'].append(next_residual)
        return next_orbit_,  stats
    elif step_size <= min_step:
        stats['status'] = 0
        stats['residuals'].append(next_residual)
        return orbit_,  stats
    elif stats['nit'] == maxiter:
        stats['status'] = 2
        stats['residuals'].append(next_residual)
        return next_orbit_,  stats
    elif (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
        stats['status'] = 3
        stats['residuals'].append(next_residual)
        return next_orbit_,  stats
    else:
        if method == 'adj':
            if verbose:
                if np.mod(stats['nit'], 5000) == 0:
                    print('\n Residual={:.7f} after {} adjoint descent steps. Parameters={}'.format(
                          next_residual, stats['nit'], orbit_.parameters))
                elif np.mod(stats['nit'], 100) == 0:
                    print('#', end='')
            # Too many steps elapse for adjoint descent; this would eventually cause dramatic slow down.
            if residual_logging and np.mod(stats['nit'], 1000) == 0:
                stats['residuals'].append(next_residual)

        elif method == 'newton_descent':
            if verbose and inner_nit is not None:
                print('Residual={} after {} inner loops, for a total Newton descent step with size {}'.
                      format(next_residual, inner_nit, step_size*inner_nit))
            elif verbose and np.mod(stats['nit'], 25) == 0:
                print('Residual={} after {} Newton descent steps '.
                      format(next_residual, stats['nit']))

            if residual_logging:
                stats['residuals'].append(next_residual)
        else:
            if residual_logging:
                stats['residuals'].append(next_residual)
            if verbose:
                if np.mod(stats['nit'], 25) == 0:
                    print(' Residual={:.7f} after {} {} iterations. Parameters={}'.format(next_residual,
                          stats['nit'], method, orbit_.parameters))
                else:
                    print('#', end='')
            sys.stdout.flush()
        return next_orbit_, stats