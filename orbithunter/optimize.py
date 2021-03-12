from scipy.linalg import lstsq, pinv, solve
from scipy.optimize import minimize, root, newton_krylov, anderson
from scipy.sparse.linalg import (
    LinearOperator,
    bicg,
    bicgstab,
    gmres,
    lgmres,
    cg,
    cgs,
    qmr,
    minres,
    lsqr,
    lsmr,
    gcrotmk,
)
import sys
import numpy as np

__all__ = ["hunt"]


class OrbitResult(dict):
    """ Represents the result of applying converge; format copied from SciPy scipy.optimize.OptimizeResult.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    status : int
        Integer which tracks the type of exit from whichever numerical algorithm was applied.
        See Notes for more details.
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

    The descriptions for each value of status are as follows.
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
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def hunt(orbit_, *methods, **kwargs):
    """ Main optimization function for orbithunter

    Parameters
    ----------
    orbit_ : Orbit
        The orbit instance serving as the initial condition for optimization.
    methods : str or multiple str or tuple of str
        Representing the numerical methods to hunt with, valid choices are:
        'newton_descent', 'lstsq', 'solve', 'adj', 'lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
        'cg', 'cgs', 'qmr', 'minres', 'gcrotmk','nelder-mead', 'powell', 'cg_min', 'bfgs', 'newton-cg', 'l-bfgs-b',
        'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'hybr',
        'lm','broyden1', 'broyden2', 'root_anderson', 'linearmixing','diagbroyden', 'excitingmixing', 'root_krylov',
        ' df-sane', 'newton_krylov', 'anderson'

        adj, newton_descent, lstsq, lsqr, lsmr, bicg, bicgstab, gmres, lgmres, cg, cgs, qmr, minres,gcrotmk,
        cg_min, bfgs, newton-cg, l-bfgs-b, tns, slsqp

        The following are either not supported or highly unrecommended for the KSE
nelder-mead (very very slow), powell (very slow), cobyla (slow),


    kwargs:
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
            https://docs.scipy.org/doc/scipy/reference/populated/scipy.optimize.minimize.html#scipy.optimize.minimize
            ['cg_min', 'newton-cg', 'l-bfgs-b', 'tnc', 'bfgs']

            https://docs.scipy.org/doc/scipy/reference/populated/scipy.optimize.root.html
            ['hybr', 'lm','broyden1', 'broyden2', 'root_anderson', 'linearmixing',
                        'diagbroyden', 'excitingmixing', 'root_krylov',' df-sane', 'newton_krylov', 'anderson']

    Returns
    -------
    OrbitResult :
        OrbitResult instance including optimization properties like exit code, residuals, tol, maxiter, etc. and
        the final resulting orbit approximation.
    Notes
    -----
    User should be aware of the existence of scipy.optimize.show_options()

    Sometimes it is desirable to provide exact numerical value for tolerance, other times an approximate guideline
    `precision' is used. This may seem like bad design except the guidelines depend on the orbit_'s discretization
    and so there is no way of including this in the explicit function kwargs as opposed to **kwargs.
    unpacking unintended nested tuples i.e. ((a, b, ...)) -> (a, b, ...); leaves unnested tuples invariant.
    New shape must be tuple; i.e. iterable and have __len__
    """

    kwargs = {k: v.copy() if hasattr(v, "copy") else v for k, v in kwargs.items()}
    # so that list.pop() method can be used, cast tuple as lists
    methods = tuple(*methods) or kwargs.pop("methods", "adj")

    if len(methods) == 1 and isinstance(*methods, tuple):
        methods = tuple(*methods)
    elif isinstance(methods, str):
        methods = (methods,)

    runtime_statistics = {}
    for method in methods:
        if method == "newton_descent":
            orbit_, method_statistics = _newton_descent(orbit_, **kwargs)
        elif method == "lstsq":
            # solves Ax = b in least-squares manner
            orbit_, method_statistics = _lstsq(orbit_, **kwargs)
        elif method == "solve":
            # solves Ax = b
            orbit_, method_statistics = _solve(orbit_, **kwargs)
        elif method in [
            "lsqr",
            "lsmr",
            "bicg",
            "bicgstab",
            "gmres",
            "lgmres",
            "cg",
            "cgs",
            "qmr",
            "minres",
            "gcrotmk",
        ]:
            # solves A^T A x = A^T b using iterative method
            orbit_, method_statistics = _scipy_sparse_linalg_solver_wrapper(
                orbit_, method=method, **kwargs
            )
        elif method in [
            "nelder-mead",
            "powell",
            "cg_min",
            "bfgs",
            "newton-cg",
            "l-bfgs-b",
            "tnc",
            "cobyla",
            "slsqp",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]:
            # minimizes cost functional 1/2 F^2
            if method == "cg_min":
                # had to use an alias because this is also defined for scipy.sparse.linalg
                method = "cg"
            orbit_, method_statistics = _scipy_optimize_minimize_wrapper(
                orbit_, method=method, **kwargs
            )
        elif method in [
            "hybr",
            "lm",
            "broyden1",
            "broyden2",
            "root_anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "root_krylov",
            " df-sane",
            "newton_krylov",
            "anderson",
        ]:
            orbit_, method_statistics = _scipy_optimize_root_wrapper(
                orbit_, method=method, **kwargs
            )
        else:
            orbit_, method_statistics = _adjoint_descent(orbit_, **kwargs)

        # If runtime_statistics is empty then initialize with the previous method statistics.
        if not runtime_statistics:
            runtime_statistics = method_statistics
        else:
            # Keep a consistent order of the runtime information, scalar values converted to lists for multiple
            # method runs.
            for key in sorted({**runtime_statistics, **method_statistics}.keys()):
                if key == "status":
                    runtime_statistics[key] = method_statistics.get("status", -1)
                elif isinstance(runtime_statistics.get(key, []), list):
                    runtime_statistics.get(key, []).extend(
                        method_statistics.get(key, [])
                    )
                else:
                    runtime_statistics[key] = [
                        runtime_statistics[key],
                        method_statistics.get(key, []),
                    ]

    if kwargs.get("verbose", False):
        _print_exit_messages(orbit_, runtime_statistics["status"])
        sys.stdout.flush()

    return OrbitResult(orbit=orbit_, **runtime_statistics)


def _adjoint_descent(orbit_, tol=1e-6, maxiter=10000, min_step=1e-9, **kwargs):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    runtime_statistics = {
        "method": "adj",
        "nit": 0,
        "residuals": [orbit_.residual()],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    ftol = kwargs.get("ftol", 1e-9)
    verbose = kwargs.get("verbose", False)
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting adjoint descent")
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    mapping = orbit_.eqn(**kwargs)
    residual = mapping.residual(eqn=False)
    step_size = 1
    while residual > tol and runtime_statistics["status"] == 1:
        # Calculate the step
        gradient = orbit_.cost_function_gradient(mapping, **kwargs)
        # Negative sign -> 'descent'
        next_orbit = orbit_.increment(gradient, step_size=-1.0 * step_size)
        # Calculate the mapping and store; need it for next step and to compute residual.
        next_mapping = next_orbit.eqn(**kwargs)
        # Compute residual to see if step succeeded
        next_residual = next_mapping.residual(eqn=False)
        while next_residual >= residual and step_size > min_step:
            # reduce the step size until minimum is reached or residual decreases.
            step_size /= 2
            next_orbit = orbit_.increment(gradient, step_size=-1.0 * step_size)
            next_mapping = next_orbit.eqn(**kwargs)
            next_residual = next_mapping.residual(eqn=False)
        else:
            orbit_, runtime_statistics = _process_correction(
                orbit_,
                next_orbit,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                residual,
                next_residual,
                "adj",
                verbose=verbose,
                residual_logging=kwargs.get("residual_logging", False),
            )
            mapping = next_mapping
            residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _newton_descent(orbit_, tol=1e-6, maxiter=500, min_step=1e-9, **kwargs):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    step_size = kwargs.get("step_size", 0.001)
    residual = orbit_.residual()
    runtime_statistics = {
        "method": "newton_descent",
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    ftol = kwargs.get("ftol", 1e-9)
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting Newton descent optimization")
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    mapping = orbit_.eqn(**kwargs)
    residual = mapping.residual(eqn=False)
    while residual > tol and runtime_statistics["status"] == 1:
        # Solve A dx = b <--> J dx = - f, for dx.
        A, b = orbit_.jacobian(), -1 * mapping.state.ravel()
        inv_A = pinv(A)
        dx = orbit_.from_numpy_array(np.dot(inv_A, b))
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.eqn(**kwargs)
        next_residual = next_mapping.residual(eqn=False)
        # This modifies the step size if too large; i.e. its a very crude way of handling curvature.
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.eqn(**kwargs)
            next_residual = next_mapping.residual(eqn=False)
        else:
            inner_nit = 1
            if kwargs.get("approximation", True):
                # Re-use the same pseudoinverse for many inexact solutions to dx_n = - A^+(x) F(x + dx_{n-1})
                b = -1 * next_mapping.state.ravel()
                dx = orbit_.from_numpy_array(np.dot(inv_A, b))
                inner_orbit = next_orbit.increment(dx, step_size=step_size)
                inner_mapping = inner_orbit.eqn(**kwargs)
                inner_residual = inner_mapping.residual(eqn=False)
                while inner_residual < next_residual:
                    next_orbit = inner_orbit
                    next_mapping = inner_mapping
                    next_residual = inner_residual
                    b = -1 * next_mapping.state.ravel()
                    dx = orbit_.from_numpy_array(np.dot(inv_A, b))
                    inner_orbit = next_orbit.increment(dx, step_size=step_size)
                    inner_mapping = inner_orbit.eqn(**kwargs)
                    inner_residual = inner_mapping.residual(eqn=False)
                    inner_nit += 1
                    if inner_residual < tol:
                        next_orbit = inner_orbit
                        next_mapping = inner_mapping
                        next_residual = inner_residual
                        break
            # If the trigger that broke the while loop was step_size then
            # assume next_residual < residual was not met.
            orbit_, runtime_statistics = _process_correction(
                orbit_,
                next_orbit,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                residual,
                next_residual,
                "newton_descent",
                inner_nit=inner_nit,
                residual_logging=kwargs.get("residual_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
            residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _lstsq(orbit_, tol=1e-6, maxiter=500, min_step=1e-9, **kwargs):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt
    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    ftol = kwargs.get("ftol", 1e-9)
    mapping = orbit_.eqn(**kwargs)
    residual = mapping.residual(eqn=False)
    runtime_statistics = {
        "method": "lstsq",
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting lstsq optimization")
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
    while residual > tol and runtime_statistics["status"] == 1:
        step_size = 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)
        dx = orbit_.from_numpy_array(lstsq(A, b)[0], **kwargs)
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.eqn(**kwargs)
        next_residual = next_mapping.residual(eqn=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.eqn(**kwargs)
            next_residual = next_mapping.residual(eqn=False)
        else:
            orbit_, runtime_statistics = _process_correction(
                orbit_,
                next_orbit,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                residual,
                next_residual,
                "lstsq",
                residual_logging=kwargs.get("residual_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
            residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _solve(orbit_, maxiter=500, tol=1e-6, min_step=1e-9, **kwargs):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt
    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    mapping = orbit_.eqn(**kwargs)
    residual = mapping.residual(eqn=False)
    ftol = kwargs.get("ftol", 1e-9)
    runtime_statistics = {
        "method": "solve",
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting lstsq optimization")
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
    while residual > tol and runtime_statistics["status"] == 1:
        step_size = 1
        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)
        dx = orbit_.from_numpy_array(solve(A, b)[0], **kwargs)
        next_orbit = orbit_.increment(dx, step_size=step_size)
        next_mapping = next_orbit.eqn(**kwargs)
        next_residual = next_mapping.residual(eqn=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.eqn(**kwargs)
            next_residual = next_mapping.residual(eqn=False)
        else:
            orbit_, runtime_statistics = _process_correction(
                orbit_,
                next_orbit,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                residual,
                next_residual,
                "lstsq",
                residual_logging=kwargs.get("residual_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
            residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _scipy_sparse_linalg_solver_wrapper(
    orbit_, method="minres", maxiter=10, tol=1e-6, min_step=1e-9, **kwargs
):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt
    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    residual = orbit_.residual()
    if kwargs.get("verbose", False):
        print(
            "\n------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    ftol = kwargs.get("ftol", 1e-9)
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    while residual > tol and runtime_statistics["status"] == 1:
        step_size = 1
        if method in ["lsmr", "lsqr"]:
            if runtime_statistics["nit"] == 0:
                scipy_kwargs = kwargs.pop("scipy_kwargs", {"atol": 1e-6, "btol": 1e-6})
            # Solving least-squares equations, A x = b

            def matvec_func(v):
                # _orbit_vector_to_orbit turns state vector into class object.
                nonlocal orbit_
                v_orbit = orbit_.from_numpy_array(v)
                return orbit_.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

            def rmatvec_func(v):
                # _orbit_vector_to_orbit turns state vector into class object.
                nonlocal orbit_
                v_orbit = orbit_.from_numpy_array(v, parameters=orbit_.parameters)
                return orbit_.rmatvec(v_orbit, **kwargs).orbit_vector().reshape(-1, 1)

            linear_operator_shape = (orbit_.state.size, orbit_.orbit_vector().size)
            A = LinearOperator(
                linear_operator_shape, matvec_func, rmatvec=rmatvec_func, dtype=float
            )
            b = -1.0 * orbit_.eqn(**kwargs).state.reshape(-1, 1)
            if method == "lsmr":
                result_tuple = lsmr(A, b, **scipy_kwargs)
            else:
                result_tuple = lsqr(A, b, **scipy_kwargs)

        else:
            if runtime_statistics["nit"] == 0:
                scipy_kwargs = kwargs.pop("scipy_kwargs", {"tol": 1e-3})

            # Solving `normal equations, A^T A x = A^T b. A^T A is its own transpose hence matvec_func=rmatvec_func
            def matvec_func(v):
                # _orbit_vector_to_orbit turns state vector into class object.
                nonlocal orbit_
                v_orbit = orbit_.from_numpy_array(v)
                return (
                    orbit_.rmatvec(orbit_.matvec(v_orbit, **kwargs), **kwargs)
                    .orbit_vector()
                    .reshape(-1, 1)
                )

            # Currently only allows solving of the normal equations. A^T A x = A^T b
            linear_operator_shape = (
                orbit_.orbit_vector().size,
                orbit_.orbit_vector().size,
            )
            ATA = LinearOperator(
                linear_operator_shape, matvec_func, rmatvec=matvec_func, dtype=float
            )
            ATb = (
                orbit_.rmatvec(-1.0 * orbit_.eqn(**kwargs))
                .orbit_vector()
                .reshape(-1, 1)
            )

            ############################################################################################################

            if method == "minres":
                result_tuple = (minres(ATA, ATb, **scipy_kwargs),)
            elif method == "bicg":
                result_tuple = (bicg(ATA, ATb, **scipy_kwargs),)
            elif method == "bicgstab":
                result_tuple = (bicgstab(ATA, ATb, **scipy_kwargs),)
            elif method == "gmres":
                result_tuple = (gmres(ATA, ATb, **scipy_kwargs),)
            elif method == "lgmres":
                result_tuple = (lgmres(ATA, ATb, **scipy_kwargs),)
            elif method == "cg":
                result_tuple = (cg(ATA, ATb, **scipy_kwargs),)
            elif method == "cgs":
                result_tuple = (cgs(ATA, ATb, **scipy_kwargs),)
            elif method == "qmr":
                result_tuple = (qmr(ATA, ATb, **scipy_kwargs),)
            elif method == "gcrotmk":
                result_tuple = (gcrotmk(ATA, ATb, **scipy_kwargs),)
            else:
                raise ValueError("Unknown solver %s" % method)

        if len(result_tuple) == 1:
            x = tuple(*result_tuple)[0]
        else:
            x = result_tuple[0]

        dx = orbit_.from_numpy_array(x, **kwargs)
        next_orbit = orbit_.increment(dx)
        next_mapping = next_orbit.eqn(**kwargs)
        next_residual = next_mapping.residual(eqn=False)
        while next_residual > residual and step_size > min_step:
            # Continues until either step is too small or residual decreases
            step_size /= 2.0
            next_orbit = orbit_.increment(dx, step_size=step_size)
            next_mapping = next_orbit.eqn(**kwargs)
            next_residual = next_mapping.residual(eqn=False)
        else:
            # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
            orbit_, runtime_statistics = _process_correction(
                orbit_,
                next_orbit,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                1,
                0,
                residual,
                next_residual,
                method,
                residual_logging=kwargs.get("residual_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            residual = next_residual

    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _scipy_optimize_minimize_wrapper(
    orbit_, method="l-bfgs-b", maxiter=10, tol=1e-6, **kwargs
):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt
    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    residual = orbit_.residual()
    ftol = kwargs.get("ftol", 1e-9)
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    scipy_kwargs = kwargs.get("scipy_kwargs", {"tol": tol})
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()

    def _minfunc(x):
        """
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        """

        """
        Note that passing Class as a function avoids dangerous statements using eval()
        """
        nonlocal orbit_
        x_orbit = orbit_.from_numpy_array(x, **kwargs)
        return x_orbit.residual()

    def _minjac(x):
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
        nonlocal orbit_
        x_orbit = orbit_.from_numpy_array(x)
        return (
            x_orbit.cost_function_gradient(x_orbit.eqn(**kwargs), **kwargs)
            .orbit_vector()
            .ravel()
        )

    if method in ["trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]:

        def _minhess(x):
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
            nonlocal orbit_
            x_orbit = orbit_.from_numpy_array(x)
            return x_orbit.cost_function_hessian(**kwargs)

        scipy_kwargs = {
            **scipy_kwargs,
            "method": method,
            "jac": _minjac,
            "hessp": _minhess,
        }
    elif method in ["linearmixing"]:
        scipy_kwargs = {**scipy_kwargs, "method": method}
    else:
        scipy_kwargs = {**scipy_kwargs, "method": method, "jac": _minjac}

    while residual > tol and runtime_statistics["status"] == 1:
        result = minimize(_minfunc, orbit_.orbit_vector(), **scipy_kwargs)
        if kwargs.get("progressive", False):
            # When solving the system repeatedly, a more stringent tolerance may be required to reduce the residual
            # by a sufficient amount due to vanishing gradients.
            scipy_kwargs["tol"] /= 10.0
        next_orbit = orbit_.from_numpy_array(result.x)
        next_residual = next_orbit.residual()
        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        orbit_, runtime_statistics = _process_correction(
            orbit_,
            next_orbit,
            runtime_statistics,
            tol,
            maxiter,
            ftol,
            1,
            0,
            residual,
            next_residual,
            method,
            residual_logging=kwargs.get("residual_logging", False),
            verbose=kwargs.get("verbose", False),
        )
        residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _scipy_optimize_root_wrapper(
    orbit_, method="lgmres", maxiter=10, tol=1e-6, **kwargs
):
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
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
    except AssertionError as assrt:
        raise TypeError("tol and maxiter must be numerical scalars or list.") from assrt
    try:
        if isinstance(tol, list) or isinstance(tol, tuple):
            tol = tol.pop(0)
        if isinstance(maxiter, list) or isinstance(maxiter, tuple):
            maxiter = maxiter.pop(0)
    except IndexError as ie:
        raise IndexError(
            ": parameters for hunt need to be iterables of same length as the number of methods."
        ) from ie

    residual = orbit_.residual()
    parameter_eqn_components = kwargs.get("parameter_eqn_components", True)
    ftol = kwargs.get("ftol", 1e-9)
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "residuals": [residual],
        "maxiter": maxiter,
        "tol": tol,
        "status": 1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_)))
        print("Constraints : {}".format(orbit_.constraints))
        print("Initial residual : {}".format(orbit_.residual()))
        print("Target residual tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()

    def _rootfunc(x):
        """
        :param x0: (n,) numpy array
        :param args: time discretization, space discretization, subClass from orbit.py
        :return: value of cost functions (0.5 * L_2 norm of spatiotemporal mapping squared)
        """
        nonlocal orbit_
        nonlocal parameter_eqn_components
        x_orbit = orbit_.from_numpy_array(x, **kwargs)
        xvec = x_orbit.eqn(**kwargs).orbit_vector().ravel()
        # Need components for the parameters, but typically they will not have an associated component in the equation;
        # however, I do not think it should be by default.
        if not parameter_eqn_components:
            xvec[-len(orbit_.parameters) :] = 0
        return xvec

    def _rootjac(x):
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
        nonlocal orbit_
        x_orbit = orbit_.from_numpy_array(x)
        # gradient does not have the same problem that the equation
        return (
            x_orbit.cost_function_gradient(x_orbit.eqn(**kwargs), **kwargs)
            .orbit_vector()
            .ravel()
        )

    while residual > tol and runtime_statistics["status"] == 1:
        if method == "newton_krylov":
            scipy_kwargs = dict({"f_tol": ftol}, **kwargs.get("scipy_kwargs", {}))
            result_orbit_vector = newton_krylov(
                _rootfunc, orbit_.orbit_vector(), **scipy_kwargs
            )
        elif method == "anderson":
            scipy_kwargs = dict({"f_tol": ftol}, **kwargs.get("scipy_kwargs", {}))
            result_orbit_vector = anderson(
                _rootfunc, orbit_.orbit_vector().ravel(), **scipy_kwargs
            )

        elif method in [
            "root_anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
            "df-sane",
        ]:
            if method == "root_anderson":
                method = "anderson"
            scipy_kwargs = dict({"tol": tol}, **kwargs.get("scipy_kwargs", {}))
            # Returns an OptimizeResult, .x attribute is where array is stored.
            result = root(
                _rootfunc,
                orbit_.orbit_vector().ravel(),
                method=method,
                jac=_rootjac,
                **scipy_kwargs
            )
            result_orbit_vector = result.x
        else:
            runtime_statistics["residuals"].append(orbit_.residual())
            return orbit_, runtime_statistics
        next_orbit = orbit_.from_numpy_array(result_orbit_vector)
        next_residual = next_orbit.residual()
        # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
        orbit_, runtime_statistics = _process_correction(
            orbit_,
            next_orbit,
            runtime_statistics,
            tol,
            maxiter,
            ftol,
            1,
            0,
            residual,
            next_residual,
            method,
            residual_logging=kwargs.get("residual_logging", False),
            verbose=kwargs.get("verbose", False),
        )
        residual = next_residual
    else:
        if orbit_.residual() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["residuals"].append(orbit_.residual())
        return orbit_, runtime_statistics


def _print_exit_messages(orbit, status):
    """

    Parameters
    ----------
    orbit
    status

    Returns
    -------

    """
    if status == -1:
        print("\nConverged. Exiting with residual={}".format(orbit.residual()))
    elif status == 0:
        print("\nStalled. Exiting with residual={}".format(orbit.residual()))
    elif status == 2:
        print(
            "\nMaximum number of iterations reached."
            " exiting with residual={}".format(orbit.residual())
        )
    elif status == 3:
        print(
            "\nInsufficient residual decrease, exiting with residual={}".format(
                orbit.residual()
            )
        )
    else:
        # A general catch all for custom status flag values; happens when multiple methods given.
        print(
            "\n Optimization termination with status {} and residual {}".format(
                status, orbit.residual()
            )
        )
    return None


def _process_correction(
    orbit_,
    next_orbit_,
    runtime_statistics,
    tol,
    maxiter,
    ftol,
    step_size,
    min_step,
    residual,
    next_residual,
    method,
    residual_logging=False,
    inner_nit=None,
    verbose=False,
):
    # If the trigger that broke the while loop was step_size then assume next_residual < residual was not met.
    runtime_statistics["nit"] += 1
    if next_residual <= tol:
        runtime_statistics["status"] = -1
        return next_orbit_, runtime_statistics
    elif step_size <= min_step:
        runtime_statistics["status"] = 0
        return orbit_, runtime_statistics
    elif runtime_statistics["nit"] == maxiter:
        runtime_statistics["status"] = 2
        return next_orbit_, runtime_statistics
    elif (residual - next_residual) / max([residual, next_residual, 1]) < ftol:
        runtime_statistics["status"] = 3
        return orbit_, runtime_statistics
    else:
        if method == "adj":
            if verbose:
                if np.mod(runtime_statistics["nit"], 5000) == 0:
                    print(
                        "\n Residual={:.7f} after {} adjoint descent steps. Parameters={}".format(
                            next_residual, runtime_statistics["nit"], orbit_.parameters
                        )
                    )
                elif np.mod(runtime_statistics["nit"], 100) == 0:
                    print("#", end="")
            # Too many steps elapse for adjoint descent; this would eventually cause dramatic slow down.
            if residual_logging and np.mod(runtime_statistics["nit"], 1000) == 0:
                runtime_statistics["residuals"].append(next_residual)

        elif method == "newton_descent":
            if verbose and inner_nit is not None:
                print(
                    "Residual={} after {} inner loops, for a total Newton descent step with size {}".format(
                        next_residual, inner_nit, step_size * inner_nit
                    )
                )
            elif verbose and np.mod(runtime_statistics["nit"], 25) == 0:
                print(
                    "Residual={} after {} Newton descent steps ".format(
                        next_residual, runtime_statistics["nit"]
                    )
                )

            if residual_logging:
                runtime_statistics["residuals"].append(next_residual)
        else:
            if residual_logging:
                runtime_statistics["residuals"].append(next_residual)
            if verbose:
                if np.mod(runtime_statistics["nit"], 25) == 0:
                    print(
                        " Residual={:.7f} after {} {} iterations. Parameters={}".format(
                            next_residual,
                            runtime_statistics["nit"],
                            method,
                            orbit_.parameters,
                        )
                    )
                else:
                    print("#", end="")
            sys.stdout.flush()
        return next_orbit_, runtime_statistics
