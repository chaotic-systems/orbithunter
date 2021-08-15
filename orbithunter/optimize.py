from scipy.linalg import lstsq, pinv, solve
from scipy.optimize import minimize, root
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
import warnings

__all__ = ["hunt"]


class OrbitResult(dict):
    """
    Represents the result of applying hunt; format copied from SciPy scipy.optimize.OptimizeResult.

    Attributes
    ----------
    orbit : ndarray
        The solution of the optimization.

    status : int
        Integer which tracks the type of exit from whichever numerical algorithm was applied.

        See Notes for more details.

    message : str
        Description of the cause of the termination.

    nit : int
        Number of iterations performed by the optimizer.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the :meth:`OrbitResult.keys()` method.

    The descriptions for each value of status are as follows; -1 used during algorithms as a "has not failed yet" flag.
    -1 : Has-not-failed-yet. Flag used to track whether or not algorithm should be terminated or not.
    0 : Minimum backtracking step size requirement not met.
    1 : Converged
    2 : Maximum number of iterations reached
    3 : Insufficient cost decrease.

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


def hunt(orbit_instance, *methods, **kwargs):
    """
    Main optimization function for orbithunter; wraps many different SciPy and custom routines

    Parameters
    ----------
    orbit_instance : Orbit
        The orbit instance serving as the initial condition for optimization.

    methods : str or multiple str or tuple of str
        Represents the numerical methods to hunt with, in order of indexing. Not all methods will work for all classes,
        performance testing is part of the development process. Options include:
        'newton_descent', 'lstsq', 'solve', 'adj', 'gd', 'lsqr', 'lsmr', 'bicg', 'bicgstab', 'gmres', 'lgmres',
        'cg', 'cgs', 'qmr', 'minres', 'gcrotmk','nelder-mead', 'powell', 'cg_min', 'bfgs', 'newton-cg', 'l-bfgs-b',
        'tnc', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'hybr',
        'lm','broyden1', 'broyden2', 'linearmixing', 'diagbroyden', 'excitingmixing',
        'df-sane', 'krylov', 'anderson'

    kwargs : dict, optional
        May contain any and all extra keyword arguments required for numerical methods and Orbit specific methods.

        `factory : callable`
            Callable with signature: factory(orbit_instance, method, kwargs) that yields relevant callables
            or options for root, minimize, or sparse.linalg methods. See Notes for details.

        `maxiter : int, optional`
            The maximum number of steps; computation time can be highly dependent on this number i.e.
            maxiter=100 for adjoint descent and lstsq have very very different computational times.

        `tol : float, optional`
            The threshold for the cost function for an orbit approximation to be declared successful.

        `ftol : float, optional`
            The threshold for the decrease of the cost function for any given step.

        `min_step : float, optional`
            Smallest backtracking step size before failure; not used in minimize.optimize algorithms.

        `scipy_kwargs : dict, optional`
            Additional arguments for SciPy solvers. There are too many to describe and they depend on the
            particular algorithm utilized, see references for links for more info.
            This set of kwargs is
            copied and then passed to numerical methods. They can include options that need to be passed to methods
            such as :meth:`Orbit.eqn` during runtime.

    Returns
    -------
    OrbitResult :
        Object which includes optimization properties like exit code, costs, tol, maxiter, etc. and
        the final resulting orbit approximation.

    Notes
    -----
    Description, links, and other comments on numerical methods. Before beginning testing/exploration I
    **highly recommend** checking the options of each solver. Typically the linear algebra solvers try to solve
    $Ax=b$ within a strict error tolerance; however, because of nonlinearity we want to iteratively update this
    and solve it sequentially. Therefore, I recommend reducing the tolerance (sometimes dramatically) and use the
    orbithunter "outer iterations" (i.e. number of steps in the sequence x_n used to define and solve A_n x_n = b_n),
    to control the absolute error. Additionally, I have tried to provide a crude heuristic via the "progressive"
    keyword argument. This will crank up the strictness per every outer iteration loop, as it is presumed that more
    outer iterations means higher accuracy. Personally I just recommend to increase maxiter and keep the tolerances low.

    The other issue is that scipy has different keyword arguments for the "same thing" everywhere. For example,
    the keyword arguments to control the tolerance across the set of methods that this provides access to are
    gtol, ftol, fatol, xtol, tol, atol, btol, ...
    They do typically represent different quantities, but that doesn't make it any less confusing.

    **scipy.optimize.minimize**

    To access options of the scipy solvers, must be passed as nested dict: hunt(x, scipy_kwargs={"options":{}})

    1. Do not take jacobian information: "nelder-mead", "powell", "cobyla"
    2. Take Jacobian (product/matrix) "cg_min", "bfgs", "newton-cg", "l-bfgs-b", "tnc",  "slsqp"
    3. Methods that either require the Hessian matrix (dogleg) or some method of computing it or its product.
       "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"

    Support for Hessian based methods `['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']`.
    For the 'dogleg' method, an argument ``hess`` is required to be passed to hunt in ``scipy_kwargs``

        `hess{callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}`

    For the other mehods, ``hessp`` is sufficient, but usage of ``hess`` is viable.

        `hessp{callable, optional}`

    Alternatively, SciPy accepts the finite difference methods str '2-point', '3-point', 'cs'
    or HessianUpdateStrategy objects. The Hessian based methods have never been tested as they were never used
    with the KSe.

    Factory function returns a (callable, dict) pair. The callable is cost function C(orbit_instance, kwargs).
    The dict contains keywords "jac" and one of the following "hess", "hessp" with the relevant callables/str see
    SciPy scipy.optimize.minimize for more details

    **scipy.optimize.root**

    To access options of the scipy solvers, must be passed as nested dict: hunt(x, scipy_kwargs={"options":{}})
    To access the "jac_options" options, it is even more annoying:  hunt(x, scipy_kwargs={"options":{"jac_options":{}}})

    1. Methods that take jacobian as argument: 'hybr', 'lm'
    2. Methods which approximate the jacobian; take keyword argument "jac_options"
       `['broyden1', 'broyden2', 'anderson',  'krylov', 'df-sane']`
    3. Methods whose performance, SciPy warns, is highly dependent on the problem.
       `['linearmixing', 'diagbroyden', 'excitingmixing']`

    Factory function should return root function F(x) (Orbit.eqn()) and if 'hybr' or 'lm' then also jac as dict.

    **scipy.sparse.linalg**

    1. Methods that solve $Ax=b$ in least squares fashion : 'lsmr', 'lsqr'. Do not use preconditioning.
    2. Solves $Ax=b$ if A square (n, n) else solve normal equations $A^T A x = A^T b$ in iterative/inexact fashion:
       'minres', 'bicg', 'bicgstab', 'gmres', 'lgmres', 'cg', 'cgs', 'qmr', 'gcrotmk'. Use preconditioning.

    Factory function should return (A, b) where A is Linear operator or matrix and b is vector.

    Other factoids worth mentioning, the design choice has been made for function/callable factories
    that they should build in constants into their definitions of the callables using nonlocals within the scope
    of the factory instead of passing constants as args. The reason for this is because the latter is not allowed
    for LinearOperator (sparse linalg) methods.

    References
    ----------
    User should be aware of the existence of :func:`scipy.optimize.show_options()`
    `Options for each optimize method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html#scipy.optimize.show_options>`_
    `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_
    `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/populated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
    `scipy.optimize.root <https://docs.scipy.org/doc/scipy/reference/populated/scipy.optimize.root.html>`_
    `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_

    """
    hunt_kwargs = {k: v.copy() if hasattr(v, "copy") else v for k, v in kwargs.items()}
    # so that list.pop() method can be used, cast tuple as lists
    methods = tuple(*methods) or hunt_kwargs.pop("methods", "adj")
    scipy_kwargs = hunt_kwargs.pop("scipy_kwargs", {})

    if len(methods) == 1 and isinstance(*methods, tuple):
        methods = tuple(*methods)
    elif isinstance(methods, str):
        methods = (methods,)

    runtime_statistics = {}
    for method in methods:
        try:
            if isinstance(scipy_kwargs, list):
                # Different SciPy methods can have different keyword arguments; passing the wrong ones raises an error.
                hunt_kwargs["scipy_kwargs"] = scipy_kwargs.pop(0)
            else:
                hunt_kwargs["scipy_kwargs"] = scipy_kwargs
        except IndexError as ie:
            raise IndexError(
                f"Providing keyword arguments for multiple methods requires a dict for each method,"
                f" even if empty. This avoids accidentally passing incorrect keyword arguments"
            ) from ie
        if method == "newton_descent":
            orbit_instance, method_statistics = _newton_descent(
                orbit_instance, **hunt_kwargs
            )
        elif method == "lstsq":
            # solves Ax = b in least-squares manner
            orbit_instance, method_statistics = _lstsq(orbit_instance, **hunt_kwargs)
        elif method == "solve":
            # solves Ax = b
            orbit_instance, method_statistics = _solve(orbit_instance, **hunt_kwargs)
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
            orbit_instance, method_statistics = _scipy_sparse_linalg_solver_wrapper(
                orbit_instance, method=method, **hunt_kwargs
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
            elif method in [
                "trust-constr",
                "dogleg",
                "trust-ncg",
                "trust-exact",
                "trust-krylov",
            ] and (
                not hasattr(orbit_instance, "costhessp")
                and not hasattr(orbit_instance, "costhess")
            ):
                err_str = "".join(
                    [
                        f"Hessian based algorithm {method} is not supported for {orbit_instance.__class__}",
                        f" because neither {orbit_instance.__class__}.costhess() nor",
                        f" {orbit_instance.__class__}.costhessp() are defined"
                        f" and no SciPy option was passed. See scipy.optimize.minimize docs for details.",
                    ]
                )
                raise AttributeError(err_str)
            orbit_instance, method_statistics = _scipy_optimize_minimize_wrapper(
                orbit_instance, method=method, **hunt_kwargs
            )
        elif method in [
            "hybr",
            "lm",
            "broyden1",
            "broyden2",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
            "df-sane",
            "anderson",
        ]:
            orbit_instance, method_statistics = _scipy_optimize_root_wrapper(
                orbit_instance, method=method, **hunt_kwargs
            )
        elif method == "gd":
            orbit_instance, method_statistics = _gradient_descent(
                orbit_instance, **hunt_kwargs
            )
        else:
            orbit_instance, method_statistics = _adjoint_descent(
                orbit_instance, **hunt_kwargs
            )

        # If the "total" runtime_statistics is empty then initialize with the previous method statistics.
        if not runtime_statistics:
            runtime_statistics = method_statistics
        else:
            # Keep a consistent order of the runtime information for combining statistics from multiple methods;
            # collections of scalar values converted to lists upon combination.
            for key in sorted({**runtime_statistics, **method_statistics}.keys()):
                if key == "status":
                    runtime_statistics[key] = method_statistics.get("status", -1)
                elif isinstance(runtime_statistics.get(key, []), list):
                    method_values = method_statistics.get(key)
                    if isinstance(method_values, list):
                        runtime_statistics.get(key, []).extend(method_values)
                    else:
                        runtime_statistics.get(key, []).append(method_values)
                else:
                    runtime_statistics[key] = [
                        runtime_statistics[key],
                        method_statistics.get(key),
                    ]

    # Collection of tolerances can be passed if multiple methods; take the "final" tolerance to be the strictest.
    final_tol = kwargs.get("tol", 1e-6)
    if isinstance(final_tol, list):
        final_tol = min(final_tol)

    final_status = runtime_statistics["status"]
    if final_status == -1 and orbit_instance.cost() <= final_tol:
        # If exited numerical methods with "has not failed yet" value, and mets tolerance requirements, then success!
        runtime_statistics["status"] = 1
    runtime_statistics["message"] = _exit_messages(
        orbit_instance,
        runtime_statistics["status"],
        verbose=kwargs.get("verbose", False),
    )
    return OrbitResult(orbit=orbit_instance, **runtime_statistics)


def _adjoint_descent(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=10000,
    min_step=1e-6,
    preconditioning=False,
    **kwargs,
):
    """
    Adjoint descent method for numerical optimization of Orbit instances.

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    Preconditioning arguments never seen outside rmatvec/costgrad/etc.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            preconditioning = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    runtime_statistics = {
        "method": "adj",
        "nit": 0,
        "costs": [orbit_instance.cost()],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    verbose = kwargs.get("verbose", False)
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting adjoint descent")
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1

    F = orbit_instance.eqn(**kwargs)
    cost = 0.5 * F.dot(F)
    step_size = kwargs.get("step_size", 1)

    kwargs = {**kwargs, "preconditioning": preconditioning}
    while cost > tol and runtime_statistics["status"] == -1:
        # Calculate the step
        gradient = orbit_instance.costgrad(F, **kwargs)
        # Negative sign -> 'descent'
        next_orbit_instance = orbit_instance.increment(
            gradient, step_size=-1.0 * step_size
        )
        # Calculate the mapping and store; need it for next step and to compute cost.
        next_F = next_orbit_instance.eqn(**kwargs)
        # Compute cost to see if step succeeded
        next_cost = 0.5 * next_F.dot(next_F)
        while next_cost >= cost and step_size > min_step:
            # reduce the step size until minimum is reached or cost decreases.
            step_size /= 2
            next_orbit_instance = orbit_instance.increment(
                gradient, step_size=-1.0 * step_size
            )
            # Calculate the mapping and store; need it for next step and to compute cost.
            next_F = next_orbit_instance.eqn(**kwargs)
            # Compute cost to see if step succeeded
            next_cost = 0.5 * next_F.dot(next_F)
        else:
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                cost,
                next_cost,
                "adj",
                verbose=verbose,
                cost_logging=kwargs.get("cost_logging", False),
            )
            F = next_F
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _gradient_descent(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=10000,
    min_step=1e-6,
    preconditioning=False,
    **kwargs,
):
    """
    Gradient descent method for numerical optimization of Orbit instances.

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    kwargs : dict
        Keyword arguments relevant for cost and costgrad methods

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----

    .. warning::
       If the cost function equals $1/2 F^2$ this does the same exact thing as 'adj', but slower. This method is written to work
       for general cost functions; the generalization leads to a decrease in speed because fewer assumptions can be
       made.


    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            kwargs["preconditioning"] = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    runtime_statistics = {
        "method": "gd",
        "nit": 0,
        "costs": [orbit_instance.cost()],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    verbose = kwargs.get("verbose", False)
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting adjoint descent")
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1
    kwargs = {**kwargs, "preconditioning": preconditioning}
    cost = orbit_instance.cost()
    step_size = kwargs.get("step_size", 1)

    while cost > tol and runtime_statistics["status"] == -1:
        # Calculate the step
        gradient = orbit_instance.costgrad()
        # Negative sign -> 'descent'
        next_orbit_instance = orbit_instance.increment(
            gradient, step_size=-1.0 * step_size
        )
        # Compute cost to see if step succeeded
        next_cost = next_orbit_instance.cost()
        while next_cost >= cost and step_size > min_step:
            # reduce the step size until minimum is reached or cost decreases.
            step_size /= 2
            next_orbit_instance = orbit_instance.increment(
                gradient, step_size=-1.0 * step_size
            )
            next_cost = next_orbit_instance.cost()
        else:
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                cost,
                next_cost,
                "gd",
                verbose=verbose,
                cost_logging=kwargs.get("cost_logging", False),
            )
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _newton_descent(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=500,
    min_step=1e-6,
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of the Newton-descent method with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    See Also
    --------
    `Newton Descent <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.046206>`_

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            kwargs["preconditioning"] = preconditioning.pop(0)

    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    step_size = kwargs.get("step_size", 1.0)
    cost = orbit_instance.cost()
    runtime_statistics = {
        "method": "newton_descent",
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting Newton descent optimization")
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()

    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1
    preconditioner_factory = kwargs.get("mfactory", None)
    M = None
    mapping = orbit_instance.eqn(**kwargs)
    cost = mapping.cost(eqn=False)
    while cost > tol and runtime_statistics["status"] == -1:
        step_size = kwargs.get("step_size", 1)
        # Solve A dx = b <--> J dx = - f, for dx.
        A, b = orbit_instance.jacobian(**kwargs), -1 * mapping.state.ravel()
        if preconditioning:
            if callable(preconditioner_factory):
                M = preconditioner_factory(orbit_instance, "lstsq", **kwargs)
            elif hasattr(orbit_instance, "preconditioner"):
                M = orbit_instance.preconditioner(**kwargs)
            elif runtime_statistics["nit"] == 0.0 and M is None:
                warn_str = "".join(
                    [
                        f"\norbithunter.optimize.hunt was passed preconditioning=True but no method of",
                        f" computing a preconditioner was provided.",
                    ]
                )
                warnings.warn(warn_str, RuntimeWarning)
                # identity just returns whatever we started with
                eye = lambda x: x
                M = LinearOperator(
                    (A.shape[1], A.shape[1]),
                    matvec=eye,
                    rmatvec=eye,
                    matmat=eye,
                    rmatmat=eye,
                )

            # Right preconditioning by using transpose. A*M = (M^T A^T)^T (only have left hand multiplication for M, MT)
            A = (M.rmatmat(A.T)).T

        inv_A = pinv(A)
        # get the vector solution
        dx = np.dot(inv_A, b)

        # get the solution to Px = x'
        if preconditioning and M is not None:
            dx = M.matvec(dx)
        dx = orbit_instance.from_numpy_array(dx)
        next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
        next_mapping = next_orbit_instance.eqn(**kwargs)
        next_cost = next_mapping.cost(eqn=False)
        # This modifies the step size if too large; i.e. its a very crude way of handling curvature.
        while next_cost > cost and step_size > min_step:
            # Continues until either step is too small or cost decreases
            step_size /= 2.0
            next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
            next_mapping = next_orbit_instance.eqn(**kwargs)
            next_cost = next_mapping.cost(eqn=False)
        else:
            inner_nit = 1
            if kwargs.get("approximation", False):
                # Re-use the same pseudoinverse for many inexact solutions to dx_n = - A^+(x) F(x + dx_{n-1})
                b = -1 * next_mapping.state.ravel()
                dx = orbit_instance.from_numpy_array(np.dot(inv_A, b))
                if preconditioning and M is not None:
                    dx = M.matvec(dx)
                inner_orbit = next_orbit_instance.increment(dx, step_size=step_size)
                inner_mapping = inner_orbit.eqn(**kwargs)
                inner_cost = inner_mapping.cost(eqn=False)
                while inner_cost < next_cost:
                    next_orbit_instance = inner_orbit
                    next_mapping = inner_mapping
                    next_cost = inner_cost
                    b = -1 * next_mapping.state.ravel()
                    dx = orbit_instance.from_numpy_array(np.dot(inv_A, b))
                    inner_orbit = next_orbit_instance.increment(dx, step_size=step_size)
                    inner_mapping = inner_orbit.eqn(**kwargs)
                    inner_cost = inner_mapping.cost(eqn=False)
                    inner_nit += 1
                    if inner_cost < tol:
                        next_orbit_instance = inner_orbit
                        next_mapping = inner_mapping
                        next_cost = inner_cost
                        break
            # If the trigger that broke the while loop was step_size then
            # assume next_cost < cost was not met.
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                cost,
                next_cost,
                "newton_descent",
                inner_nit=inner_nit,
                cost_logging=kwargs.get("cost_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _lstsq(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=500,
    min_step=1e-6,
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of scipy.linalg.lstsq solvers with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    Allows for preconditioning but only if preconditioner has rmatmat defined.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]

    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            preconditioning = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    mapping = orbit_instance.eqn(**kwargs)
    cost = mapping.cost(eqn=False)
    runtime_statistics = {
        "method": "lstsq",
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting lstsq optimization")
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )

    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1

    preconditioner_factory = kwargs.get("mfactory", None)
    M = None
    while cost > tol and runtime_statistics["status"] == -1:
        step_size = kwargs.get("step_size", 1)

        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_instance.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)
        if preconditioning:
            if callable(preconditioner_factory):
                M = preconditioner_factory(orbit_instance, "lstsq", **kwargs)
            elif hasattr(orbit_instance, "preconditioner"):
                M = orbit_instance.preconditioner(**kwargs)
            else:
                if runtime_statistics["nit"] == 0.0:
                    warn_str = "".join(
                        [
                            f"\norbithunter.optimize.hunt was passed preconditioning=True but no method of",
                            f" computing a preconditioner was provided.",
                        ]
                    )
                    warnings.warn(warn_str, RuntimeWarning)
                # identity just returns whatever we started with
                eye = lambda x: x
                M = LinearOperator(
                    (A.shape[1], A.shape[1]),
                    matvec=eye,
                    rmatvec=eye,
                    matmat=eye,
                    rmatmat=eye,
                )

            # Right preconditioning by using transpose. A*M = (M^T A^T)^T (only have left hand multiplication for M, MT)
            A = (M.rmatmat(A.T)).T

        dx = lstsq(A, b)[0]
        if preconditioning and M is not None:
            dx = M.matvec(dx)
        dx = orbit_instance.from_numpy_array(dx, **kwargs)
        next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
        next_mapping = next_orbit_instance.eqn(**kwargs)
        next_cost = next_mapping.cost(eqn=False)
        while next_cost > cost and step_size > min_step:
            # Continues until either step is too small or cost decreases
            step_size /= 2.0
            next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
            next_mapping = next_orbit_instance.eqn(**kwargs)
            next_cost = next_mapping.cost(eqn=False)
        else:
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                cost,
                next_cost,
                "lstsq",
                cost_logging=kwargs.get("cost_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _solve(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=10000,
    min_step=1e-6,
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of scipy.linalg.solve with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    Allows for preconditioning if rmatmat defined.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            preconditioning = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    # This is to handle the case where method == 'hybrid' such that different defaults are used.
    mapping = orbit_instance.eqn(**kwargs)
    cost = mapping.cost(eqn=False)
    runtime_statistics = {
        "method": "solve",
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting solve optimization")
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )

    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1

    preconditioner_factory = kwargs.get("mfactory", None)
    M = None

    while cost > tol and runtime_statistics["status"] == -1:
        step_size = kwargs.get("step_size", 1)

        # Solve A dx = b <--> J dx = - f, for dx.
        A = orbit_instance.jacobian(**kwargs)
        b = -1.0 * mapping.state.reshape(-1, 1)

        if preconditioning:
            if callable(preconditioner_factory):
                M = preconditioner_factory(orbit_instance, "solve", **kwargs)
            elif hasattr(orbit_instance, "preconditioner"):
                M = orbit_instance.preconditioner(**kwargs)
            else:
                if runtime_statistics["nit"] == 0.0:
                    warn_str = "".join(
                        [
                            f"\norbithunter.optimize.hunt was passed preconditioning=True but no method of",
                            f" computing a preconditioner was provided.",
                        ]
                    )
                    warnings.warn(warn_str, RuntimeWarning)
                # identity just returns whatever we started with
                eye = lambda x: x
                M = LinearOperator(
                    (A.shape[1], A.shape[1]),
                    matvec=eye,
                    rmatvec=eye,
                    matmat=eye,
                    rmatmat=eye,
                )

            # Right preconditioning by using transpose. A*M = (M^T A^T)^T (only have left hand multiplication for M, MT)
            A = (M.rmatmat(A.T)).T

        dx = solve(A, b)
        if preconditioning and M is not None:
            dx = M.matvec(dx)
        dx = orbit_instance.from_numpy_array(dx, **kwargs)
        next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
        next_mapping = next_orbit_instance.eqn(**kwargs)
        next_cost = next_mapping.cost(eqn=False)
        while next_cost > cost and step_size > min_step:
            # Continues until either step is too small or cost decreases
            step_size /= 2.0
            next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
            next_mapping = next_orbit_instance.eqn(**kwargs)
            next_cost = next_mapping.cost(eqn=False)
        else:
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,
                min_step,
                cost,
                next_cost,
                "solve",
                cost_logging=kwargs.get("cost_logging", False),
                verbose=kwargs.get("verbose", False),
            )
            mapping = next_mapping
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _scipy_sparse_linalg_solver_wrapper(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=10000,
    min_step=1e-6,
    method="lsmr",
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of scipy.sparse.linalg and scipy.linalg solvers with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    method : str
        The numerical method to be used as the solver from scipy.sparse.linalg
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    Currently the deployment of rmatvec assumes that the eqn() and state have the same number of components,
    i.e. there is no "parameter component" of the equation. This may not always be the case, but the argument is that
    if there is a component in the governing equation then the variable should be included in the state attribute
    and not the parameter attribute; will likely need to revist in the future.

    This function wraps the following routines. They are partitioned by the functions that they use if not require.

    Methods that this function form two main categories.

    Solves $Ax=b$ in least squares fashion :
        lsmr, lsqr

    Solves $Ax=b$ if A square (n, n) else solve normal equations $A^T A x = A^T b$ in iterative/inexact fashion.
        minres, bicg, bicgstab, gmres, lgmres, cg, cgs, qmr, gcrotmk

    ``scipy_kwargs`` is passed instead of ``kwargs`` because of conflicts between ``tol`` and ``maxiter`` keywords.

    Preconditioning only built-in for methods not including lsmr, lsqr; this can be circumvented by passing a
    factory function for the matrix A* where A* = AM or MA or whatever preconditioning strategy the user desires.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(min_step) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(min_step, list):
            min_step = min_step.pop(0)
        if isinstance(preconditioning, list):
            preconditioning = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    cost = orbit_instance.cost()
    if kwargs.get("verbose", False):
        print(
            "\n------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print("Preconditioning : {}".format(preconditioning))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    # Simplest solution to exclusion of simple backtracking.
    if not kwargs.get("backtracking", True):
        min_step = 1
    linear_system_factory = kwargs.get("factory", None) or _sparse_linalg_factory
    preconditioner_factory = kwargs.get("mfactory", None)
    scipy_kwargs = kwargs.get("scipy_kwargs", {})
    while cost > tol and runtime_statistics["status"] == -1:
        step_size = kwargs.get("step_size", 1)
        A, b = linear_system_factory(orbit_instance, method, **kwargs)
        if method in ["lsmr", "lsqr"]:
            if method == "lsmr":
                result_tuple = lsmr(A, b, **scipy_kwargs)
            else:
                result_tuple = lsqr(A, b, **scipy_kwargs)
        else:
            if preconditioning:
                if callable(preconditioner_factory):
                    scipy_kwargs["M"] = preconditioner_factory(
                        orbit_instance, method, **kwargs
                    )
                elif hasattr(orbit_instance, "preconditioner"):
                    scipy_kwargs["M"] = orbit_instance.preconditioner(**kwargs)
                elif runtime_statistics["nit"] == 0.0:
                    warn_str = "".join(
                        [
                            f"\norbithunter.optimize.hunt was passed preconditioning=True but no method of",
                            f" computing a preconditioner was provided.",
                        ]
                    )
                    warnings.warn(warn_str, RuntimeWarning)

            if method == "minres":
                result_tuple = (minres(A, b, **scipy_kwargs),)
            elif method == "bicg":
                result_tuple = (bicg(A, b, **scipy_kwargs),)
            elif method == "bicgstab":
                result_tuple = (bicgstab(A, b, **scipy_kwargs),)
            elif method == "gmres":
                result_tuple = (gmres(A, b, **scipy_kwargs),)
            elif method == "lgmres":
                result_tuple = (lgmres(A, b, **scipy_kwargs),)
            elif method == "cg":
                result_tuple = (cg(A, b, **scipy_kwargs),)
            elif method == "cgs":
                result_tuple = (cgs(A, b, **scipy_kwargs),)
            elif method == "qmr":
                # M not accepted.
                if "M" in scipy_kwargs:
                    warnings.warn(
                        f"qmr requires left/right M1, M2 and not just a single M preconditioner",
                        RuntimeWarning,
                    )
                    scipy_kwargs["M2"] = scipy_kwargs.pop("M")
                    scipy_kwargs["M1"] = scipy_kwargs["M2"]
                result_tuple = (qmr(A, b, **scipy_kwargs),)
            elif method == "gcrotmk":
                result_tuple = (gcrotmk(A, b, **scipy_kwargs),)
            else:
                raise ValueError("Unknown solver %s" % method)

        if len(result_tuple) == 1:
            x = tuple(*result_tuple)[0]
        else:
            x = result_tuple[0]

        dx = orbit_instance.from_numpy_array(x, **kwargs)
        next_orbit_instance = orbit_instance.increment(dx)
        next_mapping = next_orbit_instance.eqn(**kwargs)
        next_cost = next_mapping.cost(eqn=False)
        while next_cost > cost and step_size > min_step:
            # Continues until either step is too small or cost decreases
            step_size /= 2.0
            next_orbit_instance = orbit_instance.increment(dx, step_size=step_size)
            next_mapping = next_orbit_instance.eqn(**kwargs)
            next_cost = next_mapping.cost(eqn=False)
        else:
            # If the trigger that broke the while loop was step_size then assume next_cost < cost was not met.
            orbit_instance, cost, runtime_statistics = _process_correction(
                orbit_instance,
                next_orbit_instance,
                runtime_statistics,
                tol,
                maxiter,
                ftol,
                step_size,  # step_size; no backtracking
                min_step,  # minstep=step_size
                cost,
                next_cost,
                method,
                cost_logging=kwargs.get("cost_logging", False),
                verbose=kwargs.get("verbose", False),
            )
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _scipy_optimize_minimize_wrapper(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=1,
    method="cg",
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of scipy.optimize.minimize with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    method : str
        The numerical method to be used by the SciPy function `root`
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    This function is written as though the cost function, its jacobian and hessian are to be evaluated
    at `x` which is the vector of independent variables, not at the current `orbit_instance`

    Any preconditioning/rescaling shold be built into the matvec/rmatvec/cost/costgrad methods.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol and maxiter must be scalars or possibly list when multiple methods provided"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(preconditioning, list):
            preconditioning = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    cost = orbit_instance.cost()
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }

    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()

    scipy_kwargs = kwargs.get("scipy_kwargs", {})
    func_jac_hess_factory = kwargs.get("factory", None) or _minimize_callable_factory
    while cost > tol and runtime_statistics["status"] == -1:
        # jacobian and hessian passed as keyword arguments to scipy not arguments.
        minfunc, jac_and_hess_options = func_jac_hess_factory(
            orbit_instance, method, **{**kwargs, "preconditioning": preconditioning}
        )
        result = minimize(
            minfunc,
            orbit_instance.cdof().ravel(),
            **{**scipy_kwargs, "method": method, **jac_and_hess_options},
        )

        if kwargs.get("progressive", False):
            # When solving the system repeatedly, a more stringent tolerance may be required to reduce the cost
            # by a sufficient amount due to vanishing gradients.
            scipy_kwargs["tol"] /= 2.0

        next_orbit_instance = orbit_instance.from_numpy_array(
            result.x, *orbit_instance.constants()
        )
        next_cost = next_orbit_instance.cost()
        # If the trigger that broke the while loop was step_size then assume next_cost < cost was not met.
        orbit_instance, cost, runtime_statistics = _process_correction(
            orbit_instance,
            next_orbit_instance,
            runtime_statistics,
            tol,
            maxiter,
            ftol,
            1,
            0,
            cost,
            next_cost,
            method,
            cost_logging=kwargs.get("cost_logging", False),
            verbose=kwargs.get("verbose", False),
        )
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _scipy_optimize_root_wrapper(
    orbit_instance,
    tol=1e-6,
    ftol=1e-9,
    maxiter=10,
    method="krylov",
    preconditioning=False,
    **kwargs,
):
    """
    Wrapper that allows usage of scipy.optimize.root with Orbit objects

    Parameters
    ----------
    orbit_instance : Orbit
        Current Orbit iterate.
    tol : float
        The threshold for termination of the numerical method
    maxiter : int
        The maximum number of iterations
    method : str
        The numerical method to be used by the SciPy function `root`
    kwargs : dict
        Typically contains keyword arguments relevant to SciPy function. Too many to list to be useful; see
        `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html>` for more details

    Returns
    -------
    orbit_instance, runtime_statistics : Orbit, dict
        The result of the numerical optimization and its statistics

    Notes
    -----
    Because of how scipy.optimize.root is setup; The provided equations F=0 need to have a component for each parameter;
    because these typically do not exist for the dimensions of configuration space. Because root is trying to solve
    F=0, it onyl makes sense to append zeros to F until the requisite dimension is reached. This is only to give access
    to the root function; it is likely a bad idea numerically to even use the root function in this instance.

    Preconditioning: 'hybr' and 'lm' will take Jacobian as an array; this can be preconditioned manually. For 'krylov',
    the jacobian is not passed at all, but jac_options can be passed in options. Within jac_options, a preconditioner
    can be passed using 'inner_M', however, to make it adaptive (i.e. dependent on current state or parameters),
    it is much more hassle than worth right now, and so I leave this one thing in the user's hands.

    """
    try:
        assert type(tol) in [int, float, list, np.float64, np.int32]
        assert type(ftol) in [int, float, list, np.float64, np.int32]
        assert type(maxiter) in [int, float, list, np.float64, np.int32]
        assert type(preconditioning) in [bool, list]
    except AssertionError as assrt:
        raise TypeError(
            "tol, ftol, min_step and maxiter must be scalar numbers (or list of scalars if multiple methods provided)"
        ) from assrt

    try:
        if isinstance(tol, list):
            tol = tol.pop(0)
        if isinstance(ftol, list):
            ftol = ftol.pop(0)
        if isinstance(maxiter, list):
            maxiter = maxiter.pop(0)
        if isinstance(preconditioning, list):
            kwargs["preconditioning"] = preconditioning.pop(0)
    except IndexError as ie:
        errstr = "".join(
            [
                f": (tol, ftol, maxiter, min_step, preconditioning) need to be either scalar (or bool)",
                f" or list of same length as number of methods.",
            ]
        )
        raise IndexError(errstr) from ie

    cost = orbit_instance.cost()
    runtime_statistics = {
        "method": method,
        "nit": 0,
        "costs": [cost],
        "maxiter": maxiter,
        "tol": tol,
        "status": -1,
    }
    if kwargs.get("verbose", False):
        print(
            "\n-------------------------------------------------------------------------------------------------"
        )
        print("Starting {} optimization".format(method))
        print("Initial guess : {}".format(repr(orbit_instance)))
        print("Constraints : {}".format(orbit_instance.constraints))
        print("Initial cost : {}".format(orbit_instance.cost()))
        print("Target cost tolerance : {}".format(tol))
        print("Maximum iteration number : {}".format(maxiter))
        print(
            "-------------------------------------------------------------------------------------------------"
        )
        sys.stdout.flush()
    kwargs = {**kwargs, "preconditioning": preconditioning}
    func_jac_factory = kwargs.get("factory", None) or _root_callable_factory
    while cost > tol and runtime_statistics["status"] == -1:
        # Use factory function to produce two callables required for SciPy routine. Need to be included under
        # the while statement so they are updated.
        _rootfunc, jac_and_jac_options = func_jac_factory(
            orbit_instance, method, **{**kwargs, "preconditioning": preconditioning}
        )
        scipy_kwargs = {
            "tol": tol,
            **kwargs.get("scipy_kwargs", {}),
            "method": method,
            **jac_and_jac_options,
        }
        # Returns an OptimizeResult, .x attribute is where array is stored.
        result = root(_rootfunc, orbit_instance.cdof().ravel(), **scipy_kwargs)
        next_orbit_instance = orbit_instance.from_numpy_array(
            result.x, *orbit_instance.constants()
        )
        next_cost = next_orbit_instance.cost()
        # If the trigger that broke the while loop was step_size then assume next_cost < cost was not met.
        orbit_instance, cost, runtime_statistics = _process_correction(
            orbit_instance,
            next_orbit_instance,
            runtime_statistics,
            tol,
            maxiter,
            ftol,
            1,
            0,
            cost,
            next_cost,
            method,
            cost_logging=kwargs.get("cost_logging", False),
            verbose=kwargs.get("verbose", False),
        )
    else:
        if orbit_instance.cost() <= tol:
            runtime_statistics["status"] = -1
        runtime_statistics["costs"].append(orbit_instance.cost())
        return orbit_instance, runtime_statistics


def _sparse_linalg_factory(orbit_instance, method, **kwargs):
    """
    Function to that produces callables evaluated at current Orbit state for root function $F$ only.

    Parameters
    ----------
    orbit_instance : Orbit
        The state to use to construct the root function.

    kwargs : dict
        Keyword arguments for evaluation of governing equations e.g. :meth:`orbithunter.core.Orbit.eqn()`

    Returns
    -------
    callable, callable
        The functions which take NumPy arrays of size Orbit.orbit_vector.size and return F and grad F, respectively.
        In other words, the first callable evaluates the governing equations using NumPy array input, and the second
        callable evaluates the Jacobian using NumPy array input.

    Notes
    -----
    Potential preconditioning happens external to this callable factory. Look at 'mfactory' and 'preconditioning'
    keyword arguments.

    """
    # Because this function is only called once per outer iteration this incurs minimal cost unless
    # eqn is very expensive to evaluate; however, in that case, the sparse linalg algorithms would be a very poor choice
    degrees_of_freedom = orbit_instance.cdof().size
    # If least squares routine, need LinearOperator representing Jacobian
    if method in ["lsqr", "lsmr"]:

        def matvec_func(v):
            # matvec needs state and parameters from v.
            nonlocal orbit_instance
            nonlocal kwargs
            v_orbit = orbit_instance.from_numpy_array(v)
            return orbit_instance.matvec(v_orbit, **kwargs).state.reshape(-1, 1)

        def rmatvec_func(v):
            # in this case, v is only state information, if rectangular system
            nonlocal orbit_instance
            nonlocal kwargs
            # The rmatvec typically requires state information from v and parameters from current instance.
            v_orbit = orbit_instance.from_numpy_array(v, *orbit_instance.constants())
            return orbit_instance.rmatvec(v_orbit, **kwargs).cdof().reshape(-1, 1)

        linear_operator_shape = (
            orbit_instance.state.size,
            degrees_of_freedom,
        )
        A = LinearOperator(
            linear_operator_shape,
            matvec_func,
            rmatvec=rmatvec_func,
            dtype=orbit_instance.state.dtype,
        )
        b = -1.0 * orbit_instance.eqn(**kwargs).state.reshape(-1, 1)
        return A, b
    elif degrees_of_freedom != orbit_instance.eqn().size or method in ["cg", "cgs"]:
        # Solving `normal equations, A^T A x = A^T b. A^T A is its own transpose hence matvec_func=rmatvec_func
        # in this instance. The matrix is evaluated at the current orbit state; i.e. it is constant with
        # respect to the optimization routines.
        def matvec_(v):
            nonlocal orbit_instance
            nonlocal kwargs
            # matvec needs parameters and orbit info from v; evaluation of matvec should take instance
            # parameters from orbit_instance; is this reasonable to expect to work.
            v_orbit = orbit_instance.from_numpy_array(v)
            return (
                orbit_instance.rmatvec(
                    orbit_instance.matvec(v_orbit, **kwargs), **kwargs
                )
                .cdof()
                .reshape(-1, 1)
            )

        # Currently only allows solving of the normal equations. A^T A x = A^T b, b = -F
        normal_equations_operator_shape = (
            degrees_of_freedom,
            degrees_of_freedom,
        )
        # Your IDE might tell you the following has "unexpected arguments" but that's only because the signature
        # of LinearOperator does not match the signature of the _CustomLinearOperator class that it actually
        # calls when user provides callables and not a 2-d array.
        ATA = LinearOperator(
            normal_equations_operator_shape,
            matvec=matvec_,
            rmatvec=matvec_,
            dtype=orbit_instance.state.dtype,
        )
        ATb = (
            orbit_instance.rmatvec(-1.0 * orbit_instance.eqn(**kwargs), **kwargs)
            .cdof()
            .reshape(-1, 1)
        )
        return ATA, ATb
    else:
        # If square system of equations, then likely solving Ax = b, not the normal equations; matvec and
        # rmatvec are allowed to be distinct.
        def matvec_(v):
            # _orbit_vector_to_orbit turns state vector into class object.
            nonlocal orbit_instance
            nonlocal kwargs
            v_orbit = orbit_instance.from_numpy_array(v)
            return orbit_instance.matvec(v_orbit, **kwargs).cdof().reshape(-1, 1)

        def rmatvec_(v):
            # _orbit_vector_to_orbit turns state vector into class object.
            nonlocal orbit_instance
            nonlocal kwargs
            v_orbit = orbit_instance.from_numpy_array(v, *orbit_instance.constants())
            return orbit_instance.rmatvec(v_orbit, **kwargs).cdof().reshape(-1, 1)

        linear_operator_shape = (
            degrees_of_freedom,
            degrees_of_freedom,
        )
        # Your IDE might tell you the following has "unexpected arguments" but that's only because the signature
        # of LinearOperator does not match the signature of the _CustomLinearOperator class that it actually
        # calls when user provides callables and not a 2-d array.
        A = LinearOperator(
            linear_operator_shape,
            matvec_,
            rmatvec=rmatvec_,
            dtype=orbit_instance.state.dtype,
        )
        b = -1.0 * orbit_instance.eqn(**kwargs).cdof().reshape(-1, 1)
        return A, b


def _minimize_callable_factory(orbit_instance, method, **kwargs):
    """
    Function to that produces callables evaluated at current Orbit state for root function $F$ only.

    Parameters
    ----------
    orbit_instance : Orbit
        The state to use to construct the root function.

    kwargs : dict
        Keyword arguments for evaluation of governing equations e.g. :meth:`orbithunter.core.Orbit.eqn()`

    Returns
    -------
    callable, callable
        The functions which take NumPy arrays of size Orbit.orbit_vector.size and return F and grad F, respectively.
        In other words, the first callable evaluates the governing equations using NumPy array input, and the second
        callable evaluates the Jacobian using NumPy array input.

    Notes
    -----
    Unfortunately either hessp or hess keyword arguments can be passed to scipy, and so the hessian information
    is returned as a dict.

    """
    hess_strategy = kwargs.get("hess_strategy", None)
    jac_strategy = kwargs.get("jac_strategy", "costgrad")
    jac_and_hess_options = {}

    def _minfunc(x):
        """
        Function which returns evaluation of cost

        Parameters
        ----------
        x : ndarray
            array of same shape as orbit_vector of current instance.

        Returns
        -------
        xvec : ndarray
            The orbit vector equivalent to `Orbit(x).eqn().orbit_vector`

        """
        nonlocal orbit_instance
        nonlocal kwargs
        x_orbit = orbit_instance.from_numpy_array(
            x, *orbit_instance.constants(), **kwargs
        )
        return x_orbit.cost()

    # Jacobian defaults to costgrad method, but could be provided as either a string for finite difference approximation
    # or other options.
    if method not in ["nelder-mead", "powell", "cobyla"]:
        if jac_strategy == "costgrad":

            def _minjac(x):
                """
                The jacobian of the cost function (scalar) can be expressed as a matrix-vector product

                Parameters
                ----------
                x : ndarray
                    array of same shape as orbit_vector of current instance.

                Returns
                -------
                xvec : ndarray
                    The orbit vector equivalent to `Orbit(x).eqn().orbit_vector`

                Notes
                -----
                The gradient of 1/2 F^2 = J^T F, rmatvec is a function which does this matrix vector product

                """
                nonlocal orbit_instance
                nonlocal kwargs
                x_orbit = orbit_instance.from_numpy_array(
                    x, *orbit_instance.constants(), **kwargs
                )
                # For cases when costgrad does not require eqn
                return x_orbit.costgrad().cdof().ravel()

            jac_ = _minjac
        else:
            jac_ = jac_strategy
        jac_and_hess_options["jac"] = jac_

    if method in ["trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]:
        if hess_strategy == "costhessp":
            # Use the class' definition of costhessp
            def _minhessp(x, p):
                """
                Returns the matrix-vector product with the cost function's Hessian. H(x) * p
                """
                nonlocal orbit_instance
                nonlocal kwargs
                x_orbit = orbit_instance.from_numpy_array(
                    x, *orbit_instance.constants()
                )
                p_orbit = orbit_instance.from_numpy_array(p)
                return x_orbit.costhessp(p_orbit, **kwargs).cdof().ravel()

            hess_dict = {"hessp": _minhessp}

        elif hess_strategy == "costhess":
            # Use the class' definition of costhess

            def _minhessfunc(x):
                """
                Returns the matrix-vector product with the cost function's Hessian.
                """
                nonlocal orbit_instance
                nonlocal kwargs
                x_orbit = orbit_instance.from_numpy_array(
                    x, *orbit_instance.constants()
                )
                return x_orbit.costhess(**kwargs)

            hess_dict = {"hess": _minhessfunc}
        elif hess_strategy is not None:
            hess_dict = {"hess": hess_strategy}
        else:
            raise ValueError(
                "invalid value for hess_strategy keyword argument. Check and try again"
            )
        jac_and_hess_options = {**jac_and_hess_options, **hess_dict}

    # Jacobian and hessian callables/infomation passed as keyword arguments to minimize function; might
    # as well store as dict.
    return _minfunc, jac_and_hess_options


def _root_callable_factory(orbit_instance, method, **kwargs):
    """
    Function to that produces callables evaluated at current Orbit state for root function $F$ only.

    Parameters
    ----------
    orbit_instance : Orbit
        The state to use to construct the root function.

    kwargs : dict
        Keyword arguments for evaluation of governing equations e.g. :meth:`orbithunter.core.Orbit.eqn()`

    Returns
    -------
    callable, dict
        The functions which take NumPy arrays of size Orbit.orbit_vector.size and return F and grad F, respectively.
        In other words, the first callable evaluates the governing equations using NumPy array input, and the second
        callable evaluates the Jacobian using NumPy array input.

    """

    def _rootfunc(x):
        """
        Function which returns evaluation of equations using `orbit_vector` x

        Returns
        -------
        xvec : ndarray
            Vector containing F(x), padded with as many zeros as necessary to make this vector have as many dimensions
            as orbit_instance.orbit_vector().size

        """
        nonlocal orbit_instance
        nonlocal kwargs
        x_orbit = orbit_instance.from_numpy_array(
            x, *orbit_instance.constants(), **kwargs
        )
        xvec = x_orbit.eqn(**kwargs).state.ravel()
        # Need components for the parameters, but typically they will not have an associated component in the equation;
        # however, I do not think it should be by default.
        if x_orbit.cdof().size - xvec.size > 0:
            return np.pad(xvec, (0, x_orbit.cdof().size - xvec.size))
        else:
            return xvec

    if method in ["hybr", "lm"]:
        if kwargs.get("jac_strategy", "jacobian") == "jacobian":

            def _jac(x):
                """
                The jacobian of the root function, matrix.

                Parameters
                ----------
                x : ndarray
                    Same dimensions as orbit_vector

                Returns
                -------
                ndarray :
                    Jacobian, of F (dF/dX)

                """
                nonlocal orbit_instance
                nonlocal kwargs
                x_orbit = orbit_instance.from_numpy_array(
                    x, *orbit_instance.constants()
                )
                # gradient does not have the same problem that the equation
                J = x_orbit.jacobian(**kwargs)
                if J.shape[1] - J.shape[0] > 0:
                    return np.pad(J, ((0, J.shape[1] - J.shape[0]), (0, 0)))
                else:
                    return J

            _rootjac = {"jac": _jac}
        else:
            _rootjac = {"jac": kwargs.get("jac_strategy", "jacobian")}
    else:
        _rootjac = {"jac": None}

    return _rootfunc, _rootjac


def _exit_messages(orbit_instance, status, verbose=False):
    """
    Convert numerical status integer to string message.

    Parameters
    ----------
    orbit_instance : Orbit
        The final Orbit iterate.
    status : int
        The exit status from a numerical algorithms

    Returns
    -------
    msg : str
        A string containing information on termination.

    """

    if status == 0:
        msg = " ".join(
            [
                f"\nstep size did not meet minimum requirements defined by 'min_step',"
                f" terminating with cost {orbit_instance.cost()}."
            ]
        )
    elif status == 1:
        msg = f"\nsolution met cost function tolerance, terminating with cost {orbit_instance.cost()}."
    elif status == 2:
        msg = f"\nmaximum number of iterations reached, terminating with cost {orbit_instance.cost()}."
    elif status == 3:
        msg = " ".join(
            [
                f"\ninsufficient cost decrease, (new_cost - cost) / max([new_cost, cost, 1])<ftol",
                f"decrease ftol to proceed, terminating with cost {orbit_instance.cost()}",
            ]
        )
    else:
        # A general catch all/placeholder for custom/future status flag values.
        msg = f"\nunspecified exit message, terminating with cost {orbit_instance.cost()}."
    if verbose:
        print(msg)
        sys.stdout.flush()
    return msg


def _process_correction(
    orbit_instance,
    next_orbit_instance,
    runtime_statistics,
    tol,
    maxiter,
    ftol,
    step_size,
    min_step,
    cost,
    next_cost,
    method,
    cost_logging=False,
    inner_nit=None,
    verbose=False,
):
    """
    Helper function which parses the output of algorithms.

    Parameters
    ----------
    orbit_instance : Orbit
        Orbit representing the previous iterate in numerical optimization process.
    next_orbit_instance : Orbit
        Orbit representing the current iterate in numerical optimization process.
    runtime_statistics : dict
        Dict containing various statistics from algorithm output
    tol : float
        The numerical tolerance which determines success
    maxiter : int
        The maximum number of "steps" per numerical method
    ftol : float
        Minimal change in cost allowed per step (approximate gradient norm).
    step_size : float
        The current numerical step size for the numerical method. Would always 1 if backtracking not allowed.
    min_step : float
        The smallest step size allowed; lower bound of backtracking.
    cost : float
        The value of the cost functional evaluated at the previous iterate
    next_cost : float
        The value of the cost functional evaluated at the next iterate
    method : str
        The numerical method currently being employed
    cost_logging : bool, default False
        Whether or not to append the cost from every step of the numerical methods. Can cause dramatic slow down
        for numerical methods with many steps.
    inner_nit : int
        Only valid for Newton-descent when approximate inverses are used; essentially line-searching using step of
        length inner_nit * step_size.
    verbose : bool
        Whether to print various update messages or not.

    Returns
    -------
    next_orbit_instance, runtime_statistics : Orbit, dict
        The next Orbit corresponding to numerical stepping, and the updated runtime statistics dict.

    """
    # If the trigger that broke the while loop was step_size then assume next_cost < cost was not met.
    runtime_statistics["nit"] += 1
    if next_cost <= tol:
        runtime_statistics["status"] = -1
        return next_orbit_instance, next_cost, runtime_statistics
    elif step_size <= min_step:
        runtime_statistics["status"] = 0
        return orbit_instance, cost, runtime_statistics
    elif int(runtime_statistics["nit"]) == int(maxiter):
        runtime_statistics["status"] = 2
        if next_cost < cost:
            return next_orbit_instance, next_cost, runtime_statistics
        else:
            return orbit_instance, cost, runtime_statistics
    elif (cost - next_cost) / max([cost, next_cost, 1]) < ftol:
        runtime_statistics["status"] = 3
        return orbit_instance, cost, runtime_statistics
    else:
        if method in ["adj", "gd"]:
            if verbose:
                if np.mod(runtime_statistics["nit"], 5000) == 0:
                    print(
                        "\n cost={:.7f} after {} adjoint descent steps. Parameters={}".format(
                            next_cost,
                            runtime_statistics["nit"],
                            orbit_instance.parameters,
                        )
                    )
                elif np.mod(runtime_statistics["nit"], 100) == 0:
                    print("#", end="")
            # Too many steps elapse for adjoint descent; this would eventually cause dramatic slow down.
            if cost_logging and np.mod(runtime_statistics["nit"], 1000) == 0:
                runtime_statistics["costs"].append(next_cost)

        elif method == "newton_descent":
            if verbose and inner_nit is not None:
                print(
                    "cost={} after {} inner loops, for a total Newton descent step with size {}".format(
                        next_cost, inner_nit, step_size * inner_nit
                    )
                )
            elif verbose and np.mod(runtime_statistics["nit"], 25) == 0:
                print(
                    "cost={} after {} Newton descent steps ".format(
                        next_cost, runtime_statistics["nit"]
                    )
                )

            if cost_logging:
                runtime_statistics["costs"].append(next_cost)
        else:
            if cost_logging:
                runtime_statistics["costs"].append(next_cost)
            if verbose:
                if np.mod(runtime_statistics["nit"], 25) == 0:
                    print(
                        " cost={:.7f} after {} calls to {}. Parameters={}".format(
                            next_cost,
                            runtime_statistics["nit"],
                            method,
                            orbit_instance.parameters,
                        )
                    )
                else:
                    print("#", end="")
            sys.stdout.flush()
        return next_orbit_instance, next_cost, runtime_statistics
