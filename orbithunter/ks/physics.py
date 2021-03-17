from math import pi
import numpy as np
import warnings

__all__ = ["integrate", "dissipation", "energy", "energy_variation", "power"]


def _averaging_wrapper(instance_with_state_to_average, average="spacetime"):
    """ Apply time, space, or spacetime average to field of instance.


    Parameters
    ----------
    instance_with_state_to_average : Orbit
    average : str
        Which dimensions/values to average with respect to.

    Returns
    -------
    ndarray or float :
        The averages either along certain dimensions or all dimensions.

    """
    if average == "space":
        return (
            1.0 / instance_with_state_to_average.x
        ) * instance_with_state_to_average.state.mean(axis=1)
    elif average == "time":
        return (
            1.0 / instance_with_state_to_average.t
        ) * instance_with_state_to_average.state.mean(axis=0)
    elif average == "spacetime":
        # numpy average is over flattened array by default
        return (
            (1.0 / instance_with_state_to_average.t)
            * (1.0 / instance_with_state_to_average.x)
            * instance_with_state_to_average.state.mean()
        )
    elif average == "discrete_spacetime":
        # numpy average is over flattened array by default
        return (
            (1.0 / instance_with_state_to_average.n)
            * (1.0 / instance_with_state_to_average.m)
            * instance_with_state_to_average.state.mean()
        )
    else:
        return instance_with_state_to_average.state


def dissipation(orbit_instance, average="spacetime"):
    """ Average energy dissipation or corresponding field.

    Returns
    -------
    float or ndarray :
        Spatiotemporal dissipation returned as a field, or averaged along an axis (ndarray) or spacetime (float).

    Notes
    -----
    Dissipation = u_xx^2.

    """
    return _averaging_wrapper(
        orbit_instance.dx(order=2).transform(to="field") ** 2, average=average
    )


def energy(orbit_instance, average="spacetime"):
    """ Average energy or corresponding field

    Returns
    -------
    float or ndarray :
        Spatiotemporal power returned as a field, or averaged along an axis (ndarray) or spacetime (float).

    Notes
    -----
    Energy = 1/2 u^2

    """
    return _averaging_wrapper(
        0.5 * orbit_instance.transform(to="field") ** 2, average=average
    )


def energy_variation(orbit_instance, average="spacetime"):
    """ The field u_t * u whose spatial average should equal power - dissipation.

    Returns
    -------
    float or ndarray :
        Spatiotemporal energy variation returned as a field, or averaged along an axis (ndarray) or spacetime (float).
    Returns
    -------
    Field equivalent to u_t * u.

    """
    return _averaging_wrapper(
        orbit_instance.transform(to="field")
        * orbit_instance.dt().transform(to="field"),
        average=average,
    )


def power(orbit_instance, average="spacetime"):
    """ Average power or corresponding field

    Returns
    -------
    float or ndarray :
        Spatiotemporal power returned as a field, or averaged along an axis (ndarray) or spacetime (float).

    Notes
    -----
    power = u_x^2

    """
    return _averaging_wrapper(
        orbit_instance.dx().transform(to="field") ** 2, average=average
    )


def integrate(orbit_, **kwargs):
    """ Exponential time-differencing Runge-Kutta 4th order integration scheme.

    Parameters
    ----------
    orbit_ : Orbit
        The Orbit which contains the initial starting point for the integration
    kwargs : dict
        verbose : bool

        starting_point :
            The row to select as the initial value.
        integration_time : float
            The total amount of time to integrate, if not provided then defaults to orbit's period (for reproduction
            of orbit, typically).
        step_size : float
            The integration step
        return_trajectory : bool
            Whether to store and return all integration steps or only the endpoint.

    Returns
    -------
    Orbit :
        Orbit instance with either an integrated trajectory or its final value as a state.
    Notes
    -----
    Adapter https://epubs.siam.org/doi/abs/10.1137/S1064827502410633?journalCode=sjoce3

    By default, when input is an instance of relative periodic orbit then shift is calculated off of the
    integrated trajectory. This will lead to plotting issues so unless desired, you should convert to the
    base orbit type first.

    """
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    from scipy.fft import rfftfreq

    verbose = kwargs.get("verbose", False)
    orbit_ = orbit_.transform(to="spatial_modes")
    integration_time = kwargs.get("integration_time", orbit_.t)
    start_point = kwargs.get("starting_point", -1)
    # Take the last row (t=0) or first row (t=T) so this works for relative periodic solutions as well.
    orbit_t = orbit_.__class__(
        state=orbit_.state[start_point, :].reshape(1, -1),
        basis=orbit_.basis,
        parameters=orbit_.parameters,
    ).transform(to="spatial_modes")
    # stepsize
    step_size = kwargs.get("step_size", 0.01)
    m, x = orbit_t.m, orbit_t.x
    q = (2 * pi * m / x) * rfftfreq(m)[1:-1].reshape(1, -1)
    q = np.concatenate((q, q))
    # Because N = 1, this is just the spatial matrices, negative sign b.c. other side of equation.
    lin_diag = (q ** 2 - q ** 4).reshape(-1, 1)

    E = np.exp(step_size * lin_diag)
    E2 = np.exp(step_size * lin_diag / 2.0)

    n_roots = 16
    roots_of_unity = np.exp(
        1.0j * pi * (np.arange(1, n_roots + 1, 1) - 0.5) / n_roots
    ).reshape(1, n_roots)

    # Matrix quantities for exponential time differencing.
    LR = step_size * np.tile(lin_diag, (1, n_roots)) + np.tile(
        roots_of_unity, (orbit_t.shapes()[2][1], 1)
    )
    Q = step_size * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = step_size * np.real(
        np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1)
    )
    f2 = step_size * np.real(
        np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1)
    )
    f3 = step_size * np.real(
        np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1)
    )

    Q = Q.reshape(1, -1)
    f1 = f1.reshape(1, -1)
    f2 = f2.reshape(1, -1)
    f3 = f3.reshape(1, -1)
    E = E.reshape(1, -1)
    E2 = E2.reshape(1, -1)
    if orbit_t.__class__.__name__ == "AntisymmetricOrbitKS":
        Q[: -orbit_t.m, :] = 0
        f1[: -orbit_t.m, :] = 0
        f2[: -orbit_t.m, :] = 0
        f3[: -orbit_t.m, :] = 0
        E[: -orbit_t.m, :] = 0
        E2[: -orbit_t.m, :] = 0

    u = orbit_t.transform(to="field").state
    v = orbit_t.transform(to="spatial_modes")
    nmax = int(integration_time / step_size)
    if verbose:
        print("Integration progress [", end="")
    if kwargs.get("return_trajectory", True):
        u = np.zeros([nmax, orbit_t.shapes()[0][1]])
    for step in range(1, nmax + 1):
        Nv = -0.5 * (v.transform(to="field") ** 2).dx(
            computation_basis="spatial_modes", return_basis="spatial_modes"
        )
        a = v * E2 + Nv * Q
        Na = -0.5 * (a.transform(to="field") ** 2).dx(
            computation_basis="spatial_modes", return_basis="spatial_modes"
        )
        b = v * E2 + Na * Q
        Nb = -0.5 * (b.transform(to="field") ** 2).dx(
            computation_basis="spatial_modes", return_basis="spatial_modes"
        )
        c = a * E2 + (2 * Nb - Nv) * Q
        Nc = -0.5 * (c.transform(to="field") ** 2).dx(
            computation_basis="spatial_modes", return_basis="spatial_modes"
        )
        v = v * E + Nv * f1 + (2 * (Na + Nb)) * f2 + Nc * f3
        if orbit_.__class__.__name__ in ["AntisymmetricOrbitKS", "EquilibriumOrbitKS"]:
            v.real = 0

        if kwargs.get("return_trajectory", True):
            u[-step, :] = v.transform(to="field").state.ravel()
        else:
            u = v.transform(to="field").state
        if not np.mod(step, nmax // 25) and verbose:
            print("#", end="")
    if verbose:
        print("]", end="")
    # By default do not assign spatial shift S.
    if kwargs.get("return_trajectory", True):
        int_orbit = orbit_.__class__(
            state=u.reshape(nmax, -1),
            basis="field",
            parameters=(integration_time, orbit_.x, 0),
        )
    else:
        int_orbit = orbit_.__class__(
            state=u.reshape(1, -1),
            basis="field",
            parameters=(integration_time, orbit_.x, 0),
        )
    warnings.resetwarnings()
    return int_orbit
