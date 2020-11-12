from .arrayops import swap_modes
from math import pi
import numpy as np
import itertools

__all__ = ['integrate', 'dissipation', 'energy', 'energy_variation', 'power', 'shadowing']


def _averaging_wrapper(instance_with_state_to_average, average=None):
    """ Apply time, space, or spacetime average to field of instance.


    Parameters
    ----------
    instance_with_state_to_average
    average

    Returns
    -------

    """

    if average == 'space':
        return (1.0 / instance_with_state_to_average.L) * instance_with_state_to_average.state.mean(axis=1)
    elif average == 'time':
        return (1.0 / instance_with_state_to_average.T) * instance_with_state_to_average.state.mean(axis=0)
    elif average == 'spacetime':
        # numpy average is over flattened array by default
        return ((1.0 / instance_with_state_to_average.T) * (1.0 / instance_with_state_to_average.L)
                * instance_with_state_to_average.state.mean())
    else:
        return instance_with_state_to_average.state


def _dx_spatial_modes(orbit_, power=1):
    """ Modification of spatial derivative method

    Parameters
    ----------
    orbit_
    power

    Returns
    -------

    Notes
    -----
    Rewritten to accomodate returning spatial modes.

    """

    modes = orbit_.transform(to='s_modes').state
    # Elementwise multiplication of modes with frequencies, this is the derivative.
    dxn_modes = np.multiply(orbit_.elementwise_dxn(orbit_.dx_parameters, power=power), modes)

    # If the order of the differentiation is odd, need to swap imaginary and real components.
    if np.mod(power, 2):
        dxn_modes = swap_modes(dxn_modes, axis=1)

    return orbit_.__class__(state=dxn_modes, basis='s_modes', parameters=orbit_.parameters)


def dissipation(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.

    Field computed is u_xx**2.
    """

    return _averaging_wrapper(orbit_instance.dx(power=2).transform(to='field')**2,
                              average=average)


def energy(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(0.5 * orbit_instance.transform(to='field')**2,
                              average=average)


def energy_variation(orbit_instance, average=None):
    """ The field u_t * u whose spatial average should equal power - dissipation.

    Returns
    -------
    Field equivalent to u_t * u. Spatial average, <u_t * u> should equal <power> - <dissipation> = <u_x**2> - <u_xx**2>
    """
    return _averaging_wrapper(orbit_instance.transform(to='field').statemul(orbit_instance.dt().transform(to='field')),
                              average=average)


def power(orbit_instance, average=None):
    """ Amount of energy production
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(orbit_instance.dx().transform(to='field')**2,
                              average=average)


def shadowing(window_orbit, base_orbit, **kwargs):

    window_orbit.transform(to='field', inplace=True)
    base_orbit.transform(to='field', inplace=True)
    twindow, xwindow = window_orbit.field_shape
    tbase, xbase = base_orbit.field_shape
    assert twindow < tbase and xwindow < xbase, 'Shadowing window is larger than the orbit being searched. Reshape. '

    # First, need to calculate the norm of the window squared and base squared (squared because then it detects
    # the shape rather than the color coded field), for all translations of the window.
    norm_matrix = np.zeros([tbase-twindow, xbase-xwindow])
    for i, tcorner in enumerate(range(0, base_orbit.field_shape[1]-window_orbit.field_shape[1])):
        if kwargs.get('verbose', False):
            if np.mod(i, (base_orbit.field_shape[1]-window_orbit.field_shape[1])//10) == 0:
                print('#', end='')
        for j, xcorner in enumerate(range(0,  base_orbit.field_shape[0]-window_orbit.field_shape[1])):
            norm_matrix[i, j] = np.linalg.norm(base_orbit.state[tcorner:tcorner+twindow, xcorner:xcorner+xwindow]**2
                                               -window_orbit.state**2)
    indices = np.where(norm_matrix < np.percentile(norm_matrix.ravel(), kwargs.get('norm_percentile', 0.025)))
    non_nan_points = []

    # Once the norms are calculated, the positions with minimal norm represent the correct positions of the
    # window corners, however we need the portions of the field corresponding to cutouts.

    # By iterating over all corners, generating the corresponding windows, and then taking their intersection,
    # we are left with the correct shadowing regions for the tolerance.
    for t, x in zip(*indices):
        if t < tbase-twindow and x < xbase-xwindow:
            tslice = range(t, t+twindow)
            xslice = range(x, x+xwindow)
            testslice = itertools.product(tslice, xslice)
            non_nan_points.extend(list(testslice))
            non_nan_points = list(set(non_nan_points))

    tindex = np.array(tuple(pairs[0] for pairs in non_nan_points))
    xindex = np.array(tuple(pairs[1] for pairs in non_nan_points))
    final_indices = (tindex, xindex)

    masked_orbit = base_orbit.copy()
    mask = np.zeros([tbase, xbase])
    mask[final_indices] = 1
    masked_orbit.state[mask != 1] = np.nan
    return masked_orbit, mask


def integrate(orbit_, **kwargs):
    """ Exponential time-differencing Runge-Kutta 4th order integration scheme.


    Parameters
    ----------
    orbit_
    kwargs

    Returns
    -------

    Notes
    -----
    Adapter
    https://epubs.siam.org/doi/abs/10.1137/S1064827502410633?journalCode=sjoce3

    By default, when input is an instance of relative periodic orbit then shift is calculated off of the
    integrated trajectory. This will lead to plotting issues so unless desired, you should convert to the
    base orbit type first.
    """
    verbose = kwargs.get('verbose', False)
    orbit_ = orbit_.transform(to='s_modes')
    integration_time = kwargs.get('integration_time', orbit_.T)
    start_point = kwargs.get('starting_point', -1)
    # Take the last row (t=0) or first row (t=T) so this works for relative periodic solutions as well.
    orbit_t_equals_0 = orbit_.__class__(state=orbit_.state[start_point, :].reshape(1, -1), basis=orbit_.basis,
                                        parameters=orbit_.parameters).transform(to='s_modes')
    # stepsize
    step_size = kwargs.get('step_size', 0.01)

    # Because N = 1, this is just the spatial matrices, negative sign b.c. other side of equation.
    lin_diag = -1.0*(orbit_t_equals_0.elementwise_dxn(orbit_t_equals_0.dx_parameters, power=2)
                     + orbit_t_equals_0.elementwise_dxn(orbit_t_equals_0.dx_parameters, power=4)).reshape(-1, 1)

    E = np.exp(step_size*lin_diag)
    E2 = np.exp(step_size*lin_diag/2.0)

    n_roots = 16
    roots_of_unity = np.exp(1.0j*pi*(np.arange(1, n_roots+1, 1)-0.5)/n_roots).reshape(1, n_roots)

    # Matrix quantities for exponential time differencing.
    LR = step_size*np.tile(lin_diag, (1, n_roots)) + np.tile(roots_of_unity, (orbit_t_equals_0.mode_shape[1], 1))
    Q = step_size*np.real(np.mean((np.exp(LR/2.)-1.0)/LR, axis=1))
    f1 = step_size*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, axis=1))
    f2 = step_size*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, axis=1))
    f3 = step_size*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, axis=1))

    Q = Q.reshape(1, -1)
    f1 = f1.reshape(1, -1)
    f2 = f2.reshape(1, -1)
    f3 = f3.reshape(1, -1)
    E = E.reshape(1, -1)
    E2 = E2.reshape(1, -1)
    if orbit_t_equals_0.__class__.__name__ == 'AntisymmetricOrbitKS':
            Q = np.concatenate((0*Q, Q), axis=1)
            f1 = np.concatenate((0*f1, f1), axis=1)
            f2 = np.concatenate((0*f2, f2), axis=1)
            f3 = np.concatenate((0*f3, f3), axis=1)
            E = np.concatenate((0*E, E), axis=1)
            E2 = np.concatenate((0*E2, E2), axis=1)

    u = orbit_t_equals_0.transform(to='field').state
    v = orbit_t_equals_0.transform(to='s_modes')
    nmax = int(integration_time / step_size)
    if verbose:
        print('Integration progress [', end='')
    if kwargs.get('return_trajectory', True):
        u = np.zeros([nmax, orbit_t_equals_0.field_shape[1]])
    for step in range(1, nmax):
        Nv = -0.5*_dx_spatial_modes(v.transform(to='field')**2, power=1)
        a = v.statemul(E2) + Nv.statemul(Q)
        Na = -0.5*_dx_spatial_modes(a.transform(to='field')**2, power=1)
        b = v.statemul(E2) + Na.statemul(Q)
        Nb = -0.5*_dx_spatial_modes(b.transform(to='field')**2, power=1)
        c = a.statemul(E2) + (2.0 * Nb - Nv).statemul(Q)
        Nc = -0.5*_dx_spatial_modes(c.transform(to='field')**2, power=1)
        v = (v.statemul(E) + Nv.statemul(f1)
             + (2.0 * (Na + Nb)).statemul(f2) + Nc.statemul(f3))
        if kwargs.get('return_trajectory', True):
            u[-step, :] = v.transform(to='field').state.ravel()
        else:
            u = v.transform(to='field').state
        if not np.mod(step, nmax // 25) and verbose:
            print('#', end='')
    if verbose:
        print(']', end='')
    # By default do not assign spatial shift S.
    if kwargs.get('return_trajectory', True):
        return orbit_.__class__(state=u.reshape(nmax, -1), basis='field', parameters=(integration_time, orbit_.L, 0))
    else:
        return orbit_.__class__(state=u.reshape(1, -1), basis='field', parameters=(integration_time, orbit_.L, 0))


