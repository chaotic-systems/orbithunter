from math import pi
from .arrayops import swap_modes
import numpy as np

__all__ = ['kse_integrate']


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

    modes = orbit_.convert(to='s_modes').state
    # Elementwise multiplication of modes with frequencies, this is the derivative.
    dxn_modes = np.multiply(orbit_.elementwise_dxn(orbit_.dx_parameters, power=power), modes)

    # If the order of the differentiation is odd, need to swap imaginary and real components.
    if np.mod(power, 2):
        dxn_modes = swap_modes(dxn_modes, axis=1)

    return orbit_.__class__(state=dxn_modes, basis='s_modes', parameters=orbit_.parameters)


def kse_integrate(orbit_, **kwargs):
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
    orbit_ = orbit_.convert(to='s_modes')
    integration_time = kwargs.get('integration_time', orbit_.T)
    start_point = kwargs.get('starting_point', -1)
    # Take the last row (t=0) or first row (t=T) so this works for relative periodic solutions as well.
    orbit_t_equals_0 = orbit_.__class__(state=orbit_.state[start_point, :].reshape(1, -1), basis=orbit_.basis,
                                        parameters=orbit_.parameters).convert(to='s_modes')
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
    LR = step_size*np.tile(lin_diag, (1, n_roots)) + np.tile(roots_of_unity, (orbit_t_equals_0.M-2, 1))
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

    u = orbit_t_equals_0.convert(to='field').state
    v = orbit_t_equals_0.convert(to='s_modes')
    nmax = int(integration_time / step_size)
    if verbose:
        print('Integration progress [', end='')
    for step in range(0, nmax):
        Nv = -0.5*_dx_spatial_modes(v.convert(to='field')**2, power=1)
        a = v.statemul(E2) + Nv.statemul(Q)
        Na = -0.5*_dx_spatial_modes(a.convert(to='field')**2, power=1)
        b = v.statemul(E2) + Na.statemul(Q)
        Nb = -0.5*_dx_spatial_modes(b.convert(to='field')**2, power=1)
        c = a.statemul(E2) + (2.0 * Nb - Nv).statemul(Q)
        Nc = -0.5*_dx_spatial_modes(c.convert(to='field')**2, power=1)
        v = (v.statemul(E) + Nv.statemul(f1)
             + (2.0 * (Na + Nb)).statemul(f2) + Nc.statemul(f3))
        if kwargs.get('return_trajectory', True):
            u = np.append(v.convert(to='field').state, u)
        else:
            u = v.convert(to='field').state
        if not np.mod(step, nmax // 25) and verbose:
            print('#', end='')
    if verbose:
        print(']', end='')
    # By default do not assign spatial shift S.
    if kwargs.get('return_trajectory', True):
        return orbit_.__class__(state=u.reshape(nmax+1, -1), basis='field', parameters=orbit_.parameters)
    else:
        return orbit_.__class__(state=u.reshape(1, -1), basis='field', parameters=orbit_.parameters)
