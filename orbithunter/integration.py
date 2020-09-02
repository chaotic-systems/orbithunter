from math import pi
from orbithunter import *
import numpy as np


def dx_spatial_modes(orbit_instance, power=1):
    modes = orbit_instance.convert(to='s_modes').state
    # Elementwise multiplication of modes with frequencies, this is the derivative.
    dxn_modes = np.multiply(orbit_instance.elementwise_dxn(orbit_instance.parameters, power=power), modes)

    # If the order of the differentiation is odd, need to swap imaginary and real components.
    if np.mod(power, 2):
        dxn_modes = swap_modes(dxn_modes, dimension='space')

    return orbit_instance.__class__(state=dxn_modes, state_type='s_modes',
                                    T=orbit_instance.T, L=orbit_instance.L, S=orbit_instance.S)


def integrate(orbit_instance, **kwargs):
    """ Exponential time-differencing Runge-Kutta 4th order integration scheme.


    Parameters
    ----------
    orbit_instance
    kwargs

    Returns
    -------

    Notes
    -----
    Adapter
    https://epubs.siam.org/doi/abs/10.1137/S1064827502410633?journalCode=sjoce3
    """
    # Take the last row (T=0) so this works for relative periodic solutions as well. i
    orbit_instance_t_equals_0 = orbit_instance.__class__(state=orbit_instance.state[-1, :].reshape(1,-1),
                                                         state_type=orbit_instance.state_type,
                                                         T=orbit_instance.T, L=orbit_instance.L,
                                                         S=orbit_instance.S).convert(to='s_modes')
    # stepsize
    h = kwargs.get('step_size', 0.01)

    # Because N = 1, this is just the spatial matrices, negative sign b.c. other side of equation.
    lin_diag = -1.0*(orbit_instance_t_equals_0.elementwise_dxn(orbit_instance_t_equals_0.parameters, power=2)
                     + orbit_instance_t_equals_0.elementwise_dxn(orbit_instance_t_equals_0.parameters, power=4)).reshape(-1, 1)

    E = np.exp(h*lin_diag)
    E2 = np.exp(h*lin_diag/2.0)

    n_roots = 16
    roots_of_unity = np.exp(1.0j*pi*(np.arange(1, n_roots+1, 1)-0.5)/n_roots).reshape(1, n_roots)

    # Matrix quantities for exponential time differencing.
    LR = h*np.tile(lin_diag, (1, n_roots)) + np.tile(roots_of_unity, (orbit_instance_t_equals_0.M-2, 1))
    Q = h*np.real(np.mean((np.exp(LR/2.)-1.0)/LR, axis=1))
    f1 = h*np.real(np.mean((-4.0-LR+np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2.0+LR+np.exp(LR)*(-2.0+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4.0-3.0*LR-LR**2+np.exp(LR)*(4.0-LR))/LR**3, axis=1))

    Q = Q.reshape(1, -1)
    f1 = f1.reshape(1, -1)
    f2 = f2.reshape(1, -1)
    f3 = f3.reshape(1, -1)
    E = E.reshape(1, -1)
    E2 = E2.reshape(1, -1)

    u = orbit_instance_t_equals_0.convert(to='field').state
    v = orbit_instance_t_equals_0.convert(to='s_modes')
    tmax = orbit_instance_t_equals_0.T
    nmax = int(tmax/h)

    for step in range(0, nmax):
        Nv = -0.5*dx_spatial_modes(v.convert(to='field')**2, power=1)
        a = v.statemul(E2) + Nv.statemul(Q)
        Na = -0.5*dx_spatial_modes(a.convert(to='field')**2, power=1)
        b = v.statemul(E2) + Na.statemul(Q)
        Nb = -0.5*dx_spatial_modes(b.convert(to='field')**2, power=1)
        c = a.statemul(E2) + (2.0 * Nb - Nv).statemul(Q)
        Nc = -0.5*dx_spatial_modes(c.convert(to='field')**2, power=1)
        v = (v.statemul(E) + Nv.statemul(f1)
             + (2.0 * (Na + Nb)).statemul(f2) + Nc.statemul(f3))
        u = np.append(v.convert(to='field').state, u)

    return orbit_instance.__class__(state=u.reshape(nmax+1, -1), state_type='field', T=orbit_instance.T, L=orbit_instance.L, S=orbit_instance.S)




