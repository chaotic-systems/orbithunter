from math import pi
from .orbits import spatial_frequencies
import numpy as np

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
        return (1.0 / instance_with_state_to_average.x) * instance_with_state_to_average.state.mean(axis=1)
    elif average == 'time':
        return (1.0 / instance_with_state_to_average.t) * instance_with_state_to_average.state.mean(axis=0)
    elif average == 'spacetime':
        # numpy average is over flattened array by default
        return ((1.0 / instance_with_state_to_average.t) * (1.0 / instance_with_state_to_average.x)
                * instance_with_state_to_average.state.mean())
    else:
        return instance_with_state_to_average.state


def dissipation(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.

    Field computed is u_xx**2.
    """

    return _averaging_wrapper(orbit_instance.dx(order=2).transform(to='field')**2,
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
    return _averaging_wrapper(orbit_instance.transform(to='field') * orbit_instance.dt().transform(to='field'),
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


def shadowing(base_orbit, window_orbit, verbose=False, threshold=0.02, threshold_type='percentile', stride=1,
              mask_region='exterior'):

    window_orbit = window_orbit.transform(to='field')
    base_orbit = base_orbit.transform(to='field')
    twindow, xwindow = window_orbit.shapes()[0]
    tbase, xbase = base_orbit.shapes()[0]

    assert twindow < tbase and xwindow < xbase, 'Shadowing window is larger than the base orbit. resize first. '

    # First, need to calculate the norm of the window squared and base squared (squared because then it detects
    # the shape rather than the color coded field), for all translations of the window.
    mask = np.zeros([tbase, xbase], dtype=bool)
    norm_matrix = np.zeros([(tbase-twindow)//stride, (xbase-xwindow)//stride])
    for i, t in enumerate(range(0, tbase-twindow, stride)):
        if verbose and np.mod(i, ((tbase-twindow)//stride)//10) == 0:
            print('#', end='')
        for j, x in enumerate(range(0,  xbase-xwindow, stride)):
            trange = np.arange(t, t+twindow)
            xrange = np.arange(x, x+xwindow)
            # numpy meshgrid returns list but when used slicing needs a tuple.
            window_slice_indices = tuple(np.meshgrid(trange, xrange, indexing='ij'))
            norm_matrix[i, j] = np.linalg.norm(base_orbit.state[window_slice_indices]**2 - window_orbit.state**2)
    if threshold_type == 'percentile':
        threshold = np.percentile(norm_matrix[norm_matrix > 0.],  threshold)
        indices = np.where((norm_matrix > 0.) & (norm_matrix < threshold))
    else:
        indices = np.where((norm_matrix > 0.) & (norm_matrix < threshold))
    for i, j in zip(*indices):
        t, x = stride*i,  stride*j
        trange = np.arange(t, t+twindow)
        xrange = np.arange(x, x+xwindow)
        # numpy meshgrid returns list but slicing needs a tuple.
        mask[tuple(np.meshgrid(trange, xrange, indexing='ij'))] = True

    # TODO : Look at the mask returned and ensure it is the correct region
    masked_orbit = base_orbit.copy()
    if mask_region == 'exterior':
        mask = np.invert(mask)
    masked_orbit.state = np.ma.masked_array(masked_orbit.transform(to='field').state, mask=mask)
    return masked_orbit, mask, threshold

def cover(base_orbit, *window_orbits, verbose=False, threshold=0.02, threshold_type='percentile', stride=1,
              mask_region='exterior'):
    for window_orbit in window_orbits:
        window_orbit = window_orbit.transform(to='field')
        base_orbit = base_orbit.transform(to='field')
        twindow, xwindow = window_orbit.shapes()[0]
        tbase, xbase = base_orbit.shapes()[0]

        assert twindow < tbase and xwindow < xbase, 'Shadowing window is larger than the base orbit. resize first. '

        # First, need to calculate the norm of the window squared and base squared (squared because then it detects
        # the shape rather than the color coded field), for all translations of the window.
        mask = np.zeros([tbase, xbase], dtype=bool)
        norm_matrix = np.zeros([(tbase-twindow)//stride, (xbase-xwindow)//stride])
        for i, t in enumerate(range(0, tbase-twindow, stride)):
            if verbose and np.mod(i, ((tbase-twindow)//stride)//10) == 0:
                print('#', end='')
            for j, x in enumerate(range(0,  xbase-xwindow, stride)):
                trange = np.arange(t, t+twindow)
                xrange = np.arange(x, x+xwindow)
                # numpy meshgrid returns list but when used slicing needs a tuple.
                window_slice_indices = tuple(np.meshgrid(trange, xrange, indexing='ij'))
                norm_matrix[i, j] = np.linalg.norm(base_orbit.state[window_slice_indices]**2 - window_orbit.state**2)
        if threshold_type == 'percentile':
            threshold = np.percentile(norm_matrix[norm_matrix > 0.],  threshold)
            indices = np.where((norm_matrix > 0.) & (norm_matrix < threshold))
        else:
            indices = np.where((norm_matrix > 0.) & (norm_matrix < threshold))
        for i, j in zip(*indices):
            t, x = stride*i,  stride*j
            trange = np.arange(t, t+twindow)
            xrange = np.arange(x, x+xwindow)
            # numpy meshgrid returns list but slicing needs a tuple.
            mask[tuple(np.meshgrid(trange, xrange, indexing='ij'))] = True

    # TODO : Look at the mask returned and ensure it is the correct region
    masked_orbit = base_orbit.copy()
    if mask_region == 'exterior':
        mask = np.invert(mask)
    masked_orbit.state = np.ma.masked_array(masked_orbit.transform(to='field').state, mask=mask)
    return masked_orbit, mask, threshold



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
    orbit_ = orbit_.transform(to='spatial_modes')
    integration_time = kwargs.get('integration_time', orbit_.t)
    start_point = kwargs.get('starting_point', -1)
    # Take the last row (t=0) or first row (t=T) so this works for relative periodic solutions as well.
    orbit_t = orbit_.__class__(state=orbit_.state[start_point, :].reshape(1, -1), basis=orbit_.basis,
                                        parameters=orbit_.parameters).transform(to='spatial_modes')
    # stepsize
    step_size = kwargs.get('step_size', 0.01)

    # Because N = 1, this is just the spatial matrices, negative sign b.c. other side of equation.
    lin_diag = -1.0*(spatial_frequencies(orbit_t.x, orbit_t.M, orbit_t.shapes()[1][0], order=2)
                     + spatial_frequencies(orbit_t.x, orbit_t.M, orbit_t.shapes()[1][0], order=4)).reshape(-1, 1)

    E = np.exp(step_size*lin_diag)
    E2 = np.exp(step_size*lin_diag/2.0)

    n_roots = 16
    roots_of_unity = np.exp(1.0j*pi*(np.arange(1, n_roots+1, 1)-0.5)/n_roots).reshape(1, n_roots)

    # Matrix quantities for exponential time differencing.
    LR = step_size*np.tile(lin_diag, (1, n_roots)) + np.tile(roots_of_unity, (orbit_t.shapes()[2][1], 1))
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
    if orbit_t.__class__.__name__ == 'AntisymmetricOrbitKS':
            Q[:-orbit_t.m, :] = 0
            f1[:-orbit_t.m, :] = 0
            f2[:-orbit_t.m, :] = 0
            f3[:-orbit_t.m, :] = 0
            E[:-orbit_t.m, :] = 0
            E2[:-orbit_t.m, :] = 0

    u = orbit_t.transform(to='field').state
    v = orbit_t.transform(to='spatial_modes')
    nmax = int(integration_time / step_size)
    if verbose:
        print('Integration progress [', end='')
    if kwargs.get('return_trajectory', True):
        u = np.zeros([nmax, orbit_t.shapes()[0][1]])
    for step in range(1, nmax+1):
        Nv = -0.5*(v.transform(to='field')**2).dx(computation_basis='spatial_modes', return_basis='spatial_modes')
        a = v * E2 + Nv * Q
        Na = -0.5*(a.transform(to='field')**2).dx(computation_basis='spatial_modes', return_basis='spatial_modes')
        b = v * E2 + Na * Q
        Nb = -0.5*(b.transform(to='field')**2).dx(computation_basis='spatial_modes', return_basis='spatial_modes')
        c = a * E2 + (2 * Nb - Nv) * Q
        Nc = -0.5*(c.transform(to='field')**2).dx(computation_basis='spatial_modes', return_basis='spatial_modes')
        v = v * E + Nv * f1 + (2 * (Na + Nb)) * f2 + Nc * f3
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
        return orbit_.__class__(state=u.reshape(nmax, -1), basis='field', parameters=(integration_time, orbit_.x, 0))
    else:
        return orbit_.__class__(state=u.reshape(1, -1), basis='field', parameters=(integration_time, orbit_.x, 0))


