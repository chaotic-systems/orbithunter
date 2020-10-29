import numpy as np
import itertools

__all__ = ['dissipation', 'energy', 'energy_variation', 'power', 'shadowing']


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


def dissipation(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.

    Field computed is u_xx**2.
    """

    return _averaging_wrapper(orbit_instance.dx(power=2).convert(to='field')**2,
                              average=average)


def energy(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(0.5 * orbit_instance.convert(to='field')**2,
                              average=average)


def energy_variation(orbit_instance, average=None):
    """ The field u_t * u whose spatial average should equal power - dissipation.

    Returns
    -------
    Field equivalent to u_t * u. Spatial average, <u_t * u> should equal <power> - <dissipation> = <u_x**2> - <u_xx**2>
    """
    return _averaging_wrapper(orbit_instance.convert(to='field').statemul(orbit_instance.dt().convert(to='field')),
                              average=average)


def power(orbit_instance, average=None):
    """ Amount of energy production
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(orbit_instance.dx().convert(to='field')**2,
                              average=average)


def shadowing(window_orbit, base_orbit, **kwargs):

    window_orbit.convert(to='field', inplace=True)
    base_orbit.convert(to='field', inplace=True)
    twindow, xwindow = window_orbit.field_shape
    tbase, xbase = base_orbit.field_shape
    assert twindow < tbase and xwindow < xbase, 'Shadowing window is larger than the orbit being searched. Reshape. '

    # First, need to calculate the norm of the window squared and base squared (squared because then it detects
    # the shape rather than the color coded field), for all translations of the window.
    norm_matrix = np.zeros([tbase-twindow, xbase-xwindow])
    for i, tcorner in enumerate(range(0, base_orbit.field_shape[1]-window_orbit.field_shape[1])):
        if kwargs.get('verbose', False):
            if np.mod(i, 1984//10) == 0:
                print('#', end='')
        for j, xcorner in enumerate(range(0,  base_orbit.field_shape[0]-window_orbit.field_shape[1])):
            norm_matrix[i,j] = np.linalg.norm(base_orbit.state[tcorner:tcorner+twindow, xcorner:xcorner+xwindow]**2
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
