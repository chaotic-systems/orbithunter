
__all__ = ['kse_dissipation', 'kse_energy', 'kse_energy_variation', 'kse_power']


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


def kse_dissipation(orbit_instance, average=None):
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


def kse_energy(orbit_instance, average=None):
    """ Amount of energy dissipation
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(0.5 * orbit_instance.convert(to='field')**2,
                              average=average)


def kse_energy_variation(orbit_instance, average=None):
    """ The field u_t * u whose spatial average should equal power - dissipation.

    Returns
    -------
    Field equivalent to u_t * u. Spatial average, <u_t * u> should equal <power> - <dissipation> = <u_x**2> - <u_xx**2>
    """
    return _averaging_wrapper(orbit_instance.convert(to='field').statemul(orbit_instance.dt().convert(to='field')),
                              average=average)


def kse_power(orbit_instance, average=None):
    """ Amount of energy production
    Notes
    -----
    Returns the field, not a scalar (or array thereof), so space or spacetime averages are an additional
    step. This is to be  as flexible as possible; this is really only a function that exists for readability
    but also allows for visualization.
    """
    return _averaging_wrapper(orbit_instance.dx().convert(to='field')**2,
                              average=average)
