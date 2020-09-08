import numpy as np

__all__ = ['correct_aspect_ratios', 'rediscretize']


def correct_aspect_ratios(orbit, other_orbit, direction='space', **kwargs):
    mode_orbit = orbit.convert(inplace=False)
    mode_other_orbit = other_orbit.convert(inplace=False)
    if direction == 'space':
        # Find what fraction of the total domain each constituent comprises
        orbit_speriod_fraction = mode_orbit.L / (mode_orbit.L + mode_other_orbit.L)
        other_orbit_speriod_fraction = mode_other_orbit.L / (mode_orbit.L + mode_other_orbit.L)
        totalM = (mode_orbit.M + mode_other_orbit.M)

        # In order to concatenate the arrays need to have the same size in the transverse direction
        # i.e. in order to concatenate horizontally need to have same height
        new_N = max([mode_orbit.N, mode_other_orbit.N])

        # The division/multiplication by 2 ensures that new discretization size is an even number.
        orbit_correct_shape = rediscretize(mode_orbit, new_N=new_N,
                                           new_M=2*(int(orbit_speriod_fraction*totalM + 1) // 2))
        other_orbit_correct_shape = rediscretize(mode_other_orbit, new_N=new_N,
                                                 new_M=2*(int(other_orbit_speriod_fraction*totalM + 1) // 2))
    else:
        # Identical explanation as above just different dimension
        orbit_period_fraction = mode_orbit.T / (mode_orbit.T + mode_other_orbit.T)
        other_orbit_period_fraction = mode_other_orbit.T / (mode_orbit.T + mode_other_orbit.T)
        totalN = (mode_orbit.N + mode_other_orbit.N)
        new_M = max([mode_orbit.M, mode_other_orbit.M])
        orbit_correct_shape = rediscretize(mode_orbit,
                                           new_N=2*(int(orbit_period_fraction*totalN + 1) // 2), new_M=new_M)
        other_orbit_correct_shape = rediscretize(mode_other_orbit,
                                                 new_N=2*(int(other_orbit_period_fraction*totalN + 1) // 2), new_M=new_M)
    return (orbit_correct_shape.convert(to=orbit.state_type),
            other_orbit_correct_shape.convert(to=other_orbit.state_type))


def _parameter_based_discretization(orbit, **kwargs):
    """ Follow orbithunter conventions for discretization size.


    Parameters
    ----------
    orbit : Orbit or Orbit subclass
    orbithunter class instance whose time, space periods will be used to determine the new discretization values.
    kwargs :
    resolution : str
    Takes values 'coarse', 'normal', 'fine'. These options return one of three orbithunter conventions for the
    discretization size.

    Returns
    -------
    int, int
    The new spatiotemporal discretization given as the number of time points (rows) and number of space points (columns)

    Notes
    -----
    This function should only ever be called by rediscretize, the returned values can always be accessed by
    the appropriate attributes of the rediscretized orbit.
    """
    resolution=kwargs.get('resolution', 'normal')
    if resolution == 'coarse':
        return np.max([2**(int(np.log2(orbit.T)-2)), 32]), np.max([2**(int(np.log2(orbit.L)-1)), 32])
    elif resolution == 'normal':
        return np.max([2**(int(np.log2(orbit.T))), 32]), np.max([2**(int(np.log2(orbit.L)) + 1), 32])
    elif resolution == 'fine':
        return np.max([2**(int(np.log2(orbit.T)+4)), 32]), np.max([2**(int(np.log2(orbit.L))+2), 32])
    else:
        return orbit.N, orbit.M


def rediscretize(orbit, parameter_based=False, **kwargs):
    # Return class object with new discretization size. Not performed in place
    # because it is used in other functions; don't want user to be caught unawares
    # Copy state information to new orbit; don't perform operations inplace, only create new orbit
    placeholder_orbit = orbit.__class__(state=orbit.state, state_type=orbit.state_type,
                                        T=orbit.T, L=orbit.L, S=orbit.S).convert(to='modes')
    if parameter_based:
        new_N, new_M = _parameter_based_discretization(orbit, **kwargs)
    else:
        new_N, new_M = kwargs.get('new_N', orbit.N), kwargs.get('new_M', orbit.M)

    if new_N == orbit.N and new_M == orbit.M:
        return orbit
    else:
        if np.mod(new_N, 2) or np.mod(new_M, 2):
            raise ValueError('New discretization size must be an even number, preferably a power of 2')
        else:
            if new_M == orbit.M:
                pass
            elif new_M > orbit.M:
                placeholder_orbit = placeholder_orbit.mode_padding(new_M, dimension='space')
            elif new_M < orbit.M:
                placeholder_orbit = placeholder_orbit.mode_truncation(new_M, dimension='space')

            if new_N == orbit.N:
                pass
            elif new_N > orbit.N:
                placeholder_orbit = placeholder_orbit.mode_padding(new_N, dimension='time')
            elif new_N < orbit.N:
                placeholder_orbit = placeholder_orbit.mode_truncation(new_N, dimension='time')

            return placeholder_orbit.convert(to=orbit.state_type)

