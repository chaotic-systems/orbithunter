import numpy as np

__all__ = ['correct_aspect_ratios', 'rediscretize', '_parameter_based_discretization']


def correct_aspect_ratios(orbit_, other_orbit_, direction='space', **kwargs):
    mode_orbit = orbit_.convert(inplace=False)
    mode_other_orbit = other_orbit_.convert(inplace=False)
    if direction == 'space':
        # Find what fraction of the total domain each constituent comprises
        orbit_speriod_fraction = mode_orbit_.L / (mode_orbit_.L + mode_other_orbit_.L)
        other_orbit_speriod_fraction = mode_other_orbit_.L / (mode_orbit_.L + mode_other_orbit_.L)
        totalM = (mode_orbit_.M + mode_other_orbit_.M)

        # In order to concatenate the arrays need to have the same size in the transverse direction
        # i.e. in order to concatenate horizontally need to have same height
        new_N = max([mode_orbit_.N, mode_other_orbit_.N])

        # The division/multiplication by 2 ensures that new discretization size is an even number.
        orbit_correct_shape = rediscretize(mode_orbit, new_N=new_N,
                                           new_M=2*(int(orbit_speriod_fraction*totalM + 1) // 2))
        other_orbit_correct_shape = rediscretize(mode_other_orbit, new_N=new_N,
                                                 new_M=2*(int(other_orbit_speriod_fraction*totalM + 1) // 2))
    else:
        # Identical explanation as above just different dimension
        orbit_period_fraction = mode_orbit_.T / (mode_orbit_.T + mode_other_orbit_.T)
        other_orbit_period_fraction = mode_other_orbit_.T / (mode_orbit_.T + mode_other_orbit_.T)
        totalN = (mode_orbit_.N + mode_other_orbit_.N)
        new_M = max([mode_orbit_.M, mode_other_orbit_.M])
        orbit_correct_shape = rediscretize(mode_orbit,
                                           new_N=2*(int(orbit_period_fraction*totalN + 1) // 2), new_M=new_M)
        other_orbit_correct_shape = rediscretize(mode_other_orbit,
                                                 new_N=2*(int(other_orbit_period_fraction*totalN + 1) // 2), new_M=new_M)
    return (orbit_correct_shape.convert(to=orbit_.state_type),
            other_orbit_correct_shape.convert(to=other_orbit_.state_type))


def _parameter_based_discretization(parameters, **kwargs):
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
    the appropriate attributes of the rediscretized orbit_.
    """
    resolution = kwargs.get('resolution', 'normal')
    T, L = parameters['T'], parameters['L']
    if kwargs.get('N', None) is None:
        if resolution == 'coarse':
            N = np.max([2**(int(np.log2(T+1)-2)), 32])
        elif resolution == 'fine':
            N = np.max([2**(int(np.log2(T+1)+4)), 32])
        else:
            N = np.max([2**(int(np.log2(T+1))), 32])
    else:
        N = kwargs.get('M', None)

    if kwargs.get('M', None) is None:
        if resolution == 'coarse':
            M = np.max([2**(int(np.log2(L+1)-1)), 32])
        elif resolution == 'fine':
            M = np.max([2**(int(np.log2(L+1))+2), 32])
        else:
            M = np.max([2**(int(np.log2(L+1)) + 1), 32])
    else:
        M = kwargs.get('M', None)

    return N, M


def rediscretize(orbit_, parameter_based=False, **kwargs):
    # Return class object with new discretization size. Not performed in place
    # because it is used in other functions; don't want user to be caught unawares
    # Copy state information to new orbit; don't perform operations inplace, only create new orbit
    placeholder_orbit = orbit_.__class__(state=orbit_.state, state_type=orbit_.state_type, parameters=orbit_.parameters
                                        ).convert(to='modes')
    if parameter_based:
        new_N, new_M = _parameter_based_discretization(orbit_.parameters, **kwargs)
    else:
        new_N, new_M = kwargs.get('new_N', orbit_.N), kwargs.get('new_M', orbit_.M)

    if new_N == orbit_.N and new_M == orbit_.M:
        return orbit_
    else:
        if new_M == orbit_.M:
            pass
        elif new_M > orbit_.M:
            placeholder_orbit = placeholder_orbit.mode_padding(new_M, dimension='space')
        elif new_M < orbit_.M:
            placeholder_orbit = placeholder_orbit.mode_truncation(new_M, dimension='space')

        if new_N == orbit_.N:
            pass
        elif new_N > orbit_.N:
            placeholder_orbit = placeholder_orbit.mode_padding(new_N, dimension='time')
        elif new_N < orbit_.N:
            placeholder_orbit = placeholder_orbit.mode_truncation(new_N, dimension='time')

        return placeholder_orbit.convert(to=orbit_.state_type, inplace=True)

