import numpy as np

__all__ = ['correct_aspect_ratios', 'parameter_based_discretization', 'rediscretize']

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
        newN = max([mode_orbit.N, mode_other_orbit.N])

        # The division/multiplication by 2 ensures that new discretization size is an even number.
        orbit_correct_shape = rediscretize(mode_orbit, newN=newN, newM=2*(int(orbit_speriod_fraction*totalM + 1) // 2))
        other_orbit_correct_shape = rediscretize(mode_other_orbit, newN=newN, newM=2*(int(other_orbit_speriod_fraction*totalM + 1) // 2))
    else:
        # Identical explanation as above just different dimension
        orbit_period_fraction = mode_orbit.T / (mode_orbit.T + mode_other_orbit.T)
        other_orbit_period_fraction = mode_other_orbit.T / (mode_orbit.T + mode_other_orbit.T)
        totalN = (mode_orbit.N + mode_other_orbit.N)
        newM = max([mode_orbit.M, mode_other_orbit.M])
        orbit_correct_shape = rediscretize(mode_orbit, newN=2*(int(orbit_period_fraction*totalN + 1) // 2), newM=newM)
        other_orbit_correct_shape = rediscretize(mode_other_orbit, newN=2*(int(other_orbit_period_fraction*totalN + 1) // 2), newM=newM)
    return orbit_correct_shape.convert(to=orbit.state_type), other_orbit_correct_shape.convert(to=other_orbit.state_type)

def parameter_based_discretization(orbit, **kwargs):
    resolution=kwargs.get('resolution', 'normal')
    if resolution == 'coarse':
        return np.max([2**(int(np.log2(orbit.T)-2)), 32]), np.max([2**(int(np.log2(orbit.L)-1)), 32])
    elif resolution == 'normal':
        return np.max([2**(int(np.log2(orbit.T))), 32]), np.max([2**(int(np.log2(orbit.L)) + 1), 32])
    elif resolution == 'fine':
        return np.max([2**(int(np.log2(orbit.T)+4)), 32]), np.max([2**(int(np.log2(orbit.L))+2), 32])
    else:
        return orbit

def rediscretize(x, parameter_based=False, **kwargs):
    # Return class object with new discretization size. Not performed in place
    # because it is used in other functions; don't want user to be caught unawares
    if parameter_based:
        newN, newM = parameter_based_discretization(x)
    else:
        newN, newM = kwargs.get('newN',x.N), kwargs.get('newM',x.M)
    # Copy state information to new orbit; don't perform operations inplace, only create new orbit
    placeholder_orbit = x.__class__(state=x.state, state_type=x.state_type, T=x.T, L=x.L, S=x.S)
    placeholder_orbit = placeholder_orbit.convert(to='modes')
    if newN == x.N and newM == x.M:
        return x
    else:
        #placeholders for workaround of rfft normalization change
        oldN, oldM = x.N, x.M
        if np.mod(newN, 2) or np.mod(newM, 2):
            raise ValueError('New discretization size must be an even number, preferably a power of 2')
        else:
            if newM == x.M:
                pass
            elif newM > x.M:
                placeholder_orbit = placeholder_orbit.mode_padding(newM, dimension='space')
            elif newM < x.M:
                placeholder_orbit = placeholder_orbit.mode_truncation(newM, dimension='space')

            if newN == x.N:
                pass
            elif newN > x.N:
                placeholder_orbit = placeholder_orbit.mode_padding(newN, dimension='time')
            elif newN < x.N:
                placeholder_orbit = placeholder_orbit.mode_truncation(newN, dimension='time')


            # Keep the norm of the physical field the same, need to
            # account for discrepancy between new and old Fourier transform
            # normalization factors. This is equivalent to normalizing
            # by 1/N and 1/M on forward transforms; the reason why
            # it was not implemented this way is to not have to keep
            # track of normalization factors anywhere but here.
            placeholder_orbit.state = (np.sqrt(newN*newM)/np.sqrt(oldN*oldM))*placeholder_orbit.state
            placeholder_orbit.convert(inplace=True, to=x.state_type)

            return placeholder_orbit
