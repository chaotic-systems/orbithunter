import numpy as np

__all__ = ['correct_aspect_ratios', 'rediscretize', '_parameter_based_discretization']


def correct_aspect_ratios(iterable_of_orbits, axis=0, **kwargs):

    # need to iterate over all orbits, find the total dimension and # of points, then create the new_shape tuples.
    total_dimension_extent = 0
    total_number_of_discretization_points = 0
    for o in iterable_of_orbits:
        total_dimension_extent += tuple(o.parameters.values())[axis]
        total_number_of_discretization_points += o.parameters['field_shape'][axis]

    # Replace the # of points along axis with the rescaled values based upon the total extent of the axis dimension.
    new_shapes = [(2*(int((total_number_of_discretization_points * (tuple(o.parameters.values())[axis]
                  / total_dimension_extent))+1))//2) if i == axis else o.parameters['field_shape'][i]
                  for o in iterable_of_orbits for i in range(len(o.shape))]


    iterable_of_reshaped_orbits = [rediscretize(o, new_shape=shp)
                                   for o, shp in zip(iterable_of_orbits, new_shapes)]
    return iterable_of_reshaped_orbits


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
    equation = kwargs.get('equation', 'KS')
    if equation == 'KS':
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
    else:
        return None


def rediscretize(orbit_, parameter_based=False, **kwargs):
    # Return class object with new discretization size. Not performed in place
    # because it is used in other functions; don't want user to be caught unawares
    # Copy state information to new orbit; don't perform operations inplace, only create new orbit
    placeholder_orbit = orbit_.__class__(state=orbit_.state, state_type=orbit_.state_type, parameters=orbit_.parameters
                                         ).convert(to='modes')

    equation = kwargs.get('equation', 'KS')
    if equation == 'KS':
        if parameter_based:
            new_shape = _parameter_based_discretization(orbit_.parameters, **kwargs)
        else:
            new_shape = kwargs.get('new_shape', orbit_.parameters['field_shape'])

        if orbit_.shape == new_shape:
            return orbit_
        else:
            for i, d in enumerate(new_shape):
                if d < orbit_.parameters['field_shape'][i]:
                    placeholder_orbit = placeholder_orbit.mode_truncation(d, axis=i)
                elif d > orbit_.parameters['field_shape'][i]:
                    placeholder_orbit = placeholder_orbit.mode_padding(d, axis=i)
                else:
                    pass
            return placeholder_orbit.convert(to=orbit_.state_type, inplace=True)
    else:
        return orbit_
