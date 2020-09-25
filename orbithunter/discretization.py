import numpy as np

__all__ = ['rediscretize', 'rediscretize_tiling_dictionary', 'parameter_based_discretization']


def correct_aspect_ratios(array_of_orbits, axis=0):

    iterable_of_dims = [o.dimensions[axis] for o in array_of_orbits.ravel()]
    iterable_of_shapes = [o.field_shape[axis] for o in array_of_orbits.ravel()]

    disc_total = np.sum(iterable_of_shapes)
    dim_total = np.sum(iterable_of_dims)

    fraction_array = np.array(iterable_of_dims) / dim_total
    new_discretization_sizes = (2 * np.round((fraction_array * disc_total) / 2)).astype(int)
    number_of_dimensionless_orbits = np.sum(fraction_array == 0)
    number_of_dimensionful_orbits = len(new_discretization_sizes) - number_of_dimensionless_orbits

    # Due to how the fractions and rounding can work out the following is a safety measure to ensure the
    # new discretization sizes add up to the original; this is a requirement in order to glue strips together.
    # The simplest way of adjusting these is to simply make every even (it already should be and I actually cannot
    # think of a case where this would not be true, but still), then add or subtract from the largest discretization
    # until we get to the original total.
    while np.sum(new_discretization_sizes) != disc_total:
        new_discretization_sizes[np.mod(np.sum(new_discretization_sizes), 2) == 1] += 1
        if np.sum(new_discretization_sizes) < disc_total:
            # Add points to the first instance of the minimum discretization size not equal to 0.
            non_zero_minimum = new_discretization_sizes[new_discretization_sizes > 0].min()
            new_discretization_sizes[np.min(np.where(new_discretization_sizes == non_zero_minimum)[0])] += 2
        else:
            # non_zero_maximum is same as maximum, just a precaution and consistency with minimum
            non_zero_maximum = new_discretization_sizes[new_discretization_sizes > 0].max()
            # Using np.argmin here would return the index relative to the sliced disc_size array, not the original as
            # we desire.
            new_discretization_sizes[np.min(np.where(new_discretization_sizes == non_zero_maximum)[0])] -= 2

    fraction_array = np.array(new_discretization_sizes) / np.sum(new_discretization_sizes)

    # Can't have orbits with 0 discretization but don't want them too large because they have no inherent scale.
    # If there are only orbits of this type however we need to return the original sizes.
    if number_of_dimensionless_orbits == len(iterable_of_shapes):
        # if gluing all equilibria in time, don't change anything.
        new_shapes = [o.shape for o in array_of_orbits.ravel()]

    elif number_of_dimensionless_orbits > 0 and number_of_dimensionful_orbits >= 1:
        # The following is the most convoluted piece of code perhaps in the entire package. It attempts to find the
        # best minimum discretization size for the dimensionless orbits; simply having them take the literal minimum
        # size is as good as throwing them out.
        # If the number of elements in the strip is large, then it is possible that the previous defaults will result
        # in 0's.
        half_dimless_disc = np.min([np.max([2 * (iterable_of_shapes[0] // (2*number_of_dimensionless_orbits
                                                                           * number_of_dimensionful_orbits)),
                                    disc_total // (len(iterable_of_shapes)*number_of_dimensionless_orbits),
                                    2*int((np.round(1.0 / np.min(fraction_array[fraction_array != 0]))+1) // 2)]),
                                    iterable_of_shapes[0]//2])

        # Find the number of points to take from each orbit. Multiply by two to get an even total.
        how_much_to_take_away_for_each = 2 * np.round(half_dimless_disc * fraction_array)
        # Subtract the number of points to take away from each per dimensionless
        new_discretization_sizes = (new_discretization_sizes
                                             - (number_of_dimensionless_orbits * how_much_to_take_away_for_each))
        # The size of the dimensionless orbits is of course how much we took from the other orbits.
        new_discretization_sizes[new_discretization_sizes == 0] = np.sum(how_much_to_take_away_for_each).astype(int)

        # The new shapes of each orbit are the new sizes along the gluing axis, and the originals for axes not being
        # glued.
        new_shapes = [tuple(int(new_discretization_sizes[j]) if i == axis else o.field_shape[i]
                        for i in range(len(o.shape))) for j, o in enumerate(array_of_orbits.ravel())]

    else:
        new_shapes = [tuple(int(new_discretization_sizes[j]) if i == axis else o.field_shape[i]
                      for i in range(len(o.shape))) for j, o in enumerate(array_of_orbits.ravel())]

    # Return the strip of orbits with corrected proportions.
    return np.array([rediscretize(o, new_shape=shp) for o, shp in zip(array_of_orbits.ravel(), new_shapes)])


def parameter_based_discretization(parameters, **kwargs):
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
    equation = kwargs.get('equation', 'ks')
    if equation == 'ks':
        T, L = parameters[:2]
        if kwargs.get('N', None) is None:
            if resolution == 'coarse':
                N = np.max([2*(int(2**(np.log2(T+1)-2))//2), 16])
            elif resolution == 'fine':
                N = np.max([2*(int(2**(np.log2(T+1)+4))//2), 32])
            else:
                N = np.max([2*(int(2**(np.log2(T+1)))//2), 16])
        else:
            N = kwargs.get('N', None)

        if kwargs.get('M', None) is None:
            if resolution == 'coarse':
                M = np.max([2*(int(2**(np.log2(L+1)-1))//2), 16])
            elif resolution == 'fine':
                M = np.max([2*(int(2**(np.log2(L+1))+2)//2), 32])
            else:
                M = np.max([2*(int(2**(np.log2(L+1)) + 1)//2), 16])
        else:
            M = kwargs.get('M', None)

        return N, M
    else:
        return None


def rediscretize(orbit_, **kwargs):
    # Return class object with new discretization size. Not performed in place
    # because it is used in other functions; don't want user to be caught unawares
    # Copy state information to new orbit; don't perform operations inplace, only create new orbit
    equation = kwargs.get('equation', 'ks')
    if equation == 'ks':
        placeholder_orbit = orbit_.convert(to='field').copy().convert(to='modes', inplace=True)
        if kwargs.get('new_shape', None) is None:
            new_shape = parameter_based_discretization(orbit_.orbit_parameters, **kwargs)
        else:
            new_shape = kwargs.get('new_shape')

        if orbit_.field_shape == new_shape:
            return orbit_
        else:
            for i, d in enumerate(new_shape):
                if d < orbit_.field_shape[i]:
                    placeholder_orbit = placeholder_orbit._truncate(d, axis=i)
                elif d > orbit_.field_shape[i]:
                    placeholder_orbit = placeholder_orbit._pad(d, axis=i)
                else:
                    pass
            return placeholder_orbit.convert(to=orbit_.state_type, inplace=True)
    else:
        return orbit_


def rediscretize_tiling_dictionary(tiling_dictionary, **kwargs):
    orbits = list(tiling_dictionary.values())
    new_shape = kwargs.get('new_shape', None)

    if new_shape is None:
        average_dimensions = [np.mean(x) for x in tuple(zip(*(o.dimensions for o in orbits)))]
        new_shape = parameter_based_discretization(average_dimensions, **kwargs)

    return {td_key: rediscretize(td_val, new_shape=new_shape) for td_key, td_val in tiling_dictionary.items()}
