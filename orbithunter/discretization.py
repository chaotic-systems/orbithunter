import numpy as np

__all__ = ['rediscretize', 'rediscretize_tiling_dictionary', 'parameter_based_discretization']


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
            if T in [0, 0.]:
                N = 1
            elif resolution == 'coarse':
                N = np.max([2*(int(2**(np.log2(T+1)-2))//2), 16])
            elif resolution == 'fine':
                N = np.max([2*(int(2**(np.log2(T+1)+4))//2), 32])
            elif resolution == 'power':
                N = np.max([2**(int(np.log2(T))), 16])
            else:
                N = np.max([4*int(T**(1./2.)), 16])
        else:
            N = kwargs.get('N', None)

        if kwargs.get('M', None) is None:
            if resolution == 'coarse':
                M = np.max([2*(int(2**(np.log2(L+1)-1))//2), 16])
            elif resolution == 'fine':
                M = np.max([2*(int(2**(np.log2(L+1))+2)//2), 32])
            elif resolution == 'power':
                M = np.max([2**(int(np.log2(L))+1), 16])
            else:
                M = np.max([6*int(L**(1./2.)), 16])
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
            new_shape = parameter_based_discretization(orbit_.parameters, **kwargs)
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
            return placeholder_orbit.convert(to=orbit_.basis, inplace=True)
    else:
        return orbit_


def rediscretize_tiling_dictionary(tiling_dictionary, **kwargs):
    orbits = list(tiling_dictionary.values())
    new_shape = kwargs.get('new_shape', None)

    if new_shape is None:
        average_dimensions = [np.mean(x) for x in tuple(zip(*(o.dimensions for o in orbits)))]
        new_shape = parameter_based_discretization(average_dimensions, **kwargs)

    return {td_key: rediscretize(td_val, new_shape=new_shape) for td_key, td_val in tiling_dictionary.items()}
