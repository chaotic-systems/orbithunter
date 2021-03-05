import gudhi as gh
import matplotlib.pyplot as plt
import inspect
import numpy as np

__all__ = ['orbit_complex', 'orbit_persistence', 'gudhi_plot', 'gudhi_distance']


def orbit_complex(orbit_,  **kwargs):
    """ Wrapper for Gudhi persistent homology package

    Parameters
    ----------
    orbit_ : Orbit
        The orbit whose persistent homology will be computed.

    min_persistence

    kwargs :
        periodic_dimensions : tuple
        Contains bools which flag which axes of the orbit's field are assumed to be periodic for the persistence
        calculations. i.e. for Kuramoto-Sivashinsky, periodic_dimensions=(False, True) would make time aperiodic
        and space periodic for the construction of the PeriodicCubicalComplex. Generalizes to any dimension.

    Returns
    -------
    cubical_complex : PeriodicCubicalComplex
    complex_ : list
        list of pairs(dimension, pair(birth, death)) – the persistence of the complex.
    Notes
    -----
    I do not think orbithunter support vector fields for now, I think each component would have its own cubical complex?
    """
    periodic_dimensions = kwargs.get('periodic_dimensions', orbit_.periodic_dimensions())
    cubical_complex = gh.PeriodicCubicalComplex(dimensions=orbit_.state.shape,
                                                top_dimensional_cells=orbit_.state.ravel(),
                                                periodic_dimensions=periodic_dimensions)
    return cubical_complex


def orbit_persistence(orbit_, **kwargs):
    """

    Parameters
    ----------
    orbit_

    Returns
    -------
    persistence_ :

    Notes
    -----
    Convenience function because of how Gudhi is structured.
    """
    persistence_kwargs = {'homology_coeff_field': kwargs.get('homology_coeff_field', len(orbit_.dimensions())),
                          'min_persistence': kwargs.get('min_persistence', -1)}
    complex_kwargs = {'periodic_dimensions': kwargs.get('periodic_dimensions', orbit_.periodic_dimensions())}
    return orbit_complex(orbit_, **complex_kwargs).persistence(**persistence_kwargs)


def gudhi_plot(orbit_, gudhi_method='diagram', **kwargs):
    """
    Parameters
    ----------
    persistence_ : iterable
        Iterable of length N that contains elements of length 2 (i.e. N x 2 array) containing the persistence intervals
        (birth, death)
    gudhi_method : str
        Plotting gudhi_method. Takes one of the following values: 'diagram', 'barcode', 'density'.
    gudhi_kwargs :
        kwargs related to gudhi plotting functions. See Gudhi docs for details.
    """
    orbit_persistence_ = orbit_persistence(orbit_, **kwargs)
    plot_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(gh.plot_persistence_diagram).args}
    if gudhi_method == 'diagram':
        gh.plot_persistence_diagram(orbit_persistence_, **plot_kwargs)
    elif gudhi_method == 'barcode':
        gh.plot_persistence_barcode(orbit_persistence_, **plot_kwargs)
    elif gudhi_method == 'density':
        gh.plot_persistence_diagram(orbit_persistence_, **plot_kwargs)
    else:
        raise ValueError('Gudhi plotting gudhi_method not recognized.')
    plt.show()
    return None


def gudhi_distance(orbit1, orbit2, gudhi_metric='bottleneck', **kwargs):
    """ Compute the distance between two Orbits' persistence diagrams.

    Parameters
    ----------
    orbit1 : Orbit
    orbit2 : Orbit
    gudhi_metric : str
        The persistence diagram distance gudhi_metric to use
    kwargs

    Returns
    -------

    """
    diagram1 = np.array([p1[-1] for p1 in orbit_persistence(orbit1, **kwargs)])
    diagram2 = np.array([p2[-1] for p2 in orbit_persistence(orbit2, **kwargs)])
    if gudhi_metric == 'bottleneck':
        distance_func = gh.bottleneck.bottleneck_distance
    elif gudhi_metric == 'hera_bottleneck':
        distance_func = gh.hera.bottleneck_distance
    elif gudhi_metric == 'wasserstein':
        distance_func = gh.hera.wasserstein_distance
    else:
        raise ValueError('Distance gudhi_metric not recognized as gudhi gudhi_metric.')
    return distance_func(diagram1, diagram2)


def gudhi_distance_from_persistence(orbit_persistence1, orbit_persistence2,
                                    gudhi_metric='bottleneck', with_betti=True, **kwargs):
    """ Compute the distance between two Orbits' persistence diagrams.

    Parameters
    ----------
    orbit1 : Orbit
    orbit2 : Orbit
    gudhi_metric : str
        The persistence diagram distance gudhi_metric to use
    kwargs

    Returns
    -------
    float : 
        Distance gudhi_metric computed with respect to two provided orbit persistences
        
    Notes
    -----
    It is often more efficient to calculate the persistences en masse and store them; the other distance function
    does not account for this and I do not want to do type checking for persistence objects from Gudhi. 
    """
    if with_betti:
        diagram1 = np.array([p1[-1] for p1 in orbit_persistence1])
        diagram2 = np.array([p2[-1] for p2 in orbit_persistence2])
    else:
        diagram1 = orbit_persistence1
        diagram2 = orbit_persistence2

    if gudhi_metric == 'bottleneck':
        distance_func = gh.bottleneck.bottleneck_distance
    elif gudhi_metric == 'hera_bottleneck':
        distance_func = gh.hera.bottleneck_distance
    elif gudhi_metric == 'wasserstein':
        distance_func = gh.hera.wasserstein_distance
    else:
        raise ValueError('Distance gudhi_metric not recognized as gudhi metric.')
    return distance_func(diagram1, diagram2)


def persistence_array_converter(persistence, pivot_tuple):
    if isinstance(persistence, np.ndarray):
        gudhi_persistence = []
        persistence_slice = slice_persistence_array(persistence, pivot_tuple)
        for row in persistence_slice:
            betti = row[-3]
            interval = row[-2:]
            gudhi_persistence.append((betti, tuple(interval)))
        return gudhi_persistence
    else:
        persistence_array = np.concatenate([np.concatenate(([row[0]],
                                                            row[-1])) for row in persistence]).reshape(-1, 3)
        persistence_count = persistence_array.shape[0]
        # Go backwards so pivot tuple can be concatenated yet read from left to right.
        for pivot in pivot_tuple[::-1]:
            persistence_array = np.concatenate((np.array([pivot]*persistence_count).reshape(-1, 1),
                                                persistence_array), axis=1)
        return persistence_array


def slice_persistence_array(persistence_array, *pivots):
    """ Slice numpy array containing collection of persistences.

    Parameters
    ----------
    persistence_array
    pivots

    Returns
    -------

    Notes
    -----
    Note that this method, while faster than other, looks for all pivots which are in the set of values
    in each dimension.
    # Because there are so many pivots, and there is no uniformity in the persistence sizes, slicing is actually
    # the fastest query method. This only works for "rectangular" slicings; i.e. those of the form [::i, ::j]

    The work around for 'irregular' collections of pivots is to call the function with a single pivot at a time.
    """
    if len(pivots) == 1 and isinstance(*pivots, tuple):
        pivots = list(*pivots)
    else:
        pivots = list(pivots)

    dim = len(persistence_array[0, :-3])
    pivots = np.array(pivots).reshape(-1, dim)
    for i in range(dim):
        persistence_array = persistence_array[np.isin(persistence_array[:, i], pivots[:, i])]
    return persistence_array


def orbit_persistence_array(base_orbit, window_orbit, strides, scanning_shapes, persistence_function, **kwargs):
    window = window_orbit.state
    base = base_orbit.state
    score_array_shape, pad_shape = scanning_shapes

    padding = tuple((0, pad) if pad > 0 else (0, 0) for pad in pad_shape)
    pbase = np.pad(base, padding, mode='wrap')

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.

    # the periodic_dimensions key here determines the periodic dimensions in the gudhi.PeriodicCubicalComplex
    gudhikwargs = kwargs.get('gudhi_kwargs', {'periodic_dimensions': tuple(len(window.shape)*[False]),
                                              'min_persistence': 0.01})
    verbose = kwargs.get('verbose', False)
    pivots = kwargs.get('pivots', None)
    persistence_array = None
    if pivots is None:
        pivots = np.ndindex(score_array_shape)
        base_pivot_tuples = tuple(tuple(strides[i] * p for i, p in enumerate(piv)) for piv in pivots)
        n_pivots = len(base_pivot_tuples)
    else:
        base_pivot_tuples = tuple(pivots)
        n_pivots = len(base_pivot_tuples)

    for i, base_pivot_tuple in enumerate(base_pivot_tuples):
        # Required to cache the persistence with the correct key.
        # If the current pivot doesn't have a stored value, then calculate it and add it to the cache.
        # Pivot tuples iterate over the score_array, not the actual base orbit, need to convert.
        if verbose and i % max([1, n_pivots//25]) == 0:
            print('#', end='')
        # by definition base slices cannot be periodic unless w_dim == b_dim and that dimension is periodic.
        # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
        # the window's dimensions.
        window_slices = []
        # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
        for base_pivot, span in zip(base_pivot_tuple, window.shape):
            window_slices.append(slice(base_pivot, base_pivot + span))
        base_slice_orbit = window_orbit.__class__(**{**vars(window_orbit),
                                                     'state': pbase[tuple(window_slices)]})
        base_slice_persistence = persistence_function(base_slice_orbit, **gudhikwargs)
        persistence_array_slice = persistence_array_converter(base_slice_persistence, base_pivot_tuple)

        if persistence_array is None:
            persistence_array = persistence_array_slice.copy()
        else:
            persistence_array = np.concatenate((persistence_array, persistence_array_slice), axis=0)

    return persistence_array


