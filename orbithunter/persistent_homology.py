import gudhi as gh
import matplotlib.pyplot as plt

__all__ = ['orbit_periodic_cubical_complex', 'gudhi_plot', 'gudhi_distance']


def orbit_periodic_cubical_complex(orbit_instance,  **kwargs):
    """ Wrapper for Gudhi persistent homology package

    Parameters
    ----------
    orbit_instance : Orbit
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
    persistence_ : list
        list of pairs(dimension, pair(birth, death)) – the persistence of the complex.
    Notes
    -----
    Doesn't support vector fields for now, I think each component would have its own cubical complex?
    """
    orbit_instance = orbit_instance.convert(to='field')
    periodic_dimensions = kwargs.pop('periodic_dimensions', tuple(len(orbit_instance.field_shape)*[True]))
    cubical_complex = gh.PeriodicCubicalComplex(dimensions=orbit_instance.state.shape,
                                                top_dimensional_cells=orbit_instance.state.ravel(),
                                                periodic_dimensions=periodic_dimensions)
    return cubical_complex


def gudhi_plot(persistence_, method='diagram', **gudhi_kwargs):
    """
    Parameters
    ----------
    persistence_ : iterable
    Iterable of length N that contains elements of length 2 (i.e. N x 2 array) containing the persistence intervals
    (birth, death)
    method : str
    Takes one of the following values: 'diagram', 'barcode', 'density'.
    gudhi_kwargs :
    kwargs related to gudhi plotting functions. See Gudhi docs for details.

    Returns
    -------

    """
    if method == 'diagram':
        gh.plot_persistence_diagram(persistence_, **gudhi_kwargs)
    elif method == 'barcode':
        gh.plot_persistence_barcode(persistence_, **gudhi_kwargs)
    elif method == 'density':
        gh.plot_persistence_diagram(persistence_, **gudhi_kwargs)
    else:
        raise ValueError('Gudhi plotting method not recognized.')
    plt.show()
    return None


def gudhi_distance(orbit1, orbit2, metric='bottleneck_distance', **kwargs):
    persistence1 = orbit_periodic_cubical_complex(orbit1, **kwargs).persistence()
    persistence2 = orbit_periodic_cubical_complex(orbit2, **kwargs).persistence()
    diagram1 = [p[-1] for p in persistence1]
    diagram2 = [p[-1] for p in persistence2]
    if metric == 'bottleneck_distance':
        distance_func = gh.bottleneck.bottleneck_distance
    elif metric == 'hera_bottleneck_distance':
        distance_func = gh.hera.bottleneck_distance
    else:
        distance_func = gh.hera.wasserstein_distance
    return distance_func(*(diagram1, diagram2))
