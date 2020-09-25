import gudhi as gh
import matplotlib.pyplot as plt

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
    orbit_ = orbit_.convert(to='field')
    periodic_dimensions = kwargs.pop('periodic_dimensions', tuple(len(orbit_.field_shape)*[True]))
    cubical_complex = gh.PeriodicCubicalComplex(dimensions=orbit_.state.shape,
                                                top_dimensional_cells=orbit_.state.ravel(),
                                                periodic_dimensions=periodic_dimensions)
    return cubical_complex


def orbit_persistence(orbit_, **kwargs):
    """

    Parameters
    ----------
    orbit_complex

    Returns
    -------
    persistence_ :

    Notes
    -----
    Convenience function because of how Gudhi is structured.
    """
    return orbit_complex(orbit_).persistence(**kwargs)


def gudhi_plot(orbit_persistence_, method='diagram', **gudhi_kwargs):
    """
    Parameters
    ----------
    persistence_ : iterable
        Iterable of length N that contains elements of length 2 (i.e. N x 2 array) containing the persistence intervals
        (birth, death)
    method : str
        Plotting method. Takes one of the following values: 'diagram', 'barcode', 'density'.
    gudhi_kwargs :
        kwargs related to gudhi plotting functions. See Gudhi docs for details.
    """
    if method == 'diagram':
        gh.plot_persistence_diagram(orbit_persistence_, **gudhi_kwargs)
    elif method == 'barcode':
        gh.plot_persistence_barcode(orbit_persistence_, **gudhi_kwargs)
    elif method == 'density':
        gh.plot_persistence_diagram(orbit_persistence_, **gudhi_kwargs)
    else:
        raise ValueError('Gudhi plotting method not recognized.')
    plt.show()
    return None


def gudhi_distance(orbit1, orbit2, metric='bottleneck', **kwargs):
    """ Compute the distance between two Orbits' persistence diagrams.

    Parameters
    ----------
    orbit1 : Orbit
    orbit2 : Orbit
    metric : str
        The persistence diagram distance metric to use
    kwargs

    Returns
    -------

    """
    diagram1 = [p1[-1] for p1 in orbit_persistence(orbit1, **kwargs)]
    diagram2 = [p2[-1] for p2 in orbit_persistence(orbit2, **kwargs)]
    if metric == 'bottleneck':
        distance_func = gh.bottleneck.bottleneck_distance
    elif metric == 'hera_bottleneck':
        distance_func = gh.hera.bottleneck_distance
    elif metric == 'wasserstein':
        distance_func = gh.hera.wasserstein_distance
    else:
        raise ValueError('Distance metric not recognized as gudhi metric.')
    return distance_func(diagram1, diagram2, **kwargs)
