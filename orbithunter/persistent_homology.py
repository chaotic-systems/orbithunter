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
    persistence_kwargs = {'homology_coeff_field': kwargs.get('homology_coeff_field', 2),
                          'min_persistence': kwargs.get('min_persistence', -1)}
    return orbit_complex(orbit_, **kwargs).persistence(**persistence_kwargs)


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
    persistence_kwargs = {'homology_coeff_field': kwargs.get('homology_coeff_field', 2),
                          'min_persistence': kwargs.get('min_persistence', -1)}
    diagram1 = np.array([p1[-1] for p1 in orbit_persistence(orbit1, **persistence_kwargs)])
    diagram2 = np.array([p2[-1] for p2 in orbit_persistence(orbit2, **persistence_kwargs)])
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