import gudhi as gh
import matplotlib.pyplot as plt
import inspect
import numpy as np
from gudhi.hera import wasserstein_distance, bottleneck_distance

__all__ = ["orbit_complex", "orbit_persistence", "gudhi_plot", "gudhi_distance"]


def orbit_complex(orbit_, **kwargs):
    """ Wrapper for Gudhi persistent homology package

    Parameters
    ----------
    orbit_ : Orbit
        The orbit whose persistent homology will be computed.

    min_persistence

    kwargs :
        boundary_conditions : tuple
        Contains bools which flag which axes of the orbit's field are assumed to be periodic for the persistence
        calculations. i.e. for Kuramoto-Sivashinsky, boundary_conditions=(False, True) would make time aperiodic
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
    boundary_conditions = kwargs.get(
        "boundary_conditions", orbit_.boundary_conditions()
    )
    cubical_complex = gh.PeriodicCubicalComplex(
        dimensions=orbit_.state.shape,
        top_dimensional_cells=orbit_.state.ravel(),
        boundary_conditions=boundary_conditions,
    )
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
    persistence_kwargs = {
        # 'homology_coeff_field': kwargs.get('homology_coeff_field', len(orbit_.dimensions())),
        "min_persistence": kwargs.get("min_persistence", 0.0)
    }
    complex_kwargs = {
        "boundary_conditions": kwargs.get(
            "boundary_conditions", orbit_.boundary_conditions()
        )
    }
    persistence = orbit_complex(orbit_, **complex_kwargs).persistence(
        **persistence_kwargs
    )
    if kwargs.get("persistence_format", "numpy") == "numpy":
        return np.array([[x[0], x[1][0], x[1][1]] for x in persistence])
    else:
        return persistence


def gudhi_plot(orbit_, gudhi_method="diagram", **kwargs):
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
    orbit_persistence_ = orbit_persistence(
        orbit_, **{**kwargs, "persistence_format": "gudhi"}
    )
    plot_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.getfullargspec(gh.plot_persistence_diagram).args
    }
    if gudhi_method in ["diagram", "barcode", "density"]:
        gh.plot_persistence_diagram(orbit_persistence_, **plot_kwargs)
    else:
        raise ValueError("Gudhi plotting gudhi_method not recognized.")
    plt.show()


def gudhi_distance(orbit1, orbit2, gudhi_metric="bottleneck", **kwargs):
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
    diagram1 = orbit_persistence(orbit1, **kwargs)[:, 1:]
    diagram2 = orbit_persistence(orbit2, **kwargs)[:, 1:]
    if gudhi_metric == "bottleneck":
        distance_func = bottleneck_distance
    elif gudhi_metric == "wasserstein":
        distance_func = wasserstein_distance
    else:
        raise ValueError(f"{gudhi_metric} not recognized as gudhi metric.")
    return distance_func(diagram1, diagram2)


def gudhi_distance_from_persistence(
    orbit_persistence1,
    orbit_persistence2,
    gudhi_metric="bottleneck",
    with_betti=True,
    **kwargs,
):
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
        diagram1 = np.array(orbit_persistence1)[:, 1:]
        diagram2 = np.array(orbit_persistence2)[:, 1:]
    else:
        diagram1 = orbit_persistence1
        diagram2 = orbit_persistence2

    if gudhi_metric == "bottleneck":
        distance_func = bottleneck_distance
    elif gudhi_metric == "wasserstein":
        distance_func = wasserstein_distance
    else:
        raise ValueError(f"{gudhi_metric} not recognized as gudhi metric.")
    return distance_func(diagram1, diagram2)
