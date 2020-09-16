import gudhi as gh


def orbit_persistence(orbit_instance, periodic_dimensions, min_persistence=0.1):
    """ Wrapper for Gudhi persistent homology package

    Parameters
    ----------
    orbit_instance
    periodic_dimensions
    min_persistence

    Returns
    -------

    Notes
    -----
    Doesn't support vector fields for now, I think each component would have its own cubical complex?
    """
    orbit_instance = orbit_instance.convert(to='field')
    cubical_complex = gh.PeriodicCubicalComplex(dimensions=orbit_instance.state.shape,
                                                top_dimensional_cells=orbit_instance.state.ravel(),
                                                periodic_dimensions=periodic_dimensions)

    persistence = cubical_complex.persistence(min_persistence=min_persistence)
    return cubical_complex, persistence



