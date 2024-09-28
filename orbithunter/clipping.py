import numpy as np


__all__ = ["clip", "clipping_mask"]


def clip(orbit_instance, window_dimensions, **kwargs):
    """Create Orbit instance whose state array is a subdomain of the provided Orbit.

    Parameters
    ----------
    orbit_instance : Orbit
        The orbit whose state the subdomain is extracted from.
    window_dimensions : tuple of tuples.
        Contains one tuple for each continuous dimension, each defining the interval of the dimension to slice out.

    kwargs : dict
        Keyword arguments for Orbit instantiate.

    Returns
    -------
    Orbit :
        Orbit whose state and parameters reflect the subdomain defined by the provided dimensions.

    Notes
    -----
    The intervals provided refer to the :meth:`Orbit._plotting_dimensions()` method. The motivation
    here is to allow for clipping using visualization techniques as a direct guide.
    If a dimension has zero extent; i.e. equilibrium in that dimension, then the corresponding window_dimension
    tuple must be passed as (None, None).

    Examples
    --------

    Extract subdomain from an Orbit

    >>> orb = Orbit(state=np.ones([128, 128, 128, 128]), basis='physical',
    ...                              parameters=(100, 100, 100, 100))
    >>> one_sixteeth_subdomain_orbit = clip(orb, ((0, 50), (0, 50), (0, 50), (0, 50)))

    It is 1/16th the size because it takes half of the points in 4 different dimensions.



    """
    clipping_type = kwargs.get("clipping_type", orbit_instance.__class__)
    slices, dimensions = _slices_from_window(orbit_instance, window_dimensions)

    # It of course is better to get the dimensions/parameters from the clipping directly, but if the user wants to
    # this gives them the ability to override.
    parameters = kwargs.pop(
        "parameters",
        tuple(
            dimensions[i] if i < len(dimensions) else p
            for i, p in enumerate(orbit_instance.parameters)
        ),
    )

    clipped_orbit = clipping_type(
        state=orbit_instance.transform(to=orbit_instance.bases_labels()[0]).state[
            slices
        ],
        basis=orbit_instance.bases_labels()[0],
        parameters=parameters,
        **kwargs
    )
    return clipped_orbit


def clipping_mask(orbit_instance, *windows, invert=True):
    """
    Produce an array mask which shows the clipped regions corresponding to windows upon plotting.

    Parameters
    ----------
    orbit_instance : Orbit
        An instance whose state is to be masked.
    windows : list or tuple
        An iterable of window tuples; see Notes below.
    invert : bool
        Whether to logically invert the boolean mask; equivalent to showing the "interior" or "exterior" of the clipping
        if True or False, respectively.

    Returns
    -------
    Orbit :
        Orbit instance whose state is a numpy masked array.

    """
    # Create boolean mask to manipulate for numpy masked arrays.
    mask = np.zeros(orbit_instance.shapes()[0]).astype(bool)
    if isinstance(*windows, tuple) and len(windows) == 1:
        windows = tuple(*windows)

    if type(windows) in [list, tuple]:
        for window in windows:
            # Do not need dimensions, as we are not clipping technically.
            window_slices, _ = _slices_from_window(orbit_instance, window)
            mask[window_slices] = True
    else:
        # Do not need dimensions, as we are not clipping technically.
        window_slices, _ = _slices_from_window(orbit_instance, windows)
        mask[window_slices] = True

    if invert:
        mask = np.invert(mask)
    masked_field = np.ma.masked_array(
        orbit_instance.transform(to=orbit_instance.bases_labels()[0]).state, mask=mask
    )
    return orbit_instance.__class__(
        state=masked_field,
        basis=orbit_instance.bases_labels()[0],
        parameters=orbit_instance.parameters,
    )


def _slices_from_window(orbit_instance, window_dimensions):
    """
    Slices for Orbit state which represents the subdomain defined by provided window_dimensions

    Parameters
    ----------
    orbit_instance : Orbit
        The orbit instance whose state will be sliced.
    window_dimensions : tuple of tuples
        A tuple containing the intervals which define the dimensions of the new slice.

    Returns
    -------
    tuple, tuple :
        Tuples containing the slices for the Orbit state and the new corresponding dimensions

    """
    shape = orbit_instance.shapes()[0]
    # Returns the dimensions which would be shown on a plot (easier to eye-ball clipping then), including units.
    # Should be a tuple of tuples (d_min, d_max), one for each dimension.
    plot_dimensions = orbit_instance.plotting_dimensions()
    # Returns a tuple of the "length" > 0 of each axis.
    actual_dimensions = orbit_instance.dimensions()

    clipping_slices = []
    clipping_dimensions = []
    for i, (d_min, d_max) in enumerate(window_dimensions):
        # the division and multiplication by 2's keeps the sizes even.
        if d_min is None:
            slice_start = 0
        else:
            # Clipping out of bounds does not make sense.
            assert (
                d_min >= plot_dimensions[i][0]
            ), "Trying to clip out of bounds. Please revise clipping domain."
            # Some coordinate axis range from, for example, -1 to 1. Account for this by rescaling the interval.
            # An example clipping in this case could be from -1 to -0.5. To handle this, rescale the plotting dimensions
            # to [0, plotting_dimension_max - plotting_dimension_min], by subtracting the minimum.
            # This makes the d_min value equal to the fraction of the domain.

            rescaled_domain_min = (d_min - plot_dimensions[i][0]) / (
                plot_dimensions[i][1] - plot_dimensions[i][0]
            )
            slice_start = int(shape[i] * rescaled_domain_min)

        if d_max is None:
            slice_end = shape[i]
        else:
            assert (
                d_max <= plot_dimensions[i][1]
            ), "Trying to clip out of bounds. Please revise clipping domain."
            rescaled_domain_max = (d_max - plot_dimensions[i][0]) / (
                plot_dimensions[i][1] - plot_dimensions[i][0]
            )
            slice_end = int(shape[i] * rescaled_domain_max)

        # Apply a transformation if increasing the index corresponds to the negative dimension direction.
        if not orbit_instance.positive_indexing()[i]:
            slice_start = shape[i] - slice_start
            slice_end = shape[i] - slice_end
            slice_start, slice_end = slice_end, slice_start

        if np.mod(slice_end - slice_start, 2):
            # If the difference is odd, then floor dividing and multiplying by two switches whichever is odd to even.
            # By definition, only one can be odd if the difference is odd; hence only once number is changing.
            slice_start, slice_end = 2 * (slice_start // 2), 2 * (slice_end // 2)
        clipping_slices.append(slice(slice_start, slice_end))

        # Find the correct fraction of the length>0 then subtract the minimum to rescale back to original plot units.
        ith_clipping_dim = (
            int(np.abs(slice_end - slice_start)) / shape[i]
        ) * actual_dimensions[i]
        clipping_dimensions.append(ith_clipping_dim)

    return tuple(clipping_slices), tuple(clipping_dimensions)
