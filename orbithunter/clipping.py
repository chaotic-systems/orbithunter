import numpy as np


__all__ = ['clip', 'clipping_mask']


def _slices_from_window(orbit_, window_dimensions, time_ordering='decreasing'):
    """ Get slices

    Parameters
    ----------
    orbit_
    window_dimensions
    time_ordering : str
        The convention for orbithunter is that time is always in decreasing order, this makes the positive time
        direction "up", an artifact

    Returns
    -------

    """
    shape = orbit_.shapes()[0]
    # Returns the dimensions which would be shown on a plot (easier to eye-ball clipping then), including units.
    # Should be a tuple of tuples (d_min, d_max), one for each dimension.
    plot_dimensions = orbit_.plotting_dimensions()
    # Returns a tuple of the "length" > 0 of each axis.
    actual_dimensions = orbit_.dimensions()

    clipping_slices = []
    clipping_dimensions = []
    for i, (d_min, d_max) in enumerate(window_dimensions):
        # the division and multiplication by 2's keeps the sizes even.
        if d_min is None:
            slice_start = 0
        else:
            # Clipping out of bounds does not make sense.
            assert d_min >= plot_dimensions[i][0], 'Trying to clip out of bounds. Please revise clipping domain.'
            # Some coordinate axis range from, for example, -1 to 1. Account for this by rescaling the interval.
            # An example clipping in this case could be from -1 to -0.5. To handle this, rescale the plotting dimensions
            # to [0, plotting_dimension_max - plotting_dimension_min], by subtracting the minimum.
            # This makes the d_min value equal to the fraction of the domain.

            rescaled_domain_min = (d_min-plot_dimensions[i][0])/(plot_dimensions[i][1]-plot_dimensions[i][0])
            slice_start = int(shape[i] * rescaled_domain_min)

        if d_max is None:
            slice_end = shape[i]
        else:
            assert d_max <= plot_dimensions[i][1], 'Trying to clip out of bounds. Please revise clipping domain.'
            rescaled_domain_max = (d_max-plot_dimensions[i][0])/(plot_dimensions[i][1]-plot_dimensions[i][0])
            slice_end = int(shape[i] * rescaled_domain_max)

        if i == 0 and time_ordering == 'decreasing':
            # From the "top down convention for time.
            slice_start = shape[i] - slice_start
            slice_end = shape[i] - slice_end
            slice_start, slice_end = slice_end, slice_start

        if np.mod(slice_end-slice_start, 2):
            # If the difference is odd, then floor dividing and multiplying by two switches whichever is odd to even.
            # By definition, only one can be odd if the difference is odd; hence only once number is changing.
            slice_start, slice_end = 2*(slice_start // 2), 2*(slice_end // 2)
        clipping_slices.append(slice(slice_start, slice_end))

        # Find the correct fraction of the length>0 then subtract the minimum to rescale back to original plot units.
        ith_clipping_dim = (int(np.abs(slice_end - slice_start))/shape[i]) * actual_dimensions[i]
        clipping_dimensions.append(ith_clipping_dim)

    return tuple(clipping_slices), tuple(clipping_dimensions)


def clip(orbit_, window_dimensions, **kwargs):
    """ Take subdomain of field from Orbit instance. Aperiodic by definition.
    Parameters
    ----------
    orbit_ : Orbit
        Instance to clip from
    window_dimensions : tuple of tuples.
        Contains one tuple for each continuous field dimension, each of the form (Dimension_minimum, Dimension_maximum).
        If a dimension has zero extent; i.e. equilibrium in that dimension, then the corresponding window_dimension
        tuple must be passed as (None, None).
    clipping_class : Orbit type class
        Class to put the clipping state into.
    kwargs :

    Returns
    -------

    Notes
    -----
    This function allows for slicing based on providing the dimensions of the window; slicing of the Orbit state array
    is done directly via the __getitem__ method; this updates the dimensions as well.

    For Kuramoto-Sivashinsky, the window_dimensions would be of the form ((T_min, T_max), (X_min, X_max)).
    Originally contemplated allowing window_dimensions to be iterable of windows but if this is desired then
    just iterate outside the function. I think that is more reasonable and cleaner.

    """
    clipping_class = kwargs.get('clipping_class', orbit_.__class__)
    slices, dimensions = _slices_from_window(orbit_, window_dimensions)

    # It of course is better to get the dimensions/parameters from the clipping directly, but if the user wants to
    # this gives them the ability to override.
    parameters = kwargs.pop('parameters',
                            tuple(dimensions[i] if i < len(dimensions) else p for i, p in enumerate(orbit_.parameters)))

    clipped_orbit = clipping_class(state=orbit_.transform(to=orbit_.bases()[0]).state[slices], basis=orbit_.bases()[0],
                                   parameters=parameters, **kwargs)
    return clipped_orbit


def clipping_mask(orbit_, windows, mask_region='exterior'):
    """

    Parameters
    ----------
    orbit_ :
    windows : list


    mask_region : str
        takes values 'exterior' or 'interior', masks the corresponding regions relative to the windows.

    Returns
    -------
    Orbit :
        Orbit instance whose state is a numpy masked array.
    Notes
    -----

    `window tuples' are tuples whose elements are d-dimensional tuples
    indicating the dimensions which define the window. i.e.
    for window_tuple = ((0, 10), (0, 5)) would mean to mask the values outside the subdomain defined
    by t=(0,10) x=(0,5) (example for KS equation).
    """
    # Create boolean mask to manipulate for numpy masked arrays.
    mask = np.zeros(orbit_.shapes()[0]).astype(bool)
    if type(windows) in [list, tuple]:
        for window in windows:
            # Do not need dimensions, as we are not clipping technically.
            window_slices, _ = _slices_from_window(orbit_, window)
            mask[window_slices] = True
    else:
        # Do not need dimensions, as we are not clipping technically.
        window_slices, _ = _slices_from_window(orbit_, windows)
        mask[window_slices] = True

    if mask_region == 'exterior':
        mask = np.invert(mask)
    masked_field = np.ma.masked_array(orbit_.transform(to=orbit_.bases()[0]).state, mask=mask)
    return orbit_.__class__(state=masked_field, basis=orbit_.bases()[0], parameters=orbit_.parameters)
