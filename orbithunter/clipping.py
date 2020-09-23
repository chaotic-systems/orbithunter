from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import numpy as np

__all__ = ['clip', 'mask_orbit']


def _slices_from_window(orbit_, window_dimensions, time_ordering='decreasing'):
    field_shape = orbit_.field_shape
    # Returns the dimensions which would be shown on a plot (easier to eye-ball clipping then), including units.
    # Should be a tuple of tuples (d_min, d_max), one for each dimension.
    plot_dimensions = orbit_.plotting_dimensions
    # Returns a tuple of the "length" > 0 of each axis.
    actual_dimensions = orbit_.dimensions

    clipping_slices = []
    clipping_dimensions = []
    for i, (d_min, d_max) in enumerate(window_dimensions):


        # the division and multiplication by 2's keeps the sizes even.
        if d_min is None:
            slice_start = None
            d_min = plot_dimensions[i][0]
        else:
            # Clipping out of bounds does not make sense.
            assert d_min >= plot_dimensions[i][0], 'Trying to clip out of bounds. Please revise clipping domain.'
            # Some coordinate axis range from, for example, -1 to 1. Account for this by rescaling the interval.
            # An example clipping in this case could be from -1 to -0.5. To handle this, rescale the plotting dimensions
            # to [0, plotting_dimension_max - plotting_dimension_min], by subtracting the minimum.
            slice_start = int(2*((field_shape[i] * ((d_min - plot_dimensions[i][0])
                                                    / (plot_dimensions[i][1] - plot_dimensions[i][0])) + 1)//2))
            # Time is always taken as "up" so T=0 is actually the LAST row of the first axis.
            if i == 0 and time_ordering == 'decreasing':
                if slice_start != 0:
                    slice_start *= -1
                else:
                    slice_start = -1
        if d_max is None:
            slice_end = None
            d_max = plot_dimensions[i][1]
        else:
            # Again,
            assert d_max <= plot_dimensions[i][1], 'Trying to clip out of bounds. Please revise clipping domain.'
            slice_end = int(2*((field_shape[i] * ((d_max - plot_dimensions[i][0])
                                                  / (plot_dimensions[i][1] - plot_dimensions[i][0])) + 1)//2))
            if i == 0 and time_ordering == 'decreasing':
                slice_end *= -1
                if np.mod(np.abs(slice_end - slice_start), 2):
                    slice_end -= 1
        # Find the correct fraction of the length>0 then subtract the minimum to rescale back to original plot units.
        actual_max_dimension = (-1.0 * plot_dimensions[i][0]) + (actual_dimensions[i] * (d_max - plot_dimensions[i][0])
                                / (plot_dimensions[i][1] - plot_dimensions[i][0]))
        actual_min_dimension = (-1.0 * plot_dimensions[i][0]) + (actual_dimensions[i] * (d_min - plot_dimensions[i][0])
                                / (plot_dimensions[i][1] - plot_dimensions[i][0]))
        if i == 0:
            slice_start, slice_end = slice_end, slice_start

        clipping_slices.append(slice(slice_start, slice_end))
        clipping_dimensions.append(actual_max_dimension - actual_min_dimension)

    return tuple(clipping_slices), tuple(clipping_dimensions)


def clip(orbit_, window_dimensions, **kwargs):
    """ Take subdomain from Orbit instance. Aperiodic by definition.



    Parameters
    ----------
    orbit_ : Orbit
        Instance to clip from
    window_dimensions : tuple of tuples.
        Contains one tuple for each continuous field dimension, each of the form (Dimension_minimum, Dimension_maximum).
    kwargs :

    Returns
    -------

    Notes
    -----
    For Kuramoto-Sivashinsky, the window_dimensions would be of the form ((T_min, T_max,), (X_min, X_max)).
    Originally contemplated allowing window_dimensions to be iterable of windows but if this is desired then
    just iterate outside the function. I think that is more reasonable and cleaner.

    """

    slices, dimensions = _slices_from_window(orbit_, window_dimensions)
    orbit_parameters = tuple(dimensions[i] if i < len(dimensions) else p for i, p in enumerate(orbit_.orbit_parameters))
    clipped_orbit = orbit_.__class__(state=orbit_.convert(to='field').state[slices], state_type='field',
                                     orbit_parameters=orbit_parameters, **kwargs).convert(to=orbit_.state_type)
    return clipped_orbit


def mask_orbit(orbit_, window_dimensions, mask_region='exterior'):
    slices, dimensions = _slices_from_window(orbit_, window_dimensions)
    mask = np.zeros(orbit_.shape).astype(bool)
    mask[slices] = True
    if mask_region == 'exterior':
        mask = np.invert(mask)
    masked_field = np.ma.masked_array(orbit_.convert(to='field').state, mask)
    return orbit_.__class__(state=masked_field, state_type='field', orbit_parameters=orbit_.orbit_parameters)
