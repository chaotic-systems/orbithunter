from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import numpy as np

__all__ = ['clip', 'mask_orbit']


def _slices_from_window(orbit_, window_dimensions, time_ordering='decreasing'):
    field_shape = orbit_.field_shape
    dimensions = orbit_.dimensions

    clipping_slices = []
    clipping_dimensions = []
    for i, (d_min, d_max) in enumerate(window_dimensions):
        # the division and multiplication by 2's keeps the sizes even.
        if (d_min is not None) or (d_min not in [0, 0.]):
            slice_start = int(2*((field_shape[i] * d_min / dimensions[i]+1)//2))
            # Time is always taken as "up" so T=0 is actually the LAST row of the first axis.
            if i == 0 and time_ordering == 'decreasing':
                slice_start *= -1
        else:
            slice_start = None
            d_min = 0

        if d_max is not None or d_max != dimensions[i]:
            slice_end = int(2*((field_shape[i] * d_max / dimensions[i]+1)//2))
            if i == 0 and time_ordering == 'decreasing':
                slice_end *= -1
        else:
            slice_end = None
            d_max = dimensions[i]

        if i == 0:
            slice_start, slice_end = slice_end, slice_start

        clipping_slices.append(slice(slice_start, slice_end))
        clipping_dimensions.append(d_max - d_min)

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
    params = dict(zip(orbit_.dimensions, dimensions))
    clipped_orbit = orbit_.__class__(state=orbit_.convert(to='field').state[slices],
                                     state_type='field', orbit_parameters=params, **kwargs).convert(to=orbit_.state_type)
    return clipped_orbit


def mask_orbit(orbit_, window_dimensions):
    slices, dimensions = _slices_from_window(orbit_, window_dimensions)
    mask = np.zeros(orbit_.shape).astype(bool)
    mask[slices] = True
    if mask == 'exterior':
        mask = np.invert(mask)
    masked_field = np.ma.masked_array(orbit_.convert(to='field').state, mask)
    return orbit_.__class__(state=masked_field, state_type='field', orbit_parameters=orbit_.paramaters)
