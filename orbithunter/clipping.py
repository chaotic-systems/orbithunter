from __future__ import print_function, division, absolute_import
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.argv[0], '../..')))
import numpy as np

__all__ = ['clip']


def _slices_from_window(orbit_, window_dimensions):
    field_shape = orbit_.parameters['field_shape']
    field_dimensions = orbit_.parameters['field_dimensions']

    slices = []
    dimensions = []
    for i, (d_min, d_max) in enumerate(window_dimensions):
        # the division and multiplication by 2's keeps the sizes even.
        if d_min is not None:
            slice_start = 2*((field_shape[i] * d_min / field_dimensions[i]+1)//2)
        else:
            slice_start = None

        if d_max is not None:
            slice_end = 2*((field_shape[i] * d_max / field_dimensions[i]+1)//2)
        else:
            slice_end = None

        slices.append(slice(slice_start, slice_end))
        dimensions.append(d_max - d_min)

    return tuple(slices), tuple(dimensions)


def clip(orbit_, window_dimensions, **kwargs):
    """ Take subdomain from Orbit instance. Aperiodic by definition.



    Parameters
    ----------
    orbit_
    window_dimensions : tuple of tuples.
    kwargs

    Returns
    -------

    """

    slices, dimensions = _slices_from_window(orbit_, window_dimensions)
    params = dict(zip(list(orbit_.parameters.keys())[:len(dimensions)], dimensions))
    clipped_orbit = orbit_.__class__(state=orbit_.convert(to='field').state[slices],
                                     state_type='field', parameters=params).convert(to=orbit_.state_type)

    return clipped_orbit


def mask_orbit(orbit_, window_dimensions, **kwargs):

    slices, dimensions = _slices_from_window(orbit_, window_dimensions)
    mask = np.zeros(orbit_.shape).astype(bool)
    mask[slices] = True
    if mask == 'exterior':
        mask = np.invert(mask)
    masked_field = np.ma.masked_array(orbit_.convert(to='field').state, mask)
    return orbit_.__class__(state=masked_field, state_type='field', parameters=orbit_.paramaters)
