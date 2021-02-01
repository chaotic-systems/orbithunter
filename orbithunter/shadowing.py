import numpy as np


def amplitude_difference(base_slice, window):
    return np.linalg.norm(base_slice**2 - window**2)

def orbit_mask_from_score_mask(base_orbit, window_orbit, score_array, strides):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The base of the shadowing computations
    window_orbit : Orbit
        The window of the shadowing computations
    score_mask : ndarray
        A numpy masked array produced by masking an array returned by shadowing.
    strides : tuple
        Tuple of int; the steps that window slides by in the shadowing computations.

    Returns
    -------
    np.ma.masked_array
        The masking of the same dimension as base_orbit's state that corresponds to the provided score mask.

    Notes
    -----
    Because of the wrap technique used in the shadowing function, need a function that converts or 'folds' the mask
    of the score array into one that can be applied to the base orbit. Need the shapes of the windows and strides to do
    so.
    """
    return None


def shadowing(base_orbit, window_orbit, **kwargs):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is transl
    verbose
    threshold
    threshold_type
    stride
    mask_region

    Returns
    -------
    tuple :

    Notes
    -----
    This function takes a n-dimensional (window) orbit and slides it around with respect to the "base" orbit.
    At each new position, some metric (typically difference in amplitudes) is computed. Again, typically, the smaller
    the metric value, to more accurately a region is shadowed by the window. Computing the metric at every
    single position is available but is very inefficient; the user should experiment to see how quickly
    their metric is changing and decide on how to sample or 'stride'/slide the window.

    Base orbit dimensions should be integer multiples of window dimensions for best results.
    """
    strides = kwargs.get('strides', tuple([1]*len(base_orbit.shapes()[0])))
    scoring_function = kwargs.get('scoring_function', amplitude_difference)

    # Easiest way I've found to account for periodicity is to pad with wrapped values
    window = window_orbit.state
    base = base_orbit.state
    pbase = np.pad(base, ((0, w) for w in window.shape), mode='wrap')

    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window is larger than the base orbit. resize first. '

    # First create an array to store metric values; 1/stride is the sampling fraction, evenly spaced.
    # Want to create the number of pivots based on original array, but to handle "overflow", the wrapped array
    # is used for calculations
    score_array = np.zeros([b // s for b, s, in zip(base.shape, strides)])

    # Because the number of computations is built directly into the size of score_array,
    # can us a numpy iterator. This also allows us to know exactly which window corresponds to which
    # value, as ndindex will always return the same order of indices.
    for pivot_tuple in np.ndindex(score_array.shape):
        # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
        # the window's dimensions.
        window_slices = tuple(slice(pivot * span, (pivot+1) * span) for pivot, span in zip(pivot_tuple, window.shape))
        score_array[pivot_tuple] = scoring_function(base[window_slices], window)

    return score_array


def cover(base_orbit, *window_orbits):
    """ Function to perform multiple shadowing computations given a collection of orbits.

    Parameters
    ----------
    base_orbit
    window_orbits

    Returns
    -------

    """
    return None

