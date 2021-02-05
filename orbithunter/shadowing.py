import numpy as np

__all__ = ['scan', 'shadow', 'cover', 'scanning_mask']


def amplitude_difference(base_slice, window):
    return np.linalg.norm(base_slice**2 - window**2)


def absolute_threshold(score_array, threshold):
    mask = np.zeros(score_array.shape, dtype=bool)
    mask[np.where(score_array) <= threshold] = True
    return mask


def _nprimes(n):
    """ Returns a array of primes, 2 <= p < n

    n : int
        Upper search range for primes.

    Returns
    -------
    ndarray :
        Array with primes

    Notes
    -----
    https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    Using this to encode the orbits for covering.
    """
    sieve = np.ones(n//3 + (n % 6 == 2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k = 3*i+1 | 1
            sieve[(k**2//3)::2*k] = False
            sieve[(k**2+4*k-2*k*(i & 1))//3::2*k] = False
    return np.r_[2, 3, ((3*np.nonzero(sieve)[0]+1) | 1)]


def scanning_mask(scores, base_orbit, window_orbit, strides, show='interior'):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The base of the shadowing computations
    window_orbit : Orbit
        The window of the shadowing computations
    scores : ndarray
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

    This function originated from the fact that conditional statements applied to the score array returned by shadowing
    function can be arbitrarily complex, but the end result, taking an array of bools and applying it to
    the base orbit is still the same.
    """
    # Get the positions of the True values in scoring array mask
    mask_pivot_tuples = zip(*np.where(scores))
    grid = np.indices(window_orbit.shape)
    base_orbit_mask = np.zeros(base_orbit.shape, dtype=bool)

    # For each window pivot (equivalently, "top left" corner) compute the indices within the span of the window.
    # Once these indices have been determined, the corresponding values in the orbit mask can be set to True.
    for pivot_tuple in mask_pivot_tuples:
        # pivot grid is a tuple of coordinates, the first dimension's indices are stored in the first element, etc.
        # Each pivot_grid element is the same dimension as the window. The indices are essentially the pivot + window
        # indices.
        pivot_grid = tuple(g + p*s for g, p, s in zip(grid, pivot_tuple, strides))
        # To account for wrapping, fold indices which extend beyond the base orbit's extent using periodicity.
        for coordinates, base_extent in zip(pivot_grid, base_orbit.shape):
            coordinates[coordinates >= base_extent] -= base_extent
        base_orbit_mask[pivot_grid] = True

    if show == 'interior':
        return np.invert(base_orbit_mask)
    else:
        return base_orbit_mask


def scan(base_orbit, window_orbit, **kwargs):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is transl
    kwargs :
        strides : tuple
            The steps in between samplings/scoring calculations. A stride of 4 would mean that the window
            moves by 4 places after each computation.
        scoring_function : callable
            Takes two arguments, the slice of the base orbit and the window it is being compared to.

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
    # To get the padding/wrap number, need to see how much the windows extend "beyond" the base orbit. This can be
    # computed using the placement of the last pivot and the window dimensions.
    padding_dims = [w + (s * ((b-1) // s)) - b for b, s, w in zip(base.shape, strides, window.shape)]
    padding = tuple((0, pad) if pad > 0 else 0 for pad in padding_dims)
    pbase = np.pad(base, padding, mode='wrap')

    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

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
        window_slices = []
        # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
        for pivot, span, stride in zip(pivot_tuple, window.shape, strides):
            window_slices.append(slice(pivot * stride, pivot * stride + span))
        score_array[pivot_tuple] = scoring_function(pbase[tuple(window_slices)], window)
    return score_array


def shadow(base_orbit, window_orbit, threshold, **kwargs):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is translated around)
    threshold : Orbit
        The threshold value for the masking function to use to decide which regions are shadowing, based on the
        scoring function passed to scan.
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
    masking_function = kwargs.get('masking_function', absolute_threshold)

    scores = scan(base_orbit, window_orbit, **kwargs)
    scores = masking_function(scores, threshold)
    orbit_mask_bool = scanning_mask(scores, base_orbit, window_orbit, strides)
    return orbit_mask_bool


def cover(base_orbit, threshold, *window_orbits, mask_type='bool', **kwargs):
    """ Function to perform multiple shadowing computations given a collection of orbits.

    Parameters
    ----------
    base_orbit : Orbit
        The Orbit to be "covered"
    threshold : determined by masking_function
        The threshold for the masking function. Typically numerical in nature but need not be. The reason
        why threshold does not have a default value even though the functions DO have defaults is because the
        statistics of the scoring function depend on the base orbit and window orbits.  
    window_orbits : array_like of Orbits
        Orbits to cover the base_orbit with.
    mask_type : str
        The type of mask to return, takes values 'prime' and 'bool'. 'prime' returns an array whose elements
        are products of primes, each prime value representing a different orbit; the first n primes for n windows
        are the values used. Overlaps can be determined by the prime factorization of the masking array.
    Returns
    -------
    mask : ndarray
        A mask which is either the logical union of all single window mask values, or whose values are the product of
        the different prime categorical codes corresponding to each window orbit. Usage of prime numbers is to
        be able to identify overlaps.


    """
    if len(window_orbits) == 1 and isinstance(*window_orbits, tuple):
        window_orbits = tuple(*window_orbits)

    strides = kwargs.get('strides', tuple([1]*len(base_orbit.shapes()[0])))
    scoring_function = kwargs.get('scoring_function', amplitude_difference)
    masking_function = kwargs.get('masking_function', absolute_threshold)

    if mask_type == 'prime':
        # May cause memory issues if base_orbit is huge, but I do not want to get into sparse arrays at the moment.
        orbit_mask = np.ones(base_orbit.shape, dtype=np.int32)
        # Need unique identifiers if the union is to be separable into different window covers.
        prime_codes = _nprimes(len(window_orbits))
        for code, window in zip(prime_codes, window_orbits):
            scores = scan(base_orbit, window, scoring_function=scoring_function, strides=strides)
            # Need some numerical thresholding value; masking functions can be whatever user wants but
            # here it is assumed to be the upper bound.
            scores_bool = masking_function(scores, threshold)
            # Getting where score obeys the threshold, can now translate this back into positions in the orbit.
            orbit_mask_bool = scanning_mask(scores_bool, base_orbit, window, strides)
            orbit_mask_code = code * orbit_mask_bool
            # So that the product with the current mask doesn't destroy previous values. This makes the
            # orbit mask in terms of code value and 1, instead of 0 and 1.
            orbit_mask_code[orbit_mask_code == 0.] = 1
            # Once the mask for the orbit has been established, then we can take the product of the current mask
            orbit_mask *= (code * orbit_mask_bool)
    else:
        # May cause memory issues if base_orbit is huge, but I do not want to get into sparse arrays at the moment.
        orbit_mask = np.ones(base_orbit.shape, dtype=bool)
        for window in window_orbits:
            scores = scan(base_orbit, window, scoring_function=scoring_function, strides=strides)
            # Need some numerical thresholding value; masking functions can be whatever user wants but
            # here it is assumed to be the upper bound.
            scores_bool = masking_function(scores, threshold)
            # Getting where score obeys the threshold, can now translate this back into positions in the orbit.
            orbit_mask_bool = scanning_mask(scores_bool, base_orbit, window, strides)
            # The cumulative union of all shadowings.
            orbit_mask = np.logical_or(orbit_mask, orbit_mask_bool)
            
    return orbit_mask

