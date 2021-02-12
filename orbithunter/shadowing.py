import numpy as np

__all__ = ['scan', 'shadow', 'cover', 'scanning_mask']


def amplitude_difference(base_slice_orbit, window_orbit, *args, **kwargs):
    return (base_slice_orbit**2 - window_orbit**2).norm()


def absolute_threshold(score_array, threshold):
    mask = np.zeros(score_array.shape, dtype=bool)
    mask[np.where(score_array <= threshold)] = True
    return mask


def first_nprimes(num_primes):
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
    n = 2
    prime_codes = []
    while len(prime_codes) < num_primes:
        sieve = np.ones(n//3 + (n % 6 == 2), dtype=np.bool)
        sieve[0] = False
        for i in range(int(n**0.5)//3+1):
            if sieve[i]:
                k = 3*i+1 | 1
                sieve[(k**2//3)::2*k] = False
                sieve[(k**2+4*k-2*k*(i & 1))//3::2*k] = False
        prime_codes = np.r_[2, 3, ((3*np.nonzero(sieve)[0]+1) | 1)]
        n += 1
    return prime_codes


def factor_prime_mask(mask, primes):
    factors = []
    # return a collection of covering_masks, each for one of the primes contained in the covering_mask.
    # Integer entries upon inverse logarithm
    expcovering_mask = 10**mask
    rounded = np.round(expcovering_mask)
    # Do not want the places where 0 (uncovering_masked)
    for p in primes:
        factor_covering_mask = np.zeros(expcovering_mask.shape, dtype=np.int32)
        factor_covering_mask[np.where((rounded//p).astype(float) == rounded/p)] = p
        factors.append(factor_covering_mask)
    return factors


def scanning_mask(masked_scores, base_orbit, window_orbit, strides):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The base of the shadowing computations
    window_orbit : Orbit
        The window of the shadowing computations
    masked_scores : ndarray
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
    mask_pivot_tuples = zip(*np.where(masked_scores))
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

    return base_orbit_mask


def scan(base_orbit, window_orbit, *args, **kwargs):
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
            Takes two orbit arguments, the slice of the base orbit and the window it is being compared to.

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
    score_type = kwargs.get('score_type', 'pointwise')
    # Easiest way I've found to account for periodicity is to pad with wrapped values
    window = window_orbit.state
    base = base_orbit.state
    # To get the padding/wrap number, need to see how much the windows extend "beyond" the base orbit. This can be
    # computed using the placement of the last pivot and the window dimensions. This is only relevant for periodic
    # dimensions in the base orbit. Not specifying min_persistence makes memory requirements very large.
    gudhikwargs = kwargs.get('gudhi_kwargs', {'periodic_dimensions': tuple(len(window.shape)*[False]),
                                              'min_persistence': 0.01})
    dimension_iterator = zip(base.shape, strides, window.shape, gudhikwargs.get('periodic_dimensions'))

    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    score_array_shape = []
    pad_shape = []
    for base_extent, stride_extent, window_extent, periodic in dimension_iterator:
        # Need to determine the score array and the padding amount, if periodic in that dimension.
        # First pivot position is always at (0, 0), opposite corner is (base_extent-1, base_extent-1)
        # This is really the index of the last pivot along current dimension.
        if periodic:
            num_pivots = (base_extent-1) // stride_extent
            spill_over = (window_extent + stride_extent * num_pivots) - base_extent
            # All pivot points (separated by stride_extent) are admissible with respect to periodic dimensions.
            score_array_shape.append(base_extent // stride_extent)
            # -1 is a correction for starting at 0 indexing.
            pad_shape.append(spill_over)
        else:
            # If we need padding then the pivot position does not work for aperiodic dimensions. If window extent
            # is large compared to stride then multiple pivots could be out of bounds. Need to reel back until we find
            # one that is not.
            num_pivots = (base_extent - window_extent) // stride_extent
            score_array_shape.append(num_pivots)
            pad_shape.append(0)

    padding = tuple((0, pad) if pad > 0 else (0, 0) for pad in pad_shape)
    pbase = np.pad(base, padding, mode='wrap')

    # First create an array to store metric values; 1/stride is the sampling fraction, evenly spaced.
    # Want to create the number of pivots based on original array, but to handle "overflow", the wrapped array
    # is used for calculations
    score_array = np.zeros(score_array_shape)
    if isinstance(kwargs.get('mask', None), np.ndarray):
        mask = kwargs.get('mask', None).astype(bool)
        iterator = (x for i, x in enumerate(np.ndindex(score_array.shape)) if not mask.ravel()[i])
    else:
        iterator = np.ndindex(score_array.shape)

    if score_type == 'persistence':
        # Scoring using persistent homology requires the persistence of the complex of both the base orbit slice
        # and the windowing orbit. This is fundamentally different than pointwise scoring but to avoid boilerplate
        # code it is included here instead of in its own function.
        # Functions to compute and score the persistences must be provided by user.
        scoring_function = kwargs.get('scoring_function')
        persistence_function = kwargs.get('persistence_function')
        cached_persistences = kwargs.get('cache', {})
        # The persistences that are saved depend on the shape of the slices at the corresponding pivot points.
        # Use tuples as keys.
        if not cached_persistences.get(window.shape, []):
            base_persistences = []
            # by definition base slices cannot be periodic unless w_dim == b_dim and that dimension is periodic.
            for pivot_tuple in iterator:
                # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
                # the window's dimensions.
                window_slices = []
                # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
                for pivot, span, stride in zip(pivot_tuple, window.shape, strides):
                    window_slices.append(slice(pivot * stride, pivot * stride + span))
                base_slice_orbit = window_orbit.__class__(**{**vars(window_orbit),
                                                             'state': pbase[tuple(window_slices)]})
                base_persistences.append(persistence_function(base_slice_orbit, **gudhikwargs))
            # Once the persistences are computed for the pivots for a given slice shape, can include them in the cache.
            cached_persistences[window.shape] = base_persistences
        else:
            if kwargs.get('verbose', False):
                print('Using cached persistence values for base orbit, slices of shape {}'.format(window.shape))
            base_persistences = cached_persistences.get(window.shape, [])
        # Periodicity of windows determined by the periodic_dimensions() method.
        window_persistence = persistence_function(window_orbit, **gudhikwargs)
        # .ravel() returns a view, therefore can use it to broadcast.
        # Unfortunately what I want to do is only supported if the base slice and window have same periodic dimensions?
        for i, bp in enumerate(base_persistences):
            score_array.ravel()[i] = scoring_function(bp, window_persistence, **gudhikwargs)
        return score_array, cached_persistences
    else:
        # Pointwise scoring can use the amplitude difference default.
        scoring_function = kwargs.get('scoring_function', amplitude_difference)
        cache = kwargs.get('cache', {})

        # Because the number of computations is built directly into the size of score_array,
        # can us a numpy iterator. This also allows us to know exactly which window corresponds to which
        # value, as ndindex will always return the same order of indices.

        for pivot_tuple in iterator:
            # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
            # the window's dimensions.
            window_slices = []
            # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
            for pivot, span, stride in zip(pivot_tuple, window.shape, strides):
                window_slices.append(slice(pivot * stride, pivot * stride + span))

            base_slice_orbit = window_orbit.__class__(**{**vars(window_orbit), 'state': pbase[tuple(window_slices)]})
            score_array[pivot_tuple] = scoring_function(base_slice_orbit, window_orbit, *args, **kwargs)

        # for uniformity only. the second argument doesn't do anything in this case.
        return score_array, cache


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


def cover(base_orbit, thresholds, window_orbits, mask_type='union', **kwargs):
    """ Function to perform multiple shadowing computations given a collection of orbits.

    Parameters
    ----------
    base_orbit : Orbit
        The Orbit to be "covered"
    thresholds : determined by masking_function
        The threshold for the masking function. Typically numerical in nature but need not be. The reason
        why threshold does not have a default value even though the functions DO have defaults is because the
        statistics of the scoring function depend on the base orbit and window orbits.  
    window_orbits : array_like of Orbits
        Orbits to cover the base_orbit with. Typically a group orbit, as threshold is a single constant. Handling
        group orbits not included because it simply is another t
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

    strides = kwargs.get('strides', tuple([1]*len(base_orbit.shapes()[0])))
    cache = kwargs.get('cache', {})
    masking_function = kwargs.get('masking_function', absolute_threshold)
    # Whether or not previous runs have "cached" the persistence information; not really caching, mind you.
    # This doesn't do anything unless either mask_type=='family' or score_type=='persistence'. One or both must be true.

    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    if mask_type == 'prime':
        # May cause memory issues if base_orbit is huge, but I do not want to get into sparse arrays at the moment.
        orbit_mask = np.zeros(base_orbit.shape, dtype=np.float)
        # Need unique identifiers if the union is to be separable into different window covers.
        prime_codes = first_nprimes(len(window_orbits))
        for code, threshold, window in zip(prime_codes, thresholds, window_orbits):
            # potentially cache the persistences of base orbit slices if score_type=='persistence'.
            scores, cache = scan(base_orbit, window, **{**kwargs, 'cache': cache})
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
            # However, to avoid overflow, use the summation of logarithms instead.
            orbit_mask += np.log10(orbit_mask_code)
    elif mask_type == 'count':
        # May cause memory issues if base_orbit is huge, but I do not want to get into sparse arrays at the moment.
        orbit_mask = np.zeros(base_orbit.shape, dtype=np.int32)
        for threshold, window in zip(thresholds, window_orbits):
            scores, cache = scan(base_orbit, window, **{**kwargs, 'cache': cache})
            # Need some numerical thresholding value; masking functions can be whatever user wants but
            # here it is assumed to be the upper bound.
            scores_bool = masking_function(scores, threshold)
            # Getting where score obeys the threshold, can now translate this back into positions in the orbit.
            orbit_mask_bool = scanning_mask(scores_bool, base_orbit, window, strides)
            # The cumulative union of all shadowings.
            orbit_mask += orbit_mask_bool.astype(int)
    else:
        # Temporary sorting so that windows only get progressively smaller. This can technically provide inexact
        # results if windows cannot be ordered by size, uniquely, as it does not repeat calculations if the pivot points
        # have had a successive score. The idea here being that if an orbit is detected at a position then that
        # position is occupied by that orbit; no other orbit can be there.
        window_size_order = np.argsort([w.size for w in window_orbits])[::-1]
        scores_bool = None
        for threshold, window in zip(thresholds[window_size_order], window_orbits[window_size_order]):
            scores, cache = scan(base_orbit, window, **{**kwargs, 'cache': cache, 'mask': scores_bool})
            if scores_bool is None:
                scores_bool = masking_function(scores, threshold)
            else:
                scores_bool = np.logical_or(scores_bool, masking_function(scores, threshold))
        # Getting where score obeys the threshold, can now translate this back into positions in the orbit.
        # Because the union is happening at the pivot level, no need to continuously update the orbit_mask with unions.
        orbit_mask = scanning_mask(scores_bool, base_orbit, window, strides).astype(bool)
    return orbit_mask