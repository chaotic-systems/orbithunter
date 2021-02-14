import numpy as np

__all__ = ['scan', 'shadow', 'cover', 'scanning_mask']


def amplitude_difference(base_slice_orbit, window_orbit, *args, **kwargs):
    return (base_slice_orbit**2 - window_orbit**2).norm()


def absolute_threshold(score_array, threshold):
    mask = np.zeros(score_array.shape, dtype=bool)
    mask[np.where(score_array <= threshold)] = True
    return mask


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
    verbose = kwargs.get('verbose', False)
    pivots = kwargs.get('pivots', ())
    mask = kwargs.get('mask', None)
    # Caching doesn't really make sense for pointwise calculations but included here anyway.
    cache = kwargs.get('cache', {})
    caching = kwargs.get('caching', False)
    # Easiest way I've found to account for periodicity is to pad with wrapped values
    window = window_orbit.state
    base = base_orbit.state

    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    # To get the padding/wrap number, need to see how much the windows extend "beyond" the base orbit. This can be
    # computed using the placement of the last pivot and the window dimensions. This is only relevant for periodic
    # dimensions in the base orbit. Not specifying min_persistence makes memory requirements very large.

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(window.shape)*[False]))

    # the periodic_dimensions key here determines the periodic dimensions in the gudhi.PeriodicCubicalComplex
    gudhikwargs = kwargs.get('gudhi_kwargs', {'periodic_dimensions': tuple(len(window.shape)*[False]),
                                              'min_persistence': 0.01})
    dimension_iterator = zip(base.shape, strides, window.shape, base_orbit_periodicity)
    score_array_shape = []
    pad_shape = []
    for base_extent, stride_extent, window_extent, periodic in dimension_iterator:
        # Need to determine the score array and the padding amount, if periodic in that dimension.
        # First pivot position is always at (0, 0), opposite corner is (base_extent-1, base_extent-1)
        # This is really the index of the last pivot along current dimension.
        if periodic:
            num_pivots = ((base_extent-1) // stride_extent) + 1
            spill_over = (window_extent + stride_extent * (num_pivots-1)) - base_extent
            # All pivot points (separated by stride_extent) are admissible with respect to periodic dimensions.
            score_array_shape.append(num_pivots)
            # -1 is a correction for starting at 0 indexing.
            pad_shape.append(spill_over)
        else:
            # If we need padding then the pivot position does not work for aperiodic dimensions. If window extent
            # is large compared to stride then multiple pivots could be out of bounds. Need to reel back until we find
            # one that is not.
            # Extra 1 because iterator provides the index not the number of pivots.
            num_pivots = 1 + (base_extent - window_extent) // stride_extent
            score_array_shape.append(num_pivots)
            pad_shape.append(0)

    score_array = np.zeros(score_array_shape)
    padding = tuple((0, pad) if pad > 0 else (0, 0) for pad in pad_shape)
    pbase = np.pad(base, padding, mode='wrap')
    if not pivots:
        pivots = np.ndindex(score_array.shape)

    # Create the pivot iterator; the sites which need to be checked for detection.
    if isinstance(mask, np.ndarray):
        mask = mask.astype(bool)
        n_pivots = mask.size - mask.sum()
        # If mask == True then that indicates a detection without need to repeat calculations.
        iterator = (x for x in pivots if not mask[x])
    else:
        n_pivots = score_array.size
        iterator = pivots

    if score_type == 'persistence':
        # Scoring using persistent homology requires the persistence of the complex of both the base orbit slice
        # and the windowing orbit. Computing either of these can be time consuming, so the ability to provide
        # cached persistence scores for the base orbit (there is only one windowing orbit per function call).
        # Functions to compute and score the persistences must be provided by user.
        scoring_function = kwargs.get('scoring_function')
        persistence_function = kwargs.get('persistence_function')
        window_persistence = persistence_function(window_orbit, **gudhikwargs)
        # The persistences that are saved depend on the shape of the slices at the corresponding pivot points.
        # Use tuples as keys.
        # The persistences of the base orbit only depend on the size of the subdomain being used. Therefore, this
        # is the primary key of any potential caching dict.
        base_persistences_dict = cache.get(window.shape, {})
        for i, pivot_tuple in enumerate(iterator):
            if verbose and i % max([1, n_pivots//10]) == 0:
                print('-', end='')
            # Required to cache the persistence with the correct key.
            base_pivot_tuple = tuple(stride*p for p, stride in zip(pivot_tuple, strides))
            # If the current pivot doesn't have a stored value, then calculate it and add it to the cache.
            # Pivot tuples iterate over the score_array, not the actual base orbit, need to convert.
            if not base_persistences_dict.get(base_pivot_tuple, {}):
                # by definition base slices cannot be periodic unless w_dim == b_dim and that dimension is periodic.
                # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
                # the window's dimensions.
                window_slices = []
                # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
                for pivot, span, stride in zip(pivot_tuple, window.shape, strides):
                    window_slices.append(slice(pivot * stride, pivot * stride + span))
                base_slice_orbit = window_orbit.__class__(**{**vars(window_orbit),
                                                             'state': pbase[tuple(window_slices)]})
                bp = persistence_function(base_slice_orbit, **gudhikwargs)
                if caching:
                    # This gives the ability to use a cache but not add to it. Update the cache at the end
                    # using dict unpacking.
                    base_persistences_dict[base_pivot_tuple] = bp
            else:
                bp = base_persistences_dict[base_pivot_tuple]
            score_array[pivot_tuple] = scoring_function(bp, window_persistence, **gudhikwargs)

        if caching:
            cache[window.shape] = {**cache.get(window.shape, {}), **base_persistences_dict}
        # Once the persistences are computed for the pivots for a given slice shape, can include them in the cache.

        # Periodicity of windows determined by the periodic_dimensions() method.
        # .ravel() returns a view, therefore can use it to broadcast.
        # Unfortunately what I want to do is only supported if the base slice and window have same periodic dimensions?
        return score_array, cache
    else:
        # Pointwise scoring can use the amplitude difference default.
        scoring_function = kwargs.get('scoring_function', amplitude_difference)

        # Because the number of computations is built directly into the size of score_array,
        # can us a numpy iterator. This also allows us to know exactly which window corresponds to which
        # value, as ndindex will always return the same order of indices.

        for i, pivot_tuple in enumerate(iterator):
            if verbose and i % max([1, n_pivots//10]) == 0:
                print('-', end='')
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


def cover(base_orbit, thresholds, window_orbits, replacement=False, cover_type='mask', **kwargs):
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
    replacement : bool
        If False, then once a position in the score array is filled by a detection

    Returns
    -------
    covering_masks : dict
        Dict whose keys are the index positions of the windows in the window_orbits provided, whose values
        are the ndarrays containing either the scores or boolean mask of the

    Notes
    -----
    There are three main strategies for how the scoring is returned.

    """

    cache = kwargs.get('cache', {})
    masking_function = kwargs.get('masking_function', absolute_threshold)
    # Whether or not previous runs have "cached" the persistence information; not really caching, mind you.
    # This doesn't do anything unless either mask_type=='family' or score_type=='persistence'. One or both must be true.

    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    size_order = np.argsort([w.size for w in window_orbits])[::-1]
    mask = None
    covering_masks = {}
    for index, threshold, window in zip(size_order, thresholds[size_order], window_orbits[size_order]):
        if kwargs.get('verbose', False) and len(thresholds) % max([1, len(thresholds)//10]) == 0:
            print('#', end='')
        # Just in case of union method
        scores, cache = scan(base_orbit, window, **{**kwargs, 'cache': cache, 'mask': mask})
        scores_bool = masking_function(scores, threshold)
        print(index, scores_bool.sum())
        if not replacement:
            if mask is not None:
                # The "new" detections
                scores_bool = np.logical_xor(scores_bool, mask)
                # all detections
                mask = np.logical_or(scores_bool, mask)
            else:
                mask = scores_bool.copy()

        # If there were no detections then do not add anything to the cover.
        if scores_bool.sum() > 0:
            if cover_type == 'scores':
                covering_masks[index] = scores.copy()
            else:
                covering_masks[index] = scores_bool.copy()
    return covering_masks
