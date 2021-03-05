import numpy as np
import time

__all__ = ['score', 'shadow', 'fill', 'cover']


def amplitude_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice.state**2 - window.state**2)


def l2_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice.state - window.state)


def masked_l2_difference(base_slice, window, *args, **kwargs):
    base_mask = base_slice.copy().astype(bool)
    norm = np.linalg.norm(base_slice[base_mask] - window[base_mask])
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def masked_l2_difference_density(base_slice, window, *args, **kwargs):
    base_mask = base_slice.copy().astype(bool)
    norm = np.linalg.norm(base_slice[base_mask] - window[base_mask])
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def l2_difference(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    return np.linalg.norm(base_slice.state - window.state)


def l2_difference_density(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm(base_slice - window)
    norm_density = norm / base_slice.size
    return norm_density


def masked_l2_difference_mean_flow_correction_density(base_slice, window, *args, **kwargs):
    base_mask = base_slice.copy().astype(bool)
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm((base_slice[base_mask]-base_slice[base_mask].mean()) - window[base_mask])
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def l2_difference_mean_flow_correction(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    return np.linalg.norm((base_slice-base_slice.mean()) - window)


def l2_difference_mean_flow_correction_density(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm((base_slice-base_slice.mean()) - window)
    norm_density = norm / base_slice.size
    return norm_density


def masking_function(score_array, threshold):
    # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.
    mask = np.zeros(score_array.shape, dtype=bool)
    non_nan_coordinates = np.where(~np.isnan(score_array))
    under_threshold = score_array[non_nan_coordinates] <= threshold
    mask_indices = tuple(mask_coord[under_threshold] for mask_coord in non_nan_coordinates)
    mask[mask_indices] = True
    return mask


def inside_to_outside_iterator(padded_shape, base_shape, hull):
    """ Generator of pivot positions that iterate over the interior and then exterior.
    Parameters
    ----------
    padded_shape : tuple
        The shape of the array containing all possible pivots.
    base_shape
        The original orbit's array shape; the shape of the "interior" of the padded array.
    hull : tuple
        The shape of the convex hull (the smallest shape which can contain all windows in all dimensions).

    Yields
    ------
    tuple :
        The position of the pivot that will be used for the computation of the shadowing scoring metric.

    Notes
    -----
    An example with a three dimensional field: If the window has shape (8, 8, 8) and the pivot is at position
    (2, 4, 1) then the slice used for scoring is (slice(2, 10), slice(4, 12), slice(1, 9)).

    An example of "hull": If there are two windows with shapes (2, 8) and (24, 4) then hull = (24, 8).
    """
    for pivot_position in np.ndindex(base_shape):
        # Get the pivots for an array without padding, but add the padded dimensions to put pivots into the interior.
        yield tuple(p + (h-1) for p, h in zip(pivot_position, hull))

    # Windows completely consisting of padding are redundant; by definition these are pivots greater than
    # the base extent, therefore need only pivots which have at least one coordinate less than hull in each dimension.
    for pivot_position in np.ndindex(padded_shape):
        # If in exterior, then pivots are in the padding by definition.
        exterior_checker = tuple(p < (pad-1) for p, pad in zip(pivot_position, hull))
        if True in exterior_checker:
            # Want to work from inside to outside, but np.ndindex starts at 0's in each position. Therefore, need
            # to treat this as the distance away from the border of the interior region to work in the correct direction
            # hll - 2 because the width of the padding is always hll-1
            yield tuple((pad-2) - piv if piv < (pad-1) else piv for piv, pad in zip(pivot_position, hull))
        else:
            # If the pivot is not in the exterior, then skip it.
            continue


def pivot_iterator(padded_shape, base_shape, window_shape, hull, periodicity, **kwargs):
    """

    Parameters
    ----------
    padded_shape : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_shape : tuple of int
        Shape of the numpy array corresponding to the base orbit
    window_shape : tuple of int
        Shape of the numpy array corresponding to the windowing orbit.
    hull : tuple of int
        Shape of the convex hull of a set of windows (largest size in each dimension).
    periodicity : tuple of bool
        Elements indicate if corresponding axis is periodic. i.e. (True, False) implies axis=0 is periodic.
    kwargs : dict
        Keyword arguments for when user provides masking or pivot information.
        mask : np.ndarray or None
            numpy array, dtype boolean which indicates which pivots to mask. values of True imply exclusion of pivot.
        pivots : iterable
            Either a generator or iterable container containing the (unmasked) pivot positions to iterate over
        scanning_region : str
            Default 'all', if equals 'interior' then only permits pivots within the original base orbit's span.

    Yields
    ------
    tuple :
        Contains the positions of unmasked pivots, for whom scores will be computed.

    Notes
    -----
    Instead of constructing a generator for the desired pivots directly, it is easier to create the
    iterator for all points and then subset it using a mask, especially in the context where inside-out iteration
    is desired.

    Instead of returning a generator, yield from is used; in case of errors this function will be in the traceback
    """
    # If pivots provided, use those. Otherwise generate all possible pivots for padded array, inside to out order.
    pivots = kwargs.get('pivots', None) or inside_to_outside_iterator(padded_shape, base_shape, hull)
    # Masks can be passed in; the utility of this is coverings which have multiple steps
    if isinstance(kwargs.get('mask', None), np.ndarray):
        mask = kwargs.get('mask', None)
    else:
        mask = np.zeros(padded_shape, dtype=bool)

    # Not allowing partial matches along the boundary is the same as keeping the pivots within the interior.
    # Need to trim the pivots in the padding if aperiodic.
    for coordinates, periodic, base_size, window_size, pad_size in zip(np.indices(padded_shape), periodicity,
                                                                       base_shape, window_shape, hull):
        # Pivots with zero overlap are useless, remove those which are more than a window's length away from the
        # boundary in the prefix padding and all in the suffix padding
        mask = np.logical_or(mask, coordinates < ((pad_size-1) - (window_size-1)))
        mask = np.logical_or(mask, coordinates > (base_size + (pad_size-1)))
        if not periodic:
            if kwargs.get('scanning_region', 'exterior') == 'interior':
                # This excludes all pivots which cause windows to extend out of bounds.
                mask = np.logical_or(np.logical_or(mask, coordinates < pad_size-1),
                                     coordinates > (base_size + (pad_size-1) - window_size))
            elif kwargs.get('min_proportion', 0) != 0.:
                # This allows windows to extend out of bounds, up to a proportion of the window size.
                num_interior_points = int(window_size * kwargs.get('min_proportion', 0))
                mask = np.logical_or(np.logical_or(mask, coordinates < (pad_size - (window_size - num_interior_points)),
                                     coordinates > (base_size + (pad_size-1) - num_interior_points)))

        else:
            # If a dimension is periodic then all pivots are on the "interior", can iterate over periodic padding
            # and part of the interior to get all non-redundant information (as opposed to iterating over the
            # interior alone, from slice(pad_size, pad_size + base_size)).
            mask = np.logical_or(mask, coordinates > base_size)

    if isinstance(mask, np.ndarray):
        # If mask == True then do NOT calculate the score at that position; follows the np.ma.masked_array conventions.
        yield from (x for x in pivots if not mask[x])
    else:
        yield from pivots


def score(base_orbit, window_orbit, threshold, **kwargs):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is transl
    threshold : float
        The thresholding value used in masking_function to determine whether or not a score is deemed a "success". 
    kwargs :
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

    Old behavior with strides is now reproduced by masking.

    Only one window, no such thing as a convex hull here.
    """
    window = window_orbit.state
    base = base_orbit.state
    scoring_function = kwargs.get('scoring_function', l2_difference_mean_flow_correction)
    # Sometimes there are metrics (like persistence diagrams) which need only be computed once per window.
    if kwargs.get('window_caching_function', None) is not None:
        kwargs['window_cache'] = kwargs.get('window_caching_function', None)(window_orbit, **kwargs)
    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))
    hull = kwargs.get('convex_hull', None) or window.shape
    # This looks redundant but it is not because two padding calls with different modes are required. 
    periodic_padding = tuple((pad-1, pad-1) if bc else (0, 0) for bc, pad in zip(base_orbit_periodicity,  hull))
    aperiodic_padding = tuple((pad-1, pad-1) if not bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    # Pad the base orbit state to use for computing scores, create the score array which will contain each pivot's score
    padded_state = np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'), aperiodic_padding)
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit), 'state': padded_state})
    pivot_scores = np.zeros(padded_state.shape, dtype=float)#[tuple(slice(None, -h+1) for h in hull)]
    cached_pivots = []
    for each_pivot in pivot_iterator(padded_state.shape, base_orbit.shape, window_orbit.shape, hull,
                                     base_orbit_periodicity, **kwargs):
        if each_pivot in cached_pivots:
            raise RuntimeError('Iterating over previously scored pivot.')
        subdomain_slices = []
        for pivot, span, periodic in zip(each_pivot, hull, base_orbit_periodicity):
            if not periodic:
                # If aperiodic then we only want to score the points of the window which are on the interior.
                subdomain_slices.append(slice(max([pivot, span-1]), pivot + span))
            else:
                subdomain_slices.append(slice(pivot, pivot + span))
        # Slice out the subdomains which are on the "interior" of the calculation; this does nothing when pivots
        # are in the interior.
        base_orbit_subdomain = padded_base_orbit[tuple(subdomain_slices)]
        window_subdomain = window_orbit[tuple(slice(-shp, None) for shp in base_orbit_subdomain.shape)]
        # Compute the score for the pivot, store in the pivot score array.
        pivot_scores[each_pivot] = scoring_function(base_orbit_subdomain, window_subdomain, **kwargs)
        cached_pivots.append(each_pivot)

    # Return the scores and the masking of the scores for later use (possibly skip these pivots for future calculations)
    return pivot_scores, masking_function(pivot_scores, threshold)


def shadow(base_orbit, window_orbit, threshold, **kwargs):
    """

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is translated around)
    threshold : float
        The threshold value for the masking function to use to decide which regions are shadowing, based on the
        scoring function passed to scan.
    Returns
    -------
    tuple :

    Notes
    -----
    This takes the scores returned by the score function and converts them to the appropriate orbit sized array (mask).
    In the array returned by score, each array element is the value of the scoring function using that position as
    a "pivot". These score arrays are not the same shape as the base orbits, as periodic dimensions and windows
    which are partially out of bounds are allowed. The idea being that you may accidentally cut a shadowing region
    in half when clipping, but still want to capture it numerically.

    Therefore, it is important to score at every possible position and so there needs to be a function which converts
    these scores into the appropriately sized (the same size as base_orbit) array.

    Masking is only applied within the call to function 'score'
    """
    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))
    # Calculate the scoring function at every possible window position.
    t0 = time.time_ns()/10**9
    pivot_scores, pivot_mask = score(base_orbit, window_orbit, threshold, **kwargs)
    print(f'scoring took {time.time_ns()/10**9 - t0} seconds')
    # The next step is to take the score at each pivot point and fill in the corresponding window positions with this
    # value, if it beats the threshold.
    orbit_scores = np.full_like(pivot_scores, np.nan)
    hull = kwargs.get('convex_hull', None) or window_orbit.shape
    for each_pivot in pivot_iterator(pivot_scores.shape, base_orbit.shape, window_orbit.shape, hull,
                                     base_orbit_periodicity, **kwargs):
        pivot_score = pivot_scores[each_pivot]
        # if pivot_score <= threshold:
        # The current window's positions is the pivot plus its dimensions.
        grid = tuple(p + g for g, p in zip(np.indices(window_orbit.shape), each_pivot))
        valid_coordinates = np.ones(window_orbit.shape, dtype=bool)
        # The scores only account for the points used to compute the metric; error in previous methods.
        for coordinates, base_size, window_size, hull_size, periodic in zip(grid, base_orbit.shape,
                                                                               window_orbit.shape, hull,
                                                                               base_orbit_periodicity):
            if periodic:
                # If periodic then all coordinates in this dimension are valid once folded back into the interior
                coordinates[coordinates >= (base_size+(hull_size-1))] -= base_size
                coordinates[coordinates < (hull_size-1)] += base_size
            else:
                # If aperiodic then any coordinates in the exterior are invalid.
                valid_coordinates = np.logical_and(valid_coordinates, ((coordinates < base_size+(window_size-1))
                                                                       & (coordinates >= (window_size-1))))
        # The positions in the masking array which are valid given the periodicity and scoring dimensions.
        interior_grid = tuple(coordinates[valid_coordinates] for coordinates in grid)
        filling_window = orbit_scores[interior_grid]
        # any positions which do not have scores take the current score by default. Else, check to see
        # if other scores are higher.
        if kwargs.get('overwrite', True):
            filling_window[np.isnan(filling_window)] = pivot_score
            # This allows usage of > without improper comparison to NaN, as they have all been filled.
            filling_window[(filling_window > pivot_score)] = pivot_score
        else:
            filling_window[np.isnan(filling_window)] = pivot_score
        # once the slices' values have been filled in, can put them back into the orbit score array
        orbit_scores[interior_grid] = filling_window

    # Still need to truncate the prefixed padding.
    orbit_scores = orbit_scores[tuple(slice(hull_size-1, -(hull_size-1)) for hull_size in hull)]
    orbit_mask = masking_function(orbit_scores, threshold)
    return orbit_scores, orbit_mask, pivot_scores, pivot_mask


def cover(base_orbit, thresholds, window_orbits, replacement=False, dtype=float, reorder_by_size=True, **kwargs):
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
    dtype : type
        The dtype of mask to return
    replacement : bool
        If False, then once a position in the score array is filled by a detection

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, [np.ndarray, np.ndarray])
        NumPy arrays whose indices along the first axis correspond to the window orbits' position in ```window_orbits```
        and whose other dimensions consist of scores or masking values. If return_pivot_arrays==True then also
        return the score and masking arrays for the pivots in addition to the orbit scores and pivots.
    Notes
    -----

    """
    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # Want to sort by "complexity"; originally was discretization, now area; maybe norm?
    window_sizes = [np.prod(w.dimensions()) for w in window_orbits]

    if reorder_by_size:
        size_order = np.argsort(window_sizes)[::-1]
    else:
        size_order = list(range(len(window_orbits)))

    # the array shape that can contain all window orbit arrays.
    convex_hull = tuple(max([window.discretization[i] for window in window_orbits])
                        for i in range(len(base_orbit.discretization)))
    # Masking array is used for multi-part computations.
    mask = kwargs.get('mask', None)
    # Returning numpy arrays instead of dicts now, allows usage of any and all.
    covering_masks = np.zeros([len(window_orbits), *base_orbit.shape])
    covering_scores = np.zeros([len(window_orbits), *base_orbit.shape])
    covering_pivot_scores = None
    covering_pivot_masks = None
    for index, threshold, window in zip(size_order, thresholds[size_order], window_orbits[size_order]):
        if kwargs.get('verbose', False) and index % max([1, len(thresholds)//10]) == 0:
            print('#', end='')
        # Note return_pivot_arrays must always be true here, but not necessarily for the overall function call.
        t0 = time.time_ns()/10**9
        shadowing_tuple = shadow(base_orbit, window, threshold, **{**kwargs, 'convex_hull': convex_hull, 'mask': mask})
        print(f'shadowing took {time.time_ns()/10**9 - t0} seconds')
        orbit_scores, orbit_mask, pivot_scores, pivot_mask = shadowing_tuple
        # replacement is whether or not to skip pivots which have already detected orbits. not replacement ==> mask
        if not replacement:
            if mask is not None:
                # all detections
                mask = np.logical_or(pivot_mask, mask)
            else:
                mask = pivot_mask.copy()
        # Return the masks for convenience, thresholding could be done outside function, though, it is used for
        # determining replacement here.
        covering_masks[index, ...] = orbit_mask.copy()
        covering_scores[index, ...] = orbit_scores.copy().astype(dtype)
        # For multiple runs providing a mask of the pivots is very useful, but because these can be very large arrays,
        # do not return these by default
        if kwargs.get('return_pivot_arrays', False):
            if covering_pivot_masks is None:
                covering_pivot_masks = np.zeros([len(window_orbits), *pivot_mask.shape])
            if covering_pivot_scores is None:
                covering_pivot_scores = np.zeros([len(window_orbits), *pivot_scores.shape])
            covering_pivot_masks[index, ...] = pivot_mask.copy()
            covering_pivot_scores[index, ...] = pivot_scores.copy().astype(dtype)

    if kwargs.get('return_pivot_arrays', False):
        return covering_scores, covering_masks, covering_pivot_scores, covering_pivot_masks
    else:
        return covering_scores, covering_masks

def fill(base_orbit, thresholds, window_orbits, **kwargs):
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
    dtype : type
        The dtype of mask to return
    replacement : bool
        If False, then once a position in the score array is filled by a detection

    Returns
    -------
    covering_masks : dict
        Dict whose keys are the index positions of the windows in the window_orbits provided, whose values
        are the ndarrays containing either the scores or boolean mask of the

    Notes
    -----
    This function is similar to cover, except that it checks the scores for each orbit at once, taking the "best"
    value as the winner. Now, it used to fill in the area after a successful detection but I changed this because
    it does not account for the fuzziness of shadowing.

    Thresholds need to be threshold densities, if following burak's u - u_p methodology.

    ***Note that the following can be completely skipped if scanning_region != 'interior' (i.e. use the default setting)
    Using heterogeneously sized orbits wit constraint scanning_region=='interior' will only be able to
    scan the pivots which are valid for the *largest* orbit. Therefore, this may miss valid detections along the
    boundaries. To avoid this issue, call 'fill' multiple times, decreasing the size of the subset of orbits and
    passing a mask each time. Cannot simply call the function multiple times, once for each sized orbit, because
    that will miss out on comparison on the "interior" pivots between all valid orbits.
    """

    # Whether or not previous runs have "cached" the persistence information; not really caching, mind you.
    # This doesn't do anything unless either mask_type=='family' or score_type=='persistence'. One or both must be true.
    strides = kwargs.get('strides', tuple([1]*len(base_orbit.shapes()[0])))
    # Easiest way I've found to account for periodicity is to pad with wrapped values
    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))
    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)

    # the shape that can contain all window orbits.
    hull = tuple(max([window.discretization[i] for window in window_orbits])
                 for i in range(len(base_orbit.discretization)))
    for w_dim, b_dim in zip(hull, base_orbit.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    scoring_function = kwargs.get('scoring_function', l2_difference)
    # score_array_shape = base_orbit.shape
    window_keys = kwargs.get('window_keys', range(1, len(window_orbits)+1))
    # The following looks like a repeat of the same exact computation but unfortunately axis cannot
    # be provided in the padding function. meaning that all dimensions must be provided for both types of padding.

    periodic_padding = tuple((pad-1, pad-1) if bc else (0, 0) for bc, pad in zip(base_orbit_periodicity,  hull))
    aperiodic_padding = tuple((pad-1, pad-1) if not bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    # Pad the base orbit state to use for computing scores, create the score array which will contain each pivot's score
    padded_state = np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'), aperiodic_padding)
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit), 'state': padded_state})
    pivot_scores = np.zeros(padded_state.shape, dtype=float)#[tuple(slice(None, -h+1) for h in hull)]

    # Number of discrete points will be the discretized spacetime.
    base_size = np.prod(base_orbit.discretization)
    # Allow for storage of floats, might want to plug in physical observable value.
    mask = kwargs.get('mask', None) or np.zeros(padded_base_orbit.shape, dtype=float)
    weights = dict(zip(window_keys, len(window_keys)*[0]))
    # To get around window based pivot iterator, just make the window the same as the hull.
    for pivot in pivot_iterator(pivot_scores.shape, base_orbit.shape, hull, hull,
                                base_orbit_periodicity, **kwargs):
        # See if the site is filled in the base_orbit array.
        filled = mask[tuple(p*s for p, s in zip(pivot, strides))]
        # If a site has already been filled, then skip it and move to the next scanning position.
        if filled != 0.:
            pass
        else:
            subdomain_slices = []
            # Need the correct slice of the base orbit with which to compute metric. Take the largest possible
            # slice based on the largest shape that contains all windows, then take subdomains of this slice accordingly
            for pivot_coord, span, stride in zip(pivot, hull, strides):
                subdomain_slices.append(slice(pivot_coord * stride, pivot_coord * stride + span))

            base_orbit_subdomain = padded_base_orbit.state[tuple(subdomain_slices)]
            pivot_scores = []
            # Once the correct field slice has been retrieved, score it against all of the windows.
            for window in window_orbits:
                # Need the base orbit slice and window to be the same size (typically), therefore, take slices
                # with respect to the smaller size in each dimension. If the same size, this slicing does nothing.
                subwindow_slices = tuple(slice(min([base_shp, window_shp]))
                                         for base_shp, window_shp in zip(base_orbit_subdomain.shape, window.shape))

                window_state_slice = window.state[subwindow_slices]
                base_state_slice = base_orbit_subdomain[subwindow_slices]
                pivot_scores.append(scoring_function(base_state_slice, window_state_slice, **kwargs))

            # Comparison is between residual densities where base state is not zero?

            # Need to reproduce the size of the window used about
            minimum_score_orbit_index = int(np.argmin(pivot_scores))
            if pivot_scores[minimum_score_orbit_index] <= thresholds[minimum_score_orbit_index]:
                grid = tuple(g + p * s for g, p, s in zip(np.indices(window_orbits[minimum_score_orbit_index].shape),
                                                          pivot, strides))
                # To account for wrapping, fold indices which extend beyond the base orbit's extent using periodicity.
                # If aperiodic then this has no effect, as the coordinates are dependent upon the *padded* base state
                # If one of the dimensions is not periodic, then need to ensure that all coordinates of the
                # window slice are within the boundaries of the base orbit.
                # if False in base_orbit_periodicity:
                window_hull = window_orbits[minimum_score_orbit_index].shape
                valid_coordinates = np.ones(window_hull, dtype=bool)
                wrap_valid_coordinates = np.ones(valid_coordinates.shape, dtype=bool)
                base_coordinates = []
                wrapped_periodic_coordinates = []
                periodic_coordinates = []
                # The scores only account for the points used to compute the metric; error in previous methods.
                for coordinates, base_extent, periodic, h in zip(grid, base_orbit.shape,
                                                                 base_orbit_periodicity, window_hull):
                    # The point is to construct the filling of the base orbit, but to do so we also need to keep
                    # track of the filling of the padded orbit.
                    coord = coordinates.copy()
                    if periodic:
                        if coordinates.min() < h:
                            wrap = coordinates + base_extent
                        elif coordinates.max() > base_extent + h:
                            wrap = coordinates - base_extent
                        else:
                            # if the window is completely on the interior then wrap is redundant, but still need all
                            # coordinates for later slicing.
                            wrap = coordinates
                        wrapped_periodic_coordinates.append(wrap)
                        periodic_coordinates.append(coordinates)
                        # Due to the way pivots and wrapping works, it can accidentally extend beyond even the
                        # padded region. Cannot include these points, of course.
                        wrap_valid_coordinates = np.logical_and(wrap_valid_coordinates, (wrap < (base_extent+2*h)))
                        wrap_valid_coordinates = np.logical_and(wrap_valid_coordinates, (wrap > 0))

                        # If periodic then all coordinates in this dimension are valid.
                        coord[coord >= (base_extent+h)] = coord[coord >= (base_extent+h)] - base_extent
                        coord[coord < h] = coord[coord < h] + base_extent
                    else:
                        if True in base_orbit_periodicity:
                            # only need to add to the
                            wrapped_periodic_coordinates.append(coordinates)
                            periodic_coordinates.append(coordinates)
                        # If out of bounds in any of the dimensions, need to exclude.
                        valid_coordinates = np.logical_and(valid_coordinates, (coord < base_extent+h) & (coord >= h))
                    base_coordinates.append(coord)

                grid = tuple(coordinates[valid_coordinates] for coordinates in base_coordinates)
                # Get the correct amount of space-time which is *about* to be filled; this is the proportion added to
                # weights. The "filled" qualifier at the beginning of the loop just checked whether the
                # *pivot* was filled, NOT the entirety of the window.
                filling_window = padded_base_orbit_mask[grid]
                # This is the number of unfilled points divided by total number of points in base discretization
                unfilled_spacetime_within_window = filling_window[filling_window == 0.].size / base_size
                # use the window_keys list and not the actual dict keys because repeat keys are allowed for families.
                weights[window_keys[minimum_score_orbit_index]] += unfilled_spacetime_within_window
                # Once we count what the compute the unfilled space-time, can then fill it in with the correct value.
                filling_window[filling_window == 0.] = window_keys[minimum_score_orbit_index]
                # Finally, need to fill in the actual base orbit mask the window came from.
                # This fills the correct amount of space-time with a key related to the "best" fitting window orbit
                padded_base_orbit_mask[grid] = filling_window
                # Subtract the orbit state from the padded base orbit used to compute the score.
                if kwargs.get('subtract_field', False):
                    padded_base_orbit.state[grid] = 0
                if True in base_orbit_periodicity:
                    wrapped_grid = tuple(coordinates[wrap_valid_coordinates] for coordinates
                                               in wrapped_periodic_coordinates)
                    padded_grid = tuple(c for c in periodic_coordinates)
                    # The mask is only affected by the original, internal points, but the padded field needs
                    # to set padding equal to zero if it is nonzero (periodic padding).
                    if kwargs.get('subtract_field', False):
                        padded_base_orbit.state[wrapped_grid] = 0
                        padded_base_orbit.state[padded_grid] = 0
    # Only want to return the filling of the original orbit, padding is simply a tool
    base_orbit_mask = padded_base_orbit_mask[tuple(slice(pad, -pad) for pad in hull)]
    filled_base_orbit = padded_base_orbit[tuple(slice(pad, -pad) for pad in hull)]
    return filled_base_orbit, base_orbit_mask, weights





    # if pivot_scores[minimum_score_orbit_index] <= thresholds[minimum_score_orbit_index]:
    #     grid = tuple(g + p * s for g, p, s in zip(np.indices(window_orbits[minimum_score_orbit_index].shape),
    #                                               pivot, strides))
    #     # To account for wrapping, fold indices which extend beyond the base orbit's extent using periodicity.
    #     # If aperiodic then this has no effect, as the coordinates are dependent upon the *padded* base state
    #     # If one of the dimensions is not periodic, then need to ensure that all coordinates of the
    #     # window slice are within the boundaries of the base orbit.
    #     # if False in base_orbit_periodicity:
    #     window_hull = window_orbits[minimum_score_orbit_index].shape
    #     valid_coordinates = np.ones(window_hull, dtype=bool)
    #     wrap_valid_coordinates = np.ones(valid_coordinates.shape, dtype=bool)
    #     base_coordinates = []
    #     wrapped_periodic_coordinates = []
    #     periodic_coordinates = []
    #     # The scores only account for the points used to compute the metric; error in previous methods.
    #     for coordinates, base_extent, periodic, h in zip(grid, base_orbit.shape,
    #                                                      base_orbit_periodicity, window_hull):
    #         # The point is to construct the filling of the base orbit, but to do so we also need to keep
    #         # track of the filling of the padded orbit.
    #         coord = coordinates.copy()
    #         if periodic:
    #             if coordinates.min() < h:
    #                 wrap = coordinates + base_extent
    #             elif coordinates.max() > base_extent + h:
    #                 wrap = coordinates - base_extent
    #             else:
    #                 # if the window is completely on the interior then wrap is redundant, but still need all
    #                 # coordinates for later slicing.
    #                 wrap = coordinates
    #             wrapped_periodic_coordinates.append(wrap)
    #             periodic_coordinates.append(coordinates)
    #             # Due to the way pivots and wrapping works, it can accidentally extend beyond even the
    #             # padded region. Cannot include these points, of course.
    #             wrap_valid_coordinates = np.logical_and(wrap_valid_coordinates, (wrap < (base_extent+2*h)))
    #             wrap_valid_coordinates = np.logical_and(wrap_valid_coordinates, (wrap > 0))
    #
    #             # If periodic then all coordinates in this dimension are valid.
    #             coord[coord >= (base_extent+h)] = coord[coord >= (base_extent+h)] - base_extent
    #             coord[coord < h] = coord[coord < h] + base_extent
    #         else:
    #             if True in base_orbit_periodicity:
    #                 # only need to add to the
    #                 wrapped_periodic_coordinates.append(coordinates)
    #                 periodic_coordinates.append(coordinates)
    #             # If out of bounds in any of the dimensions, need to exclude.
    #             valid_coordinates = np.logical_and(valid_coordinates, (coord < base_extent+h) & (coord >= h))
    #         base_coordinates.append(coord)
    #
