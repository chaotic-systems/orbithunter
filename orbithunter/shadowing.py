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


# def absolute_threshold(score_array, threshold):
#     # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.
#     mask = np.zeros(score_array.shape, dtype=bool)
#     non_nan_coordinates = np.where(~np.isnan(score_array))
#     under_threshold = score_array[non_nan_coordinates] <= threshold
#     mask_indices = tuple(mask_coord[under_threshold] for mask_coord in non_nan_coordinates)
#     mask[mask_indices] = True
#     return mask

def absolute_threshold(score_array, threshold):
    # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.
    mask = np.zeros(score_array.shape, dtype=bool)
    mask[score_array <= threshold] = True
    return mask

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

    Could likely reduce memory costs by using np.ndindex and not np.indices but it would likely be a speed trade-off
    Instead of returning a generator, yield from is used; in case of errors this function will be in the traceback
    """
    # A masking array can be passed; positions where True indicate pivots to be skipped.
    if isinstance(kwargs.get('mask', None), np.ndarray):
        mask = kwargs.get('mask', None)
    else:
        mask = np.zeros(padded_shape, dtype=bool)

    for coordinates, periodic, base_size, window_size, hull_size in zip(np.indices(padded_shape), periodicity,
                                                                        base_shape, window_shape, hull):
        # These statements mask all invalid pivot points. Those in the "exterior" whose defintion depends on
        # window_size, periodicity, and the specified amount of window overlap with the interior.
        pad_size = hull_size-1
        if periodic:
            mask = np.logical_or(mask, coordinates < (pad_size - window_size))
            mask = np.logical_or(mask, coordinates > base_size)
        else:
            min_overlap_proportion = kwargs.get('min_overlap', 1)
            mask = np.logical_or(mask, coordinates < (pad_size - (1-min_overlap_proportion)*window_size))
            mask = np.logical_or(mask, coordinates > (base_size + pad_size - min_overlap_proportion*window_size))

    # Want the valid pivots only. By definition the pivots are the positions of the array; could likely reduce
    # me
    truncated_pivots = np.array(tuple(c[~mask] for c in np.indices(padded_shape))).T
    # Order the pivots by their distance from the center of the set of all pivots.
    approximate_centroid = truncated_pivots.max(axis=0)//2
    ordered_pivots = truncated_pivots[np.argsort(np.sum(np.abs(approximate_centroid - truncated_pivots), axis=-1))]
    yield from (tuple(x) for x in ordered_pivots)


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
    masking_function = kwargs.get('masking_function', absolute_threshold)
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
    pivot_scores = np.zeros(padded_state.shape, dtype=float)
    t0 = time.time_ns()/10**9
    for i, each_pivot in enumerate(pivot_iterator(padded_state.shape, base_orbit.shape, window_orbit.shape, hull,
                                                  base_orbit_periodicity, **kwargs)):
        # If aperiodic then we do not want to include zero padding in the metric scoring; slice the subdomains
        # to remove portions in the zero_padding.
        slice_list = []
        for pivot, span, hull_size, periodic in zip(each_pivot, window_orbit.shape, hull, base_orbit_periodicity):
            if not periodic:
                # "Move" the pivot to the boundary if not already within.
                slice_list.append(slice(max([pivot, hull_size-1]), pivot + span))
            else:
                slice_list.append(slice(pivot, pivot + span))
        # Slice out the subdomains; this doesn't do anything to the window if completely within bounds.
        # Slice as numpy array and not orbit because it is much faster.
        base_orbit_subdomain = padded_base_orbit.state[tuple(slice_list)]
        # The manner the pivots are setup makes it so the following slices the overlap of base and window orbits.
        window_subdomain = window_orbit.state[tuple(slice(-shp, None) for shp in base_orbit_subdomain.shape)]
        # Compute the score for the pivot, store in the pivot score array.
        pivot_scores[each_pivot] = scoring_function(base_orbit_subdomain, window_subdomain, window_orbit, **kwargs)
    t1 = time.time_ns()/10**9
    print(f" scoring took {t1-t0} seconds, {(t1-t0)/i} per step, {i} steps")
    # Return the scores and the masking of the scores for later use (possibly skip these pivots for future calculations)
    return pivot_scores, masking_function(pivot_scores, threshold)


def shadow(base_orbit, window_orbit, return_type='all', **kwargs):
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

    # Calculate the scoring function at every possible window position.
    scoring_function = kwargs.get('scoring_function', l2_difference_mean_flow_correction)
    # Sometimes there are metrics (like persistence diagrams) which need only be computed once per window.
    if kwargs.get('window_caching_function', None) is not None:
        kwargs['window_cache'] = kwargs.get('window_caching_function', None)(window_orbit, **kwargs)

    for w_dim, b_dim in zip(window_orbit.shape, base_orbit.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))
    hull = kwargs.get('convex_hull', None) or window_orbit.shape
    verbose = kwargs.get('verbose', False)
    if kwargs.get('padded_orbit', None) is not None:
        padded_orbit = kwargs.get('padded_orbit', None)
    else:
        padded_orbit = pad_orbit_with_hull(base_orbit, hull, kwargs.get('periodicity', tuple(len(hull)*[False])))
    padded_shape = padded_orbit.shape
    pivot_scores = np.full_like(padded_orbit.state, np.inf)
    pivot_coordinates = np.indices(padded_shape)
    orbit_scores = None
    if isinstance(kwargs.get('mask', None), np.ndarray):
        mask = kwargs.get('mask', None)
    else:
        mask = np.zeros(padded_shape, dtype=bool)
    coordinate_mapping_function = kwargs.get('coordinate_map', None)
    min_overlap_proportion = kwargs.get('min_overlap', 1)

    import matplotlib.pyplot as plt
    for axis_coord, base_size, window_size, hull_size, periodic in zip(pivot_coordinates, base_orbit.shape,
                                                                       window_orbit.shape, hull, periodicity):
        pad_size = hull_size-1
        if periodic:
            mask = np.logical_or(mask, axis_coord < (pad_size - window_size))
            mask = np.logical_or(mask, axis_coord >= base_size)
        else:
            mask = np.logical_or(mask, axis_coord < (pad_size - (1-min_overlap_proportion)*(window_size-1)))
            mask = np.logical_or(mask, axis_coord >= (base_size + pad_size - min_overlap_proportion*(window_size-1)))

    truncated_pivots = []
    for axis_coord, base_size, hull_size, periodic in zip(pivot_coordinates, base_orbit.shape, hull, periodicity):
        truncated_pivots.append(axis_coord[~mask].reshape(-1, 1))

    truncated_pivots = np.concatenate(truncated_pivots, axis=-1)
    approximate_centroid = truncated_pivots.max(axis=0) // 2
    ordered_pivots = truncated_pivots[np.argsort(np.sum(np.abs(approximate_centroid - truncated_pivots), axis=-1))]
    for i, each_pivot in enumerate(ordered_pivots):
        if verbose:
            if i != 0 and i % max([1, len(ordered_pivots)//10]) == 0:
                print('-', end='')
        each_pivot = tuple(each_pivot)
        base_slices = []
        window_slices = []
        for pivot, span, base_size, hull_size, periodic in zip(each_pivot, window_orbit.shape,
                                                                              base_orbit.shape, hull, periodicity):

            pad_size = hull_size - 1
            # This allows generalization of rectangular to more nonlinear domains via functional mappings
            if coordinate_mapping_function is not None:
                base_slices.append(slice(pivot, pivot + span))
                window_slices.append(slice(0, span))
            else:
                # If no function provided, simple slicing is the method used.
                if not periodic:
                    base_start = max([pivot, pad_size])
                    base_end = min([pivot + span, base_size + pad_size])
                    window_start = base_start - pivot
                    window_end = base_end - base_start
                    base_slices.append(slice(base_start, base_end))
                    window_slices.append(slice(window_start, window_end))
                else:
                    base_slices.append(slice(pivot, pivot + span))
                    window_slices.append(slice(0, span))
        base_slices = tuple(base_slices)
        window_slices = tuple(window_slices)
        subdomain_coordinates = None
        if coordinate_mapping_function is not None:
            subdomain_coordinates = coordinate_mapping_function(pivot_coordinates[(slice(None), *base_slices)].copy(),
                                                                **kwargs)
            submask = np.zeros(subdomain_coordinates.shape[1:], dtype=bool)
            for sub_coords, base_size, window_size, hull_size, periodic in zip(subdomain_coordinates,
                                                                               base_orbit.shape, window_orbit.shape,
                                                                               hull, periodicity):
                pad_size = hull_size - 1
                if periodic:
                    sub_coords[sub_coords >= (base_size+pad_size)] -= base_size
                    sub_coords[sub_coords < pad_size] += base_size
                else:
                    lower_bound = (pad_size - (1-min_overlap_proportion)*(window_size-1))
                    upper_bound = (base_size + pad_size - min_overlap_proportion*(window_size-1))
                    submask = np.logical_or(submask, sub_coords < lower_bound)
                    submask = np.logical_or(submask, sub_coords >= upper_bound)
            subdomain_coordinates = tuple(tuple(c[~submask].ravel()) for c in subdomain_coordinates)
            base_subdomain = padded_orbit.state[subdomain_coordinates]
            window_subdomain = window_orbit.state[~submask]
        elif return_type.count('pivot') == 0.:
            subdomain_coordinates = tuple(tuple(x[base_slices].ravel()) for x in pivot_coordinates)
            base_subdomain = padded_orbit.state[base_slices]
            window_subdomain = window_orbit.state[window_slices]
            for sub_coords, base_size, window_size, hull_size, periodic in zip(subdomain_coordinates,
                                                                               base_orbit.shape, window_orbit.shape,
                                                                               hull, periodicity):
                pad_size = hull_size - 1
                if periodic:
                    sub_coords[sub_coords >= (base_size+pad_size)] -= base_size
                    sub_coords[sub_coords < pad_size] += base_size
        else:
            base_subdomain = padded_orbit.state[base_slices]
            window_subdomain = window_orbit.state[window_slices]
        pivot_scores[each_pivot] = scoring_function(base_subdomain, window_subdomain, **kwargs)
        if subdomain_coordinates is not None:
            if orbit_scores is None:
                orbit_scores = np.full_like(pivot_scores, np.inf)
            filling_window = orbit_scores[subdomain_coordinates]
            filling_window[(filling_window > pivot_scores[each_pivot])] = pivot_scores[each_pivot]
            orbit_scores[subdomain_coordinates] = filling_window

    if isinstance(orbit_scores, np.ndarray):
        orbit_scores = orbit_scores[tuple(slice(hull_size-1, - (hull_size-1)) for hull_size in hull)]
    return pivot_scores, orbit_scores


def pad_orbit_with_hull(base_orbit, hull, periodicity):
    # This looks redundant but it is not because two padding calls with different modes are required.
    periodic_padding = tuple((pad-1, pad-1) if bc else (0, 0) for bc, pad in zip(periodicity,  hull))
    aperiodic_padding = tuple((pad-1, pad-1) if not bc else (0, 0) for bc, pad in zip(periodicity, hull))
    # Pad the base orbit state to use for computing scores, create the score array which will contain each pivot's score
    padded_state = np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'), aperiodic_padding)
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit), 'state': padded_state})
    return padded_base_orbit


def cover(base_orbit, thresholds, window_orbits, replacement=False, reorder_by_size=True, **kwargs):
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
        iter_order = np.argsort(window_sizes)[::-1]
    else:
        iter_order = list(range(len(window_orbits)))

    # the array shape that can contain all window orbit arrays.
    convex_hull = tuple(max([window.discretization[i] for window in window_orbits])
                        for i in range(len(base_orbit.discretization)))
    # Masking array is used for very specific or multi-part computations.
    mask = kwargs.get('mask', None)
    masking_function = kwargs.get('masking_function', absolute_threshold)
    periodicity = kwargs.get('base_orbit_periodicity', tuple(len(convex_hull)*[False]))
    return_type = kwargs.get('return_type', 'all')

    padded_orbit = pad_orbit_with_hull(base_orbit, convex_hull, periodicity)
    # To avoid excessive memory usage by default, allow user to be very specific in what they want returned.
    # This is ugly but necessary I think in the cases of large memory usage.
    if return_type in ['all', 'orbit', 'orbit_masks']:
        orbit_masks = np.zeros([len(window_orbits), *base_orbit.shape], dtype=bool)
    else:
        orbit_masks = None

    if return_type in ['all', 'orbit', 'orbit_scores']:
        orbit_scores = np.zeros([len(window_orbits), *base_orbit.shape], dtype=float)
    else:
        orbit_scores = None

    if return_type in ['all', 'pivot', 'pivot_masks']:
        pivot_masks = np.zeros([len(window_orbits), *padded_orbit.shape], dtype=bool)
    else:
        pivot_masks = None

    if return_type in ['all', 'pivot', 'pivot_scores']:
        pivot_scores = np.zeros([len(window_orbits), *padded_orbit.shape], dtype=float)
    else:
        pivot_scores = None

    for index, threshold, window_orbit in zip(iter_order, thresholds[iter_order], window_orbits[iter_order]):
        if kwargs.get('verbose', False) and index % max([1, len(thresholds)//10]) == 0:
            print('#', end='')
        shadow_kwargs = {**kwargs, 'convex_hull': convex_hull, 'mask': mask, 'padded_orbit': padded_orbit}
        shadowing_tuple = shadow(base_orbit, window_orbit, **shadow_kwargs)
        window_pivot_scores, window_orbit_scores = shadowing_tuple
        # replacement is whether or not to skip pivots that have detected shadowing.
        if not replacement:
            if mask is not None:
                # all detections
                mask = np.logical_or(masking_function(window_pivot_scores, threshold), mask)
            else:
                mask = masking_function(window_pivot_scores, threshold).copy()

        if isinstance(orbit_masks, np.ndarray):
            orbit_masks[index, ...] = masking_function(window_orbit_scores, threshold)[...]
        if isinstance(orbit_scores, np.ndarray):
            orbit_scores[index, ...] = window_orbit_scores[...]
        if isinstance(pivot_masks, np.ndarray):
            pivot_masks[index, ...] = masking_function(window_pivot_scores, threshold)[...]
        if isinstance(pivot_scores, np.ndarray):
            pivot_scores[index, ...] = window_pivot_scores[...]

    return pivot_scores, pivot_masks, orbit_scores, orbit_masks


def fill(base_orbit, window_orbits, thresholds, **kwargs):
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
            slice_list = []
            # Need the correct slice of the base orbit with which to compute metric. Take the largest possible
            # slice based on the largest shape that contains all windows, then take subdomains of this slice accordingly
            for pivot_coord, span, stride in zip(pivot, hull, strides):
                slice_list.append(slice(pivot_coord * stride, pivot_coord * stride + span))

            base_orbit_subdomain = padded_base_orbit.state[tuple(slice_list)]
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
