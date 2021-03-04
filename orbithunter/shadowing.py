import numpy as np

__all__ = ['score', 'shadow', 'fill', 'cover']


def amplitude_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice**2 - window**2)


def l2_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice - window)


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
    return np.linalg.norm(base_slice - window)


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
    mask = np.zeros(score_array.shape, dtype=bool)
    non_nan_coordinates = np.where(~np.isnan(score_array))
    under_threshold = score_array[non_nan_coordinates] <= threshold
    mask_indices = tuple(mask_coord[under_threshold] for mask_coord in non_nan_coordinates)
    mask[mask_indices] = True
    return mask


def scanning_dimensions(base_shape, window_shape, strides, base_orbit_periodicity):
    """ Helper function that is useful for caching and scanning .

    Parameters
    ----------
    base_shape
    strides
    window_shape
    base_orbit_periodicity

    Returns
    -------

    """
    score_array_shape = []
    pad_shape = []
    dimension_iterator = zip(base_shape, strides, window_shape, base_orbit_periodicity)
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
    return tuple(score_array_shape), tuple(pad_shape)


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
    for pivot_position in np.ndindex(tuple(p - (h-1) for p, h in zip(padded_shape, hull))):
        # a hack for "slicing" the generator, determine if in the subset of exterior pivots
        exterior_checker = tuple(p < (hl-1) for p, shp, hl in zip(pivot_position, base_shape, hull))
        if True in exterior_checker:
            # Want to work from inside to outside, but np.ndindex starts at 0's in each position. Therefore, need
            # to treat this as the distance away from the border of the interior region to work in the correct direction
            yield tuple((hll-2) - piv if piv < (hll-1) else piv for piv, hll in zip(pivot_position, hull))
        else:
            # If the pivot is not in the exterior, then skip it.
            continue


def pivot_iterator(padded_shape, base_shape, hull, periodicity, **kwargs):
    """

    Parameters
    ----------
    padded_shape : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_shape : tuple of int
        Shape of the numpy array corresponding to the base orbit
    hull : tuple of int
        Shape of the numpy array corresponding to the convex hull of a set of windows
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
    # If pivots provided, use those.
    pivots = kwargs.get('pivots', None) or inside_to_outside_iterator(padded_shape, base_shape, hull)

    # Masks can be passed in; the utility of this is coverings which have multiple steps
    mask = kwargs.get('mask', None)
    # Not allowing partial matches along the boundary is the same as keeping the pivots within the interior.
    if kwargs.get('scanning_region', 'exterior') == 'interior':
        if mask is None:
            mask = np.zeros(padded_shape, dtype=bool)
        # Need to trim the pivots in the padding if aperiodic.
        for coordinates, periodic, base_size, hull_size in zip(np.indices(padded_shape), periodicity, base_shape, hull):
            if not periodic:
                # Pivots in the exterior will be masked, by definition those are those in the padding
                mask = np.logical_or(np.logical_or(mask, coordinates < hull_size-1),
                                                   coordinates >= base_size + hull_size-1)

    if isinstance(mask, np.ndarray):
        mask = mask.astype(bool)
        # If mask == True then do NOT calculate the score at that position. without need to repeat calculations.
        # This intuition follows from np.ma.masked_array
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

    for w_dim, b_dim in zip(window.shape, base.shape):
        assert w_dim < b_dim, 'Shadowing window discretization is larger than the base orbit. resize first. '

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))
    hull = (int(kwargs.get('min_proportion', 0.5) * size) for size in window.shape)
    hull = window.shape
    periodic_padding = tuple((pad-1, pad-1) if bc else (0, 0) for bc, pad in zip(base_orbit_periodicity,  hull))
    aperiodic_padding = tuple((pad-1, pad-1) if not bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))

    padded_state = np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'), aperiodic_padding)
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit), 'state': padded_state})
    pivot_scores = np.zeros(padded_state.shape, dtype=float)[tuple(slice(None, -h+1) for h in hull)]

    for each_pivot in pivot_iterator(padded_base_orbit.shape, base_orbit.shape,hull,
                                     base_orbit_periodicity, **kwargs):
        # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
        # the window's dimensions.
        base_orbit_slices = []
        for pivot, span, periodic in zip(each_pivot, hull, base_orbit_periodicity):
            if not periodic:
                # If aperiodic then we only want to score the points of the window which are on the interior.
                base_orbit_slices.append(slice(max([pivot, span-1]), pivot + span))
            else:
                base_orbit_slices.append(slice(pivot, pivot + span))
        base_orbit_subdomain = padded_base_orbit.state[tuple(base_orbit_slices)]
        window_subdomain = window[tuple(slice(-shp, None) for shp in base_orbit_subdomain.shape)]
        pivot_scores[each_pivot] = scoring_function(base_orbit_subdomain, window_subdomain, **kwargs)

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
    pivot_scores, pivot_mask = score(base_orbit, window_orbit, threshold, **kwargs)
    # The next step is to take the score at each pivot point and fill in the corresponding window positions with this
    # value, if it beats the threshold.
    orbit_scores = np.full_like(pivot_scores, np.nan)
    for each_pivot in pivot_iterator(pivot_scores.shape, base_orbit.shape, window_orbit.shape,
                                     base_orbit_periodicity, **kwargs):
        pivot_score = pivot_scores[each_pivot]
        if pivot_score <= threshold:
            # The current window's positions is the pivot plus its dimensions.
            grid = tuple(p + g for g, p in zip(np.indices(window_orbit.shape), each_pivot))
            valid_coordinates = np.ones(window_orbit.shape, dtype=bool)
            # The scores only account for the points used to compute the metric; error in previous methods.
            for coordinates, base_extent, periodic, window_size in zip(grid, base_orbit.shape,
                                                                       base_orbit_periodicity, window_orbit.shape):
                if periodic:
                    # If periodic then all coordinates in this dimension are valid, once folded back into the interior
                    coordinates[coordinates >= (base_extent+window_size)] -= base_extent
                    coordinates[coordinates < window_size] += base_extent
                else:
                    # If aperiodic then any coordinates in the exterior are invalid.
                    valid_coordinates = np.logical_and(valid_coordinates, ((coordinates < base_extent+(window_size-1))
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
    orbit_scores = orbit_scores[tuple(slice(h-1, None) for h in window_orbit.shape)]
    orbit_mask = masking_function(orbit_scores, threshold)
    if kwargs.get('return_pivot_arrays', False):
        return orbit_scores, orbit_mask, pivot_scores, pivot_mask
    else:
        return orbit_scores, orbit_mask



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
    covering_masks : dict
        Dict whose keys are the indices corresponding to the position in the window_orbits list
        and whose values are the ndarrays containing the scores, dtype decided upon the respective keyword argument.

    Notes
    -----
    If window orbits are not identical in size, then replacement must be false.
    """
    # Whether or not previous runs have "cached" the persistence information; not really caching, mind you.
    # This doesn't do anything unless either mask_type=='family' or score_type=='persistence'. One or both must be true.

    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # Want to sort by "complexity"; originally was discretization, now area; maybe norm?
    window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
    # if not replacement:
    #     assrt_str = 'unequal window discretizations require replacement=True.'
    #     assert len(np.unique([w.size for w in window_orbits])) == 1, assrt_str

    if reorder_by_size:
        size_order = np.argsort(window_sizes)[::-1]
    else:
        size_order = list(range(len(window_orbits)))
    mask = None
    # Was storing in dict but it makes more sense to simply put them in an array.
    covering_masks = np.zeros([len(window_orbits), *base_orbit.shape])
    covering_scores = np.zeros([len(window_orbits), *base_orbit.shape])
    for index, threshold, window in zip(size_order, thresholds[size_order], window_orbits[size_order]):
        if kwargs.get('verbose', False) and index % max([1, len(thresholds)//10]) == 0:
            print('#', end='')
        # Just in case of union method
        shadowing_tuple = shadow(base_orbit, window, threshold, **{**kwargs, 'return_pivot_arrays': True, 'mask': mask})
        # The call of the 'score' function call within 'shadow' only iterates over unmasked pivots. Therefore, the
        # returned pivot mask will have empty intersection with 'mask'.
        orbit_scores, orbit_mask, pivot_scores, pivot_mask = shadowing_tuple
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(orbit_scores.astype(float))
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(orbit_mask.astype(float))
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(pivot_scores.astype(float))
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(pivot_mask.astype(float))
        plt.colorbar()
        plt.show()

        # # Want to mask sure that
        if not replacement:
            if mask is not None:
                # all detections
                mask = np.logical_or(pivot_mask, mask)
            else:
                mask = pivot_mask.copy()
        plt.figure(figsize=(10, 10))
        plt.imshow(mask.astype(float))
        plt.colorbar()
        plt.show()
        # If there were no detections then do not add anything to the cover?
        covering_masks[index] = orbit_mask.copy()
        covering_scores[index] = orbit_scores.copy().astype(dtype)

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
    Overlaps currently cause overwrites; this means that the

    Thresholds need to be threshold densities, if following burak's u - u_p methodology.
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
    periodic_padding = tuple((pad, pad) if bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    aperiodic_padding = tuple((pad, pad) if not bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    # If padded then
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit),
                                                'state': np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'),
                                                                aperiodic_padding)})

    # Number of discrete points will be the discretized spacetime.
    base_size = np.prod(base_orbit.discretization)
    # Allow for storage of floats, might want to plug in physical observable value.
    padded_base_orbit_mask = kwargs.get('initial_mask', None) or np.zeros(padded_base_orbit.shape, dtype=float)
    weights = dict(zip(window_keys, len(window_keys)*[0]))
    # for n_iter in range(kwargs.get('n_iter', 1)):
    # This is annoying but to fill in the most possible area, it is best to work from inside to outside,
    # i.e. pivots with
    for pivot in inside_to_outside_iterator(padded_base_orbit.shape, base_orbit.shape, hull,
                                            verbose=kwargs.get('verbose', False)):
        # for pivot in np.ndindex(padded_base_orbit.shape):
        # See if the site is filled in the base_orbit array.
        filled = padded_base_orbit_mask[tuple(p*s for p, s in zip(pivot, strides))]
        # If a site has already been filled, then skip it and move to the next scanning position.
        if filled != 0.:
            pass
        else:
            base_orbit_slices = []
            # Need the correct slice of the base orbit with which to compute metric. Take the largest possible
            # slice based on the largest shape that contains all windows, then take subdomains of this slice accordingly
            for pivot_coord, span, stride in zip(pivot, hull, strides):
                base_orbit_slices.append(slice(pivot_coord * stride, pivot_coord * stride + span))

            base_orbit_subdomain = padded_base_orbit.state[tuple(base_orbit_slices)]
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
