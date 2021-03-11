import numpy as np

__all__ = ["shadow", "fill", "cover"]


def amplitude_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice.state ** 2 - window.state ** 2)


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


def masked_l2_difference_mean_flow_correction_density(
    base_slice, window, *args, **kwargs
):
    base_mask = base_slice.copy().astype(bool)
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm(
        (base_slice[base_mask] - base_slice[base_mask].mean()) - window[base_mask]
    )
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def l2_difference_mean_flow_correction(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    return np.linalg.norm((base_slice - base_slice.mean()) - window)


def l2_difference_mean_flow_correction_density(base_slice, window, *args, **kwargs):
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm((base_slice - base_slice.mean()) - window)
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


def pivot_iterator(
    pivot_array_shape, base_shape, window_shape, hull, periodicity, **kwargs
):
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
    min_overlap_proportion = kwargs.get("min_overlap", 1)
    if isinstance(kwargs.get("mask", None), np.ndarray):
        mask = kwargs.get("mask", None)
    else:
        mask = np.zeros(pivot_array_shape, dtype=bool)
    pivot_coordinates = np.indices(pivot_array_shape)
    for axis_coord, base_size, window_size, hull_size, periodic in zip(
        pivot_coordinates, base_shape, window_shape, hull, periodicity
    ):
        pad_size = hull_size - 1
        if periodic:
            mask = np.logical_or(mask, axis_coord < (pad_size - window_size))
            mask = np.logical_or(mask, axis_coord >= base_size)
        else:
            mask = np.logical_or(
                mask,
                axis_coord
                < (pad_size - (1 - min_overlap_proportion) * (window_size - 1)),
            )
            mask = np.logical_or(
                mask,
                axis_coord
                >= (base_size + pad_size - min_overlap_proportion * (window_size - 1)),
            )

    truncated_pivots = []
    for axis_coord, base_size, hull_size, periodic in zip(
        pivot_coordinates, base_shape, hull, periodicity
    ):
        truncated_pivots.append(axis_coord[~mask].reshape(-1, 1))
    # clean up and remove the potentially very large array.
    del pivot_coordinates
    # order the pivots so they follow an inside-out approach.
    truncated_pivots = np.concatenate(truncated_pivots, axis=-1)
    approximate_centroid = truncated_pivots.max(axis=0) // 2
    ordered_pivots = truncated_pivots[
        np.argsort(np.sum(np.abs(approximate_centroid - truncated_pivots), axis=-1))
    ]
    return ordered_pivots


def subdomain_slices(pivot, base_orbit, window_orbit, hull, periodicity, **kwargs):
    base_slices = []
    window_slices = []
    for pivot, base_size, span, hull_size, periodic in zip(
        pivot, base_orbit.shape, window_orbit.shape, hull, periodicity
    ):
        pad_size = hull_size - 1
        # This allows generalization of rectangular to more nonlinear domains via functional mappings
        if kwargs.get("coordinate_map", None) is None and not periodic:
            base_start = max([pivot, pad_size])
            base_end = min([pivot + span, base_size + pad_size])
            base_slices.append(slice(base_start, base_end))
            window_slices.append(slice(base_start - pivot, base_end - base_start))
        else:
            base_slices.append(slice(pivot, pivot + span))
            window_slices.append(slice(0, span))

    return tuple(base_slices), tuple(window_slices)


def subdomains(
    pivot,
    orbit_with_hull,
    base_orbit,
    window_orbit,
    hull_grid,
    hull,
    periodicity,
    **kwargs
):
    coordinate_mapping_function = kwargs.get("coordinate_map", None)
    base_slices, window_slices = subdomain_slices(
        pivot, base_orbit, window_orbit, hull, periodicity, **kwargs
    )
    broadcasting_shaped_pivot = np.array(pivot).reshape(
        len(pivot), *tuple(len(pivot) * [1])
    )
    window_grid = hull_grid[(slice(None), *window_slices)]
    if coordinate_mapping_function is not None:
        subdomain_grid = window_grid + broadcasting_shaped_pivot
        # Add the pivot to translate the pivot; for proper broadcasting this requires reshaping
        mapped_subdomain_grid = coordinate_mapping_function(subdomain_grid, **kwargs)
        mapped_window_grid = coordinate_mapping_function(window_grid, **kwargs)
        submask = np.zeros(mapped_subdomain_grid.shape[1:], dtype=bool)
        for sub_coords, base_size, window_size, hull_size, periodic in zip(
            mapped_subdomain_grid,
            base_orbit.shape,
            window_orbit.shape,
            hull,
            periodicity,
        ):
            pad_size = hull_size - 1
            if periodic:
                sub_coords[sub_coords >= (base_size + pad_size)] -= base_size
                sub_coords[sub_coords < pad_size] += base_size
            else:
                # This type of masking was used for pivots but now because we are mapping coordinates we have
                # to recheck. The bounds are different for pivots and actual coordinates.
                submask = np.logical_or(submask, sub_coords < pad_size)
                submask = np.logical_or(submask, sub_coords >= base_size + pad_size)
        subdomain_coordinates = tuple(
            c[~submask].ravel() for c in mapped_subdomain_grid
        )
        window_coordinates = tuple(c[~submask].ravel() for c in mapped_window_grid)
        base_subdomain = orbit_with_hull.state[subdomain_coordinates]
        window_subdomain = window_orbit.state[window_coordinates]
    else:
        base_subdomain = orbit_with_hull.state[base_slices]
        window_subdomain = window_orbit.state[window_slices]
        # Get the subdomain's coordinates with respect to the padded array, if returning more than the pivot scores
        # sliced_window_grid = window_grid[(slice(None), *window_slices)]
        if kwargs.get("return_type", "all").count("pivot") == 0.0:
            subdomain_grid = (window_grid + broadcasting_shaped_pivot)[
                (slice(None), *window_slices)
            ]
            if True in periodicity:
                for sub_coords, base_size, window_size, hull_size, periodic in zip(
                    subdomain_grid,
                    base_orbit.shape,
                    window_orbit.shape,
                    hull,
                    periodicity,
                ):
                    pad_size = hull_size - 1
                    if periodic:
                        sub_coords[sub_coords >= (base_size + pad_size)] -= base_size
                        sub_coords[sub_coords < pad_size] += base_size
            # unravel and cast as tuples for basic and not advanced indexing
            subdomain_coordinates = tuple(c.ravel() for c in subdomain_grid)
        else:
            subdomain_coordinates = None
    return base_subdomain, window_subdomain, subdomain_coordinates


def shadow(base_orbit, window_orbit, **kwargs):
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
    # retrieve the function for scoring, using a default function based on the l2 difference.
    scoring_function = kwargs.get(
        "scoring_function", l2_difference_mean_flow_correction
    )
    # Sometimes there are metrics, like bottleneck distance between persistence diagrams,
    # which need only be computed once per window. To avoid redundant calculations, cache this result.
    if kwargs.get("window_caching_function", None) is not None:
        kwargs["window_cache"] = kwargs.get("window_caching_function", None)(
            window_orbit, **kwargs
        )

    for w_dim, b_dim in zip(window_orbit.shape, base_orbit.shape):
        assert (
            w_dim < b_dim
        ), "Shadowing window discretization is larger than the base orbit. resize first. "

    # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
    periodicity = kwargs.get(
        "base_orbit_periodicity", tuple(len(base_orbit.dimensions()) * [False])
    )
    hull = kwargs.get("convex_hull", None) or window_orbit.shape
    verbose = kwargs.get("verbose", False)
    return_type = kwargs.get("return_type", "all")
    if kwargs.get("padded_orbit", None) is not None:
        padded_orbit = kwargs.get("padded_orbit", None)
    else:
        padded_orbit = pad_orbit_with_hull(
            base_orbit, hull, kwargs.get("periodicity", tuple(len(hull) * [False]))
        )

    pivot_scores = np.full_like(padded_orbit.state, np.inf)
    orbit_scores = None
    window_grid = np.indices(window_orbit.shape)
    ordered_pivots = pivot_iterator(
        pivot_scores.shape,
        base_orbit.shape,
        window_orbit.shape,
        hull,
        periodicity,
        **kwargs
    )
    for i, each_pivot in enumerate(ordered_pivots):
        each_pivot = tuple(each_pivot)
        if verbose:
            if i != 0 and i % max([1, len(ordered_pivots) // 10]) == 0:
                print("-", end="")

        subdomain_tuple = subdomains(
            each_pivot,
            padded_orbit,
            base_orbit,
            window_orbit,
            window_grid,
            hull,
            periodicity,
            **kwargs
        )
        base_subdomain, window_subdomain, subdomain_coordinates = subdomain_tuple
        pivot_scores[each_pivot] = scoring_function(
            base_subdomain, window_subdomain, **kwargs
        )
        if return_type.count("pivot") == 0.0 and subdomain_coordinates is not None:
            if orbit_scores is None:
                orbit_scores = np.full_like(pivot_scores, np.inf)
            filling_window = orbit_scores[subdomain_coordinates]
            filling_window[(filling_window > pivot_scores[each_pivot])] = pivot_scores[
                each_pivot
            ]
            orbit_scores[subdomain_coordinates] = filling_window

    if isinstance(orbit_scores, np.ndarray):
        orbit_scores = orbit_scores[
            tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in hull)
        ]
    return pivot_scores, orbit_scores


def pad_orbit_with_hull(base_orbit, hull, periodicity):
    # This looks redundant but it is not because two padding calls with different modes are required.
    periodic_padding = tuple(
        (pad - 1, pad - 1) if bc else (0, 0) for bc, pad in zip(periodicity, hull)
    )
    aperiodic_padding = tuple(
        (pad - 1, pad - 1) if not bc else (0, 0) for bc, pad in zip(periodicity, hull)
    )
    # Pad the base orbit state to use for computing scores, create the score array which will contain each pivot's score
    padded_state = np.pad(
        np.pad(base_orbit.state, periodic_padding, mode="wrap"), aperiodic_padding
    )
    padded_orbit = base_orbit.__class__(**{**vars(base_orbit), "state": padded_state})
    return padded_orbit


def cover(
    base_orbit,
    window_orbits,
    thresholds,
    replacement=False,
    reorder_by_size=True,
    **kwargs
):
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
    if reorder_by_size:
        window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
        iter_order = np.argsort(window_sizes)[::-1]
    else:
        iter_order = list(range(len(window_orbits)))

    # the array shape that can contain all window orbit arrays.
    hull = tuple(
        max([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )
    # Masking array is used for very specific or multi-part computations.
    masking_function = kwargs.get("masking_function", absolute_threshold)
    periodicity = kwargs.get("base_orbit_periodicity", tuple(len(hull) * [False]))
    return_type = kwargs.get("return_type", "all")
    padded_orbit = pad_orbit_with_hull(base_orbit, hull, periodicity)
    # The largest possible range of pivots is decided upon by the initial mask.
    mask = kwargs.get("mask", None)
    if mask is None:
        mask = np.zeros(padded_orbit.shape, dtype=bool)

    maximal_set_of_pivots = pivot_iterator(
        padded_orbit.shape, base_orbit.shape, hull, hull, periodicity, **kwargs
    )
    masked_pivot_dimensions = tuple(
        axis.max() - axis.min() + 1 for axis in maximal_set_of_pivots.T
    )
    maximal_pivot_slices = tuple(
        slice(axis.min(), axis.max() + 1) for axis in maximal_set_of_pivots.T
    )
    del maximal_set_of_pivots
    # To avoid excessive memory usage by default, allow user to be very specific in what they want returned.
    # This is ugly but necessary I think in the cases of large memory usage.
    orbit_masks, orbit_scores, pivot_masks, pivot_scores = None, None, None, None
    for index, threshold, window_orbit in zip(
        iter_order, thresholds[iter_order], window_orbits[iter_order]
    ):
        if (
            kwargs.get("verbose", False)
            and index % max([1, len(thresholds) // 10]) == 0
        ):
            print("#", end="")
        shadow_kwargs = {
            **kwargs,
            "convex_hull": hull,
            "mask": mask,
            "padded_orbit": padded_orbit,
        }
        shadowing_tuple = shadow(base_orbit, window_orbit, **shadow_kwargs)
        window_pivot_scores, window_orbit_scores = shadowing_tuple
        # replacement is whether or not to skip pivots that have detected shadowing.
        if not replacement:
            if mask is not None:
                mask = np.logical_or(
                    masking_function(window_pivot_scores, threshold), mask
                )
            else:
                mask = masking_function(window_pivot_scores, threshold).copy()

        if return_type in ["all", "orbit", "orbit_masks", "masks"]:
            if orbit_masks is None:
                orbit_masks = np.zeros(
                    [len(window_orbits), *base_orbit.shape], dtype=bool
                )
            orbit_masks[index, ...] = masking_function(window_orbit_scores, threshold)[
                ...
            ]

        if return_type in ["all", "orbit", "orbit_scores", "scores"]:
            if orbit_scores is None:
                orbit_scores = np.zeros(
                    [len(window_orbits), *base_orbit.shape], dtype=float
                )
            orbit_scores[index, ...] = window_orbit_scores[...]

        if return_type in ["all", "pivot", "pivot_masks", "masks"]:
            if pivot_masks is None:
                pivot_masks = np.zeros(
                    [len(window_orbits), *masked_pivot_dimensions], dtype=bool
                )
            pivot_masks[index, ...] = masking_function(window_pivot_scores, threshold)[
                maximal_pivot_slices
            ]

        if return_type in ["all", "pivot", "pivot_scores", "scores"]:
            if pivot_scores is None:
                pivot_scores = np.zeros(
                    [len(window_orbits), *masked_pivot_dimensions], dtype=float
                )
            pivot_scores[index, ...] = window_pivot_scores[maximal_pivot_slices]

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

    # retrieve the function for scoring, using a default function based on the l2 difference.
    scoring_function = kwargs.get(
        "scoring_function", l2_difference_mean_flow_correction
    )
    # Sometimes there are metrics, like bottleneck distance between persistence diagrams,
    # which need only be computed once per window. To avoid redundant calculations, cache this result.
    if kwargs.get("window_caching_function", None) is not None:
        kwargs["window_cache"] = kwargs.get("window_caching_function", None)(
            window_orbits, **kwargs
        )

    # the array shape that can contain all window orbit arrays.
    hull = tuple(
        max([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )
    # Masking array is used for very specific or multi-part computations.
    periodicity = kwargs.get("base_orbit_periodicity", tuple(len(hull) * [False]))

    if kwargs.get("padded_orbit", None) is not None:
        padded_orbit = kwargs.get("padded_orbit", None)
    else:
        padded_orbit = pad_orbit_with_hull(
            base_orbit, hull, kwargs.get("periodicity", tuple(len(hull) * [False]))
        )

    window_keys = kwargs.get("window_keys", range(1, len(window_orbits) + 1))
    # The following looks like a repeat of the same exact computation but unfortunately axis cannot
    # be provided in the padding function. meaning that all dimensions must be provided for both types of padding.

    # Number of discrete points will be the discretized spacetime.
    pivot_scores = np.full_like(padded_orbit.state, np.inf)
    orbit_scores = np.full_like(padded_orbit.state, np.inf)
    orbit_key_scores = np.full_like(padded_orbit.state, np.inf)
    pivot_key_scores = np.full_like(padded_orbit.state, np.inf)
    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # Want to sort by "complexity"; originally was discretization, now area; maybe norm?
    window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
    smallest_window_shape = window_orbits[np.argmin(window_sizes)].shape
    ordered_pivots = pivot_iterator(
        pivot_scores.shape,
        base_orbit.shape,
        smallest_window_shape,
        hull,
        periodicity,
        **kwargs
    )
    maximal_set_of_pivots = pivot_iterator(
        padded_orbit.shape, base_orbit.shape, hull, hull, periodicity, **kwargs
    )
    maximal_pivot_slices = tuple(
        slice(axis.min(), axis.max() + 1) for axis in maximal_set_of_pivots.T
    )
    hull_grid = np.indices(hull)
    for i, each_pivot in enumerate(ordered_pivots):
        each_pivot = tuple(each_pivot)
        # See if the site is filled in the base_orbit array.
        # If a site has already been filled, then skip it and move to the next scanning position.
        if kwargs.get("subtract_field", False) and orbit_scores[each_pivot] != np.inf:
            pass
        else:
            # Need the correct slice of the base orbit with which to compute metric. Take the largest possible
            # slice based on the largest shape that contains all windows, then take subdomains of this slice accordingly
            window_scores = []
            window_slices = []
            # Once the correct field slice has been retrieved, score it against all of the windows.
            for orbit in window_orbits:
                slices = subdomain_slices(
                    each_pivot, base_orbit, orbit, hull, periodicity, **kwargs
                )
                # Need the base orbit slice and window to be the same size (typically), therefore, take slices
                # with respect to the smaller size in each dimension. If the same size, this slicing does nothing.
                subdomain_tuple = subdomains(
                    each_pivot,
                    padded_orbit,
                    base_orbit,
                    orbit,
                    hull_grid,
                    hull,
                    periodicity,
                    **kwargs
                )
                # subdomain coordinates are the coordinates in the score array that account for periodic boundary conditions
                (
                    base_subdomain,
                    window_subdomain,
                    subdomain_coordinates,
                ) = subdomain_tuple
                window_scores.append(
                    scoring_function(base_subdomain, window_subdomain, **kwargs)
                )
                window_slices.append(slices)
            minimum_score_orbit_index = int(np.argmin(window_scores))
            pivot_scores[each_pivot] = window_scores[minimum_score_orbit_index]
            pivot_key_scores[each_pivot] = window_keys[minimum_score_orbit_index]
            if (
                window_scores[minimum_score_orbit_index]
                <= thresholds[minimum_score_orbit_index]
            ):
                for scores, score_holders in zip(
                    [window_scores, window_keys], [orbit_scores, orbit_key_scores]
                ):
                    filling_window = score_holders[subdomain_coordinates]
                    # unfilled_spacetime_within_window = filling_window[filling_window == np.inf].size / base_size
                    # weights[window_keys[minimum_score_orbit_index]] += unfilled_spacetime_within_window
                    filling_window[filling_window == np.inf] = scores[
                        minimum_score_orbit_index
                    ]
                    score_holders[subdomain_coordinates] = filling_window
                    # Subtract the orbit state from the padded base orbit used to compute the score.
                    if kwargs.get("subtract_field", False):
                        # algorithms allowed for subtraction of the underlying base field.
                        padded_orbit.state[subdomain_coordinates] = 0
                        padded_orbit.state[window_slices[minimum_score_orbit_index]] = 0

    # The orbit scores in the shape of the base orbit; periodic boundaries already accounted for by this point.
    orbit_scores = orbit_scores[
        tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in hull)
    ]
    return (
        pivot_scores[maximal_pivot_slices],
        pivot_key_scores[maximal_pivot_slices],
        orbit_scores,
        orbit_key_scores,
    )

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
