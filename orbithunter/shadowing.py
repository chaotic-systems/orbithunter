import numpy as np
import warnings

__all__ = ["shadow", "fill", "cover", "scoring_functions", "process_scores"]


def scoring_functions(method):
    """
    Return a callable to act as shadowing scoring metric.

    Parameters
    ----------
    method : str
        The name of the callable to return.


    Returns
    -------
    callable :
        function which takes arguments `(base_slice, window, *args, **kwargs)` and returns a (scalar) score.
        comparing the base slice and window. `base_slice` and `window` are arrays.

    """

    if method == "amplitude":
        func_ = amplitude_difference
    elif method == "l2":
        func_ = l2_difference
    elif method == "l2_density":
        func_ = l2_difference_density
    elif method == "l2_mfc":
        func_ = l2_difference_mean_flow_correction
    elif method == "l2_density_mfc":
        func_ = l2_difference_mean_flow_correction_density
    elif method == "masked_l2_density":
        func_ = masked_l2_difference_density
    elif method == "masked_l2_density_mfc":
        func_ = masked_l2_difference_mean_flow_correction_density
    else:
        raise ValueError(
            f"name {method} of scoring function not in methods provided by orbithunter; define"
            f" callable externally if still desired to be passed to shadowing functions."
        )

    return func_


def amplitude_difference(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    The 'amplitude difference' metric between `base_slice` and `window`

    """
    return np.linalg.norm(base_slice ** 2 - window ** 2)


def l2_difference(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    The 'amplitude difference' metric between `base_slice` and `window`

    """
    # account for local mean flow by normalization; windows have zero mean flow by definition
    return np.linalg.norm(base_slice - window)


def l2_difference_density(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    The 'amplitude difference' metric between `base_slice` and `window`

    """
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm(base_slice - window)
    norm_density = norm / base_slice.size
    return norm_density


def masked_l2_difference_density(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    norm_density : float
        The density of $L_2$ difference score

    """
    base_mask = base_slice.copy().astype(bool)
    norm = np.linalg.norm(base_slice[base_mask] - window[base_mask])
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def masked_l2_difference_mean_flow_correction_density(
    base_slice, window, *args, **kwargs
):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    norm_density : float
        The density of $L_2$ difference score

    """
    base_mask = base_slice.copy().astype(bool)
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm(
        (base_slice[base_mask] - base_slice[base_mask].mean()) - window[base_mask]
    )
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


def l2_difference_mean_flow_correction(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    float :
        $L_2$ difference between normalized base slice and window.

    """
    # account for local mean flow by normalization; windows have zero mean flow by definition
    return np.linalg.norm((base_slice - base_slice.mean()) - window)


def l2_difference_mean_flow_correction_density(base_slice, window, *args, **kwargs):
    """
    Experimental shadowing metric

    Parameters
    ----------
    base_slice : Orbit
        Orbit whose state is the same shape as `window`

    window : Orbit
        Orbit whose state is the same shape as `window`

    Returns
    -------
    norm_density : float
        The density of $L_2$ difference score between normalized base slice and window.

    """
    # account for local mean flow by normalization; windows have zero mean flow by definition
    norm = np.linalg.norm((base_slice - base_slice.mean()) - window)
    norm_density = norm / base_slice.size
    return norm_density


def absolute_threshold(scores, *thresholds):
    """
    Experimental shadowing metric

    Parameters
    ----------
    score_array : np.ndarray
        An array of scores

    threshold : float
        Upper bound for the shadowing metric to be labeled a 'detection'

    Returns
    -------
    mask : np.ndarray
        An array which masks the regions of spacetime which do not represent detections of Orbits.

    """
    if len(thresholds) == 1 and isinstance(*thresholds, tuple):
        thresholds = tuple(*thresholds)

    if len(thresholds) == len(scores):
        # if more than one threshold, assume that each subset of the score array is to be scored with
        # a different threshold, i.e. score array shape (5, 32, 32) and 5 thresholds, can use broadcasting to score
        # each (32, 32) slice independently.
        thresholds = np.array(thresholds).reshape(
            (-1, *tuple(1 for i in range(len(scores))))
        )
    # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.
    mask = np.zeros(scores.shape, dtype=bool)
    mask[scores <= thresholds] = True
    return mask


def pivot_iterator(
    pivot_array_shape, base_shape, window_shape, hull, core, periodicity, **kwargs
):
    """
    Generator for the valid window pivots for shadowing metric evaluation

    Parameters
    ----------
    pivot_array_shape : tuple of int
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
    for axis_coord, base_size, window_size, hull_size, core_size, periodic in zip(
        pivot_coordinates, base_shape, window_shape, hull, core, periodicity
    ):
        pad_size = hull_size - 1
        if periodic:
            mask = np.logical_or(mask, axis_coord < (pad_size - core_size))
            mask = np.logical_or(mask, axis_coord >= base_size)
        else:
            # The minimum pivot can't be further from boundary than window size, else some windows will be
            # completely out of bounds.
            mask = np.logical_or(
                mask,
                axis_coord
                < pad_size - int((1 - min_overlap_proportion) * (core_size - 1)),
            )
            # The maximum can't be determined by window size, as smaller windows can get "closer" to the edge;
            # this leads to results where only a subset of windows can provide scores, which is undesirable.
            mask = np.logical_or(
                mask,
                axis_coord
                >= base_size - int((1 - min_overlap_proportion) * (pad_size - 1)),
            )

    truncated_pivots = []
    for axis_coord, base_size, hull_size, periodic in zip(
        pivot_coordinates, base_shape, hull, periodicity
    ):
        truncated_pivots.append(axis_coord[~mask].reshape(-1, 1))
    # clean up and remove the potentially very large array.
    del pivot_coordinates
    # If there are no pivots to order, then we can quit here.
    if truncated_pivots:
        # order the pivots so they follow an inside-out approach.
        truncated_pivots = np.concatenate(truncated_pivots, axis=-1)
        approximate_centroid = truncated_pivots.max(axis=0) // 2
        ordered_pivots = truncated_pivots[
            np.argsort(np.sum(np.abs(approximate_centroid - truncated_pivots), axis=-1))
        ]
        return ordered_pivots
    else:
        # so return type are identical, cast as array
        return np.array(truncated_pivots)


def _subdomain_slices(pivot, base_orbit, window_orbit, hull, periodicity, **kwargs):
    """
    The slices for the window at the current pivot before any transformations.

    Parameters
    ----------
    pivot : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_orbit : Orbit
        The orbit being scanned
    window_orbit : Orbit
        The orbit being used to scan with
    hull : tuple of int
        Convex hull of the set of windows of the current run.
    periodicity : tuple of bool
        Elements indicate if corresponding axis is periodic. i.e. (True, False) implies axis=0 is periodic.

    Returns
    -------
    tuple, tuple
        The slices which produce the correct subdomains of the base orbit and window orbit for
        the shadowing metric function.

    """
    base_slices = []
    window_slices = []
    for piv, base_size, span, hull_size, periodic in zip(
        pivot, base_orbit.shape, window_orbit.shape, hull, periodicity
    ):
        pad_size = hull_size - 1
        # This allows generalization of rectangular to more nonlinear domains via functional mappings
        if kwargs.get("coordinate_map", None) is None and not periodic:
            base_start = max([piv, pad_size])
            base_end = min([piv + span, base_size + pad_size])
            base_slices.append(slice(base_start, base_end))
            # base_start - piv represents the starting point after shaving off points in the boundary.
            window_start = base_start - piv
            window_end = window_start + base_end - base_start
            window_slices.append(slice(window_start, window_end))
        else:
            base_slices.append(slice(piv, piv + span))
            window_slices.append(slice(None))

    return tuple(base_slices), tuple(window_slices)


def _subdomain_windows(
    pivot,
    orbit_with_hull,
    base_orbit,
    window_orbit,
    hull_grid,
    hull,
    periodicity,
    **kwargs,
):
    """

    Parameters
    ----------
    pivot : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_orbit : Orbit
        The orbit being scanned
    window_orbit : Orbit
        The orbit being used to scan with
    hull : tuple of int
        Convex hull of the set of windows of the current run.
    hull_grid : tuple of ndarray
        Equal to np.indices(hull); saved because it is re-used constantly.
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
        coordinate_map : callable
        A function which maps a "rectangular" array shape into the desired shadowing shape
        (typically parallelipiped)

    """
    coordinate_mapping_function = kwargs.get("coordinate_map", None)
    base_slices, window_slices = _subdomain_slices(
        pivot, base_orbit, window_orbit, hull, periodicity, **kwargs
    )

    if coordinate_mapping_function is not None:
        broadcasting_shaped_pivot = np.array(pivot).reshape(
            len(pivot), *tuple(len(pivot) * [1])
        )
        window_grid = hull_grid[(slice(None), *window_slices)]
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

    return base_subdomain, window_subdomain


def _subdomain_coordinates(
    pivot, base_orbit, window_orbit, hull_grid, hull, periodicity, **kwargs
):
    """

    Parameters
    ----------
    pivot : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_orbit : Orbit
        The orbit being scanned
    window_orbit : Orbit
        The orbit being used to scan with
    hull : tuple of int
        Convex hull of the set of windows of the current run.
    hull_grid : tuple of ndarray
        Equal to np.indices(hull); saved because it is re-used constantly.
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

    """
    coordinate_mapping_function = kwargs.get("coordinate_map", None)
    base_slices, window_slices = _subdomain_slices(
        pivot, base_orbit, window_orbit, hull, periodicity, **kwargs
    )
    broadcasting_shaped_pivot = np.array(pivot).reshape(
        len(pivot), *tuple(len(pivot) * [1])
    )

    # Get the subdomain's coordinates with respect to the padded array, if returning more than the pivot scores
    if coordinate_mapping_function is not None:
        broadcasting_shaped_pivot = np.array(pivot).reshape(
            len(pivot), *tuple(len(pivot) * [1])
        )
        window_grid = hull_grid[(slice(None), *window_slices)]
        subdomain_grid = window_grid + broadcasting_shaped_pivot
        # Add the pivot to translate the pivot; for proper broadcasting this requires reshaping
        mapped_subdomain_grid = coordinate_mapping_function(subdomain_grid, **kwargs)
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
    else:
        subdomain_grid = (
            hull_grid[(slice(None), *tuple(window_slices))] + broadcasting_shaped_pivot
        )
        if True in periodicity:
            for sub_coords, base_size, window_size, hull_size, periodic in zip(
                subdomain_grid, base_orbit.shape, window_orbit.shape, hull, periodicity,
            ):
                pad_size = hull_size - 1
                if periodic:
                    sub_coords[sub_coords >= (base_size + pad_size)] -= base_size
                    sub_coords[sub_coords < pad_size] += base_size
        # unravel and cast as tuples for basic and not advanced indexing
        subdomain_coordinates = tuple(c.ravel() for c in subdomain_grid)

    return subdomain_coordinates


def shadow(base_orbit, window_orbit, **kwargs):
    """
    Evaluate a scoring function with a window at all valid pivots of a base.

    Parameters
    ----------
    base_orbit : Orbit
        The orbit instance to scan over
    window_orbit : Orbit
        The orbit to scan with (the orbit that is translated around)

    Returns
    -------
    ndarray, ndarray :
        The scores at each pivot and their mapping onto an array the same shape as `base_orbit.state`

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
    core = kwargs.get("convex_core", None) or window_orbit.shape

    verbose = kwargs.get("verbose", False)
    if kwargs.get("padded_orbit", None) is not None:
        padded_orbit = kwargs.get("padded_orbit", None)
    else:
        padded_orbit = _pad_orbit_with_hull(
            base_orbit, hull, kwargs.get("periodicity", tuple(len(hull) * [False]))
        )

    pivot_scores = np.full_like(padded_orbit.state, np.inf)
    window_grid = np.indices(window_orbit.shape)
    ordered_pivots = pivot_iterator(
        pivot_scores.shape,
        base_orbit.shape,
        window_orbit.shape,
        hull,
        core,
        periodicity,
        **kwargs,
    )
    for i, each_pivot in enumerate(ordered_pivots):
        each_pivot = tuple(each_pivot)
        if verbose:
            if i != 0 and i % max([1, len(ordered_pivots) // 10]) == 0:
                print("-", end="")

        subdomain_tuple = _subdomain_windows(
            each_pivot,
            padded_orbit,
            base_orbit,
            window_orbit,
            window_grid,
            hull,
            periodicity,
            **kwargs,
        )
        base_subdomain, window_subdomain = subdomain_tuple
        if not base_subdomain.size > 0 or not window_subdomain.size > 0:
            warn_str = " ".join(
                [
                    f"\nshadowing pivot {each_pivot} unable to be scored for {repr(window_orbit)} because all",
                    f"subdomain coordinates were mapped out of bounds."
                ]
            )
            warnings.warn(warn_str, RuntimeWarning)
        else:
            pivot_scores[each_pivot] = scoring_function(
                base_subdomain, window_subdomain, **kwargs
            )

    return pivot_scores


def process_scores(
    scores, base_orbit, windows, base_periodicity, operation="trim", **kwargs
):
    """
    Manipulate the pivot scores returned by cover and shadow.

    """

    if (
        isinstance(windows, list)
        or isinstance(windows, tuple)
        or isinstance(windows, np.ndarray)
    ):
        hull = kwargs.get("convex_hull", None) or tuple(
            max([window.discretization[i] for window in windows])
            for i in range(len(base_orbit.discretization))
        )
        core = kwargs.get("convex_core", None) or tuple(
            min([window.discretization[i] for window in windows])
            for i in range(len(base_orbit.discretization))
        )
    else:
        hull = kwargs.get("convex_hull", None) or windows.shape
        core = kwargs.get("convex_core", None) or windows.shape
        windows = (windows,)

    if operation == "trim":
        if len(scores.shape) != base_orbit.ndim:
            pivot_array_shape = scores.shape[-len(base_orbit.shape) :]
        else:
            pivot_array_shape = scores.shape
            scores = scores[np.newaxis, ...]

        maximal_set_of_pivots = pivot_iterator(
            pivot_array_shape,
            base_orbit.shape,
            hull,
            hull,
            core,
            base_periodicity,
            **kwargs,
        )

        maximal_pivot_slices = tuple(
            slice(axis.min(), axis.max() + 1) for axis in maximal_set_of_pivots.T
        )
        return scores[(slice(None), *maximal_pivot_slices)]
    elif operation == "map":
        # Map the pivot scores back to the original orbit, every spacetime point taking the minimum out of
        # all computed metric values using that location.
        """
        Evaluate a scoring function with a window at all valid pivots of a base.
    
        Parameters
        ----------
        base_orbit : Orbit
            The orbit instance to scan over
        window_orbit : Orbit
            The orbit to scan with (the orbit that is translated around)
    
        Returns
        -------
        ndarray, ndarray :
            The scores at each pivot and their mapping onto an array the same shape as `base_orbit.state`
    
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
        if len(scores.shape) == base_orbit.ndim:
            # If a single score array, create a single index along axis=0 for iteration purposes.
            scores = scores[np.newaxis, ...]

        # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
        orbit_scores = np.full_like(
            _pad_orbit_with_hull(base_orbit, hull, base_periodicity).state, np.inf
        )[np.newaxis, ...]
        orbit_scores = np.repeat(orbit_scores, len(scores), axis=0)
        for index, (window, window_scores) in enumerate(zip(windows, scores)):
            window_grid = np.indices(window.shape)
            # For each set of scores, only need to map pivot scores that are non-infinite; therefore mask
            ordered_pivots = pivot_iterator(
                window_scores.shape,
                base_orbit.shape,
                window.shape,
                hull,
                core,
                base_periodicity,
                **{**kwargs, "mask": ~(window_scores < np.inf)},
            )

            for each_pivot in ordered_pivots:
                each_pivot = tuple(each_pivot)
                orbit_coordinates = _subdomain_coordinates(
                    each_pivot,
                    base_orbit,
                    window,
                    window_grid,
                    hull,
                    base_periodicity,
                    **kwargs,
                )
                if not np.size(orbit_coordinates) > 0:
                    warn_str = " ".join(
                        [
                            f"\nshadowing pivot {each_pivot} unable to be scored for {repr(window)} because all",
                            f"subdomain coordinates were mapped out of bounds."
                        ]
                    )
                    warnings.warn(warn_str, RuntimeWarning)

                filling_window = orbit_scores[(index, *orbit_coordinates)]
                filling_window[
                    (filling_window > window_scores[each_pivot])
                ] = window_scores[each_pivot]
                orbit_scores[(index, *orbit_coordinates)] = filling_window

        orbit_scores = orbit_scores[
            (
                slice(None),
                *tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in hull),
            )
        ]
        return orbit_scores


def _pad_orbit_with_hull(base_orbit, hull, periodicity, aperiodic_mode="constant"):
    """
    Pad the base orbit with the convex hull of the set of scanning windows.

    Parameters
    ----------
    base_orbit : Orbit
        The Orbit to be padded (and scanned)
    hull : tuple of int
        The shape of the convex hull of the set of windows
    periodicity : tuple of bool
        Flags indicating whether or not base_orbit state array axes represent periodic dimensions or not.
    aperiodic_mode : str or function
        How to treat padding of aperiodic dimensions.
    Returns
    -------
    padded_orbit : Orbit
        An orbit whose state array has been padded with zeros (aperiodic dimensions) or padded by wrapping values
        around the array (periodic dimensions).

    """
    # This looks redundant but it is not because two padding calls with different modes are required.
    periodic_padding = tuple(
        (pad - 1, pad - 1) if bc else (0, 0) for bc, pad in zip(periodicity, hull)
    )
    aperiodic_padding = tuple(
        (pad - 1, pad - 1) if not bc else (0, 0) for bc, pad in zip(periodicity, hull)
    )
    # Pad the base orbit state to use for computing scores, create the score array which will contain each pivot's score
    padded_state = np.pad(
        np.pad(base_orbit.state, periodic_padding, mode="wrap"),
        aperiodic_padding,
        mode=aperiodic_mode,
    )
    padded_orbit = base_orbit.__class__(
        **{
            **vars(base_orbit),
            "state": padded_state,
            "discretization": padded_state.shape,
        }
    )
    return padded_orbit


def cover(
    base_orbit,
    window_orbits,
    thresholds,
    replacement=False,
    reorder_by_size=True,
    trim=True,
    **kwargs,
):
    """
    Function to perform multiple shadowing computations given a collection of orbits.

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
    replacement : bool
        If False, then once a position in the score array is filled by a detection
    reorder_by_size : bool
        If True, then algorithm scans with largest windows first. Only matters if replacement is False.

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, [np.ndarray, np.ndarray])
        NumPy arrays whose indices along the first axis correspond to the window orbits' position in ```window_orbits```
        and whose other dimensions consist of scores or masking values. If return_pivot_arrays==True then also
        return the score and masking arrays for the pivots in addition to the orbit scores and pivots.

    """
    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # This only matters if replacement == False
    if reorder_by_size:
        window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
        iter_order = np.argsort(window_sizes)[::-1]
    else:
        iter_order = list(range(len(window_orbits)))

    # The maximum sizes along each dimension.
    hull = tuple(
        max([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )
    # The minimum sizes along each dimension.
    core = tuple(
        min([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )

    # Masking array is used for very specific or multi-part computations.
    masking_function = kwargs.get("masking_function", absolute_threshold)
    periodicity = kwargs.get("base_orbit_periodicity", tuple(len(hull) * [False]))
    padded_orbit = _pad_orbit_with_hull(base_orbit, hull, periodicity)

    mask = kwargs.get("mask", np.zeros(padded_orbit.shape, dtype=bool))
    pivot_scores = np.zeros([len(window_orbits), *padded_orbit.shape], dtype=float)
    number_of_possible_pivots = None
    for index, threshold, window_orbit in zip(
        iter_order, thresholds[iter_order], window_orbits[iter_order]
    ):
        if kwargs.get("verbose", False):
            print("#", end="")
        shadow_kwargs = {
            **kwargs,
            "convex_hull": hull,
            "convex_core": core,
            "mask": mask,
            "padded_orbit": padded_orbit,
        }
        window_pivot_scores = shadow(base_orbit, window_orbit, **shadow_kwargs)

        if not replacement:
            if mask is not None:
                # If all pivots have been filled then we can stop.
                if number_of_possible_pivots is None:
                    # if number of masked pivots ever equals this number, we are done. Adding original mask number
                    # simply makes it easier to calculate.
                    number_of_possible_pivots = (
                        window_pivot_scores[window_pivot_scores != np.inf].size
                        + mask.sum()
                    )
                mask = np.logical_or(
                    masking_function(window_pivot_scores, threshold), mask
                )
            else:
                # If all pivots have been filled then we can stop.
                if number_of_possible_pivots is None:
                    # if number of masked pivots ever equals this number, we are done.
                    number_of_possible_pivots = window_pivot_scores[
                        window_pivot_scores != np.inf
                    ].size
                mask = masking_function(window_pivot_scores, threshold)

            if mask.sum() == number_of_possible_pivots and index != iter_order[-1]:
                print(
                    f"Covering without replacement finished using only a subset of the provided orbits; terminating."
                )
                break

        pivot_scores[index, ...] = window_pivot_scores
    if trim:
        return process_scores(pivot_scores, base_orbit, window_orbits, periodicity)
    else:
        return pivot_scores


def fill(base_orbit, window_orbits, thresholds, **kwargs):
    """
    Function to perform multiple shadowing computations given a collection of orbits.

    Parameters
    ----------
    base_orbit : Orbit
        The Orbit to be covered
        The threshold for the masking function. Typically numerical in nature but need not be. The reason
        why threshold does not have a default value even though the functions DO have defaults is because the
        statistics of the scoring function depend on the base orbit and window orbits.
    window_orbits : array_like of Orbits
        Orbits to cover the base_orbit with. Typically a group orbit, as threshold is a single constant. Handling
        group orbits not included because it simply is another t

    Returns
    -------
    covering_masks : dict
        Dict whose keys are the index positions of the windows in the window_orbits provided, whose values
        are the ndarrays containing either the scores or boolean mask of the

    Notes
    -----
    This function is similar to cover, except that it checks the scores for each orbit at once, taking the best
    value as the winner. Now, it used to fill in the area after a successful detection but I changed this because
    it does not account for the fuzziness of shadowing. Thresholds need to be threshold densities if subtracting field.
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

    # The maximum sizes along each dimension.
    hull = tuple(
        max([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )
    # The minimum sizes along each dimension.
    core = tuple(
        min([window.discretization[i] for window in window_orbits])
        for i in range(len(base_orbit.discretization))
    )

    # Masking array is used for very specific or multi-part computations.
    periodicity = kwargs.get("base_orbit_periodicity", tuple(len(hull) * [False]))

    if kwargs.get("padded_orbit", None) is not None:
        padded_orbit = kwargs.get("padded_orbit", None)
    else:
        padded_orbit = _pad_orbit_with_hull(
            base_orbit, hull, kwargs.get("periodicity", tuple(len(hull) * [False]))
        )

    window_keys = kwargs.get("window_keys", range(1, len(window_orbits) + 1))
    # The following looks like a repeat of the same exact computation but unfortunately axis cannot
    # be provided in the padding function. meaning that all dimensions must be provided for both types of padding.

    # Number of discrete points will be the discretized spacetime.
    pivot_scores = np.full_like(padded_orbit.state, np.inf)
    orbit_weights = np.full_like(padded_orbit.state, np.inf)
    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # Want to sort by "complexity"; originally was discretization, now area; maybe norm?
    window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
    smallest_window_shape = window_orbits[np.argmin(window_sizes)].shape
    ordered_pivots = pivot_iterator(
        pivot_scores.shape,
        base_orbit.shape,
        smallest_window_shape,
        hull,
        core,
        periodicity,
        **kwargs,
    )
    hull_grid = np.indices(hull)
    for i, each_pivot in enumerate(ordered_pivots):
        each_pivot = tuple(each_pivot)
        if kwargs.get("verbose", True):
            if i != 0 and i % max([1, len(ordered_pivots) // 100]) == 0:
                print("-", end="")
        # See if the site is filled in the base_orbit array.
        # If a site has already been filled, then skip it and move to the next scanning position.

        # Need the correct slice of the base orbit with which to compute metric. Take the largest possible
        # slice based on the largest shape that contains all windows, then take _subdomains of this slice accordingly
        window_scores = []
        window_slices = []
        # Once the correct field slice has been retrieved, score it against all of the windows.
        for window in window_orbits:
            slices = _subdomain_slices(
                each_pivot, base_orbit, window, hull, periodicity, **kwargs
            )
            # Need the base orbit slice and window to be the same size (typically), therefore, take slices
            # with respect to the smaller size in each dimension. If the same size, this slicing does nothing.
            base_subdomain, window_subdomain = _subdomain_windows(
                each_pivot,
                padded_orbit,
                base_orbit,
                window,
                hull_grid,
                hull,
                periodicity,
                **kwargs,
            )
            if not base_subdomain.size > 0 or not window_subdomain.size > 0:
                warn_str = " ".join(
                    [
                        f"\nshadowing pivot {each_pivot} unable to be scored for {repr(window)} because all",
                        f"subdomain coordinates were mapped out of bounds."
                    ]
                )
                warnings.warn(warn_str, RuntimeWarning)
                window_scores.append(np.inf)
            else:
                # subdomain coordinates are the coordinates in the score array that account for periodic boundary conditions
                window_scores.append(
                    scoring_function(base_subdomain, window_subdomain, **kwargs)
                )
                window_slices.append(slices)
        window_score_ratios = window_scores / thresholds
        minimum_score = np.min(window_score_ratios)
        minimum_score_index = int(np.argmin(window_score_ratios))
        pivot_scores[each_pivot] = minimum_score
        # thresholds are relative to the window orbits; use relative differences to determine how to fill.
        if minimum_score < 1:
            detection_coordinates = _subdomain_coordinates(
                each_pivot,
                base_orbit,
                window_orbits[minimum_score_index],
                hull_grid,
                hull,
                periodicity,
                **kwargs,
            )
            # Replace window values with new minimum value if less than previous values.
            filling_window = orbit_weights[detection_coordinates]
            filling_window[filling_window > minimum_score] = window_keys[
                minimum_score_index
            ]
            orbit_weights[detection_coordinates] = filling_window
            # Subtract the orbit state from the padded base orbit used to compute the score.
            if kwargs.get("subtract_field", False):
                # algorithms allowed for subtraction of the underlying base field.
                padded_orbit.state[detection_coordinates] = 0
                padded_orbit.state[window_slices[minimum_score_index]] = 0

    orbit_weights = process_scores(
        orbit_weights,
        base_orbit,
        window_orbits,
        periodicity,
        operation="trim",
        **kwargs,
    )
    return pivot_scores, orbit_weights[0, ...]
