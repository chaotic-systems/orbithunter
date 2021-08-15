import numpy as np
import warnings
from json import dumps

__all__ = ["cover", "scoring_functions", "OrbitCover"]


class OrbitCover:
    def __init__(
        self,
        base,
        windows,
        thresholds,
        hull=None,
        core=None,
        mask=None,
        min_overlap=1,
        cover_proportion=1,
        **kwargs,
    ):
        self.base = base

        if (
            isinstance(windows, list)
            or isinstance(windows, tuple)
            or isinstance(windows, np.ndarray)
        ):
            self.windows = windows
        else:
            self.windows = (windows,)
        self.mask = mask
        self.thresholds = np.array(thresholds)
        self.cover_proportion = cover_proportion
        self.hull = hull or tuple(
            max([window.discretization[i] for window in windows])
            for i in range(len(base.discretization))
        )
        self.core = core or tuple(
            min([window.discretization[i] for window in windows])
            for i in range(len(base.discretization))
        )
        self.periodicity = kwargs.get(
            "periodicity", tuple([False] * len(base.discretization))
        )
        self._scores = kwargs.get("scores", kwargs.get("_scores", None))
        self.padded_orbit = kwargs.get("padded_orbit", None)
        self.coordinate_map = kwargs.get("coordinate_map", None)
        self.window_caching_function = kwargs.get("window_caching_function", None)
        self.min_overlap = min_overlap
        self.scoring_function = kwargs.get(
            "scoring_function", l2_difference_mean_flow_correction_density
        )
        self.return_oob = kwargs.get("return_oob", False)
        self.ignore_oob = kwargs.get("ignore_oob", False)
        self.replacement = kwargs.get("replacement", False)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        dict_ = {"base shape": self.base.shape, "windows": len(self.windows)}
        # convert the dictionary to a string via json.dumps
        dictstr = dumps(dict_)
        return self.__class__.__name__ + "(" + dictstr + ")"

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, array):
        self._scores = array

    def trim(self, **kwargs):
        assert self.scores is not None, "cannot trim an empty set of scores."
        if len(self.scores.shape) != self.base.ndim:
            pivot_array_shape = self.scores.shape[-len(self.base.shape) :]
        else:
            pivot_array_shape = self.scores.shape
            self.scores = self.scores[np.newaxis, ...]

        maximal_set_of_pivots = pivot_iterator(
            pivot_array_shape,
            self.base.shape,
            self.hull,
            self.core,
            self.periodicity,
            min_overlap=self.min_overlap,
            mask=np.all(~(self.scores < np.inf), axis=0),
        )

        maximal_pivot_slices = tuple(
            slice(axis.min(), axis.max() + 1) for axis in maximal_set_of_pivots.T
        )
        return self.scores[(slice(None), *maximal_pivot_slices)]

    def map(self, **kwargs):
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
        verbose = kwargs.get("verbose", True)
        if len(self.scores.shape) == self.base.ndim:
            # If a single score array, create a single index along axis=0 for iteration purposes.
            scores = self.scores[np.newaxis, ...]
        else:
            scores = self.scores
        # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
        orbit_scores = np.full_like(
            _pad_orbit(self.base, self.hull, self.periodicity).state, np.inf
        )
        oob_pivots = []

        # Only have to iterate once per unique discretization shape, take advantage of this.
        all_window_shapes = [tuple(w.discretization) for w in self.windows]
        # array_of_window_shapes = np.array(all_window_shapes)
        unique_window_shapes = set(all_window_shapes)
        # By iterating over shapes and not windows, cut down on
        for index, window_shape in enumerate(unique_window_shapes):
            where_this_shape = tuple(
                i for i, shape in enumerate(all_window_shapes) if shape == window_shape
            )
            # relevant_window_states = np.array([self.windows[i].state for i in where_this_shape])
            # retrieve the function for scoring, using a default function based on the l2 difference.
            threshold_broadcasting_reshape = tuple([-1] + (len(scores.shape) - 1) * [1])
            window_grid = np.indices(window_shape)
            # For each set of scores, only need to map pivot scores that are sub-threshold

            mask_insufficient_scores = np.logical_not(
                np.any(
                    scores[where_this_shape, ...]
                    <= self.thresholds[np.array(where_this_shape)].reshape(
                        *threshold_broadcasting_reshape
                    ),
                    axis=0,
                )
            )
            if mask_insufficient_scores.size - mask_insufficient_scores.sum() > 0:
                ordered_pivots = pivot_iterator(
                    self.padded_orbit.shape,
                    self.base.shape,
                    self.hull,
                    self.core,
                    self.periodicity,
                    min_overlap=self.min_overlap,
                    mask=mask_insufficient_scores,
                )
                min_pivot_scores = self.scores[where_this_shape, ...].min(axis=0)
                for i, each_pivot in enumerate(ordered_pivots):
                    each_pivot = tuple(each_pivot)
                    if verbose:
                        if i != 0 and i % max([1, len(ordered_pivots) // 10]) == 0:
                            print("-", end="")

                    orbit_coordinates = _subdomain_coordinates(
                        each_pivot,
                        self.base.shape,
                        window_shape,
                        window_grid,
                        self.hull,
                        self.periodicity,
                        coordinate_map=self.coordinate_map,
                    )

                    if np.size(orbit_coordinates) > 0:
                        filling_window = orbit_scores[orbit_coordinates]
                        filling_window[
                            filling_window > min_pivot_scores[each_pivot]
                        ] = min_pivot_scores[each_pivot]
                        orbit_scores[orbit_coordinates] = filling_window

        if len(oob_pivots) > 0 and not self.ignore_oob:
            warn_str = " ".join(
                [
                    f"\n{len(oob_pivots)} shadowing pivots unable to be scored for one or more windows because all",
                    f"subdomain coordinates were mapped out-of-bounds. To disable this message and return the"
                    f" out-of-bounds pivots along with the scores, oob_pivots, set return_oob=True or ignore_oob=True",
                ]
            )
            warnings.warn(warn_str, RuntimeWarning)

        orbit_scores = orbit_scores[
            tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in self.hull)
        ]
        return orbit_scores

    def threshold(self, *args, **kwargs):
        threshold_broadcasting_reshape = tuple(
            [-1] + (len(self.scores.shape) - 1) * [1]
        )

        masked_scores = np.ma.masked_array(
            self.scores,
            mask=(
                self.scores > self.thresholds.reshape(*threshold_broadcasting_reshape)
            ),
        )
        self.masked_scores = masked_scores
        return self


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
    return np.linalg.norm(
        (base_slice - base_slice.mean())[np.newaxis, ...] - window,
        axis=tuple(range(1, len(base_slice.shape) + 1)),
    )


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


def absolute_threshold(scores, thresholds):
    """
    Experimental shadowing metric

    Parameters
    ----------
    score_array : np.ndarray
        An array of scores

    threshold : np.ndarray
        Upper bound for the shadowing metric to be labeled a 'detection'

    Returns
    -------
    mask : np.ndarray
        An array which masks the regions of spacetime which do not represent detections of Orbits.

    """
    if len(thresholds) == 1 and isinstance(*thresholds, tuple):
        thresholds = tuple(*thresholds)

    if thresholds.size == scores.shape[0]:
        # if more than one threshold, assume that each subset of the score array is to be scored with
        # a different threshold, i.e. score array shape (5, 32, 32) and 5 thresholds, can use broadcasting to score
        # each (32, 32) slice independently.
        thresholds = np.array(thresholds).reshape(
            (-1, *tuple(1 for i in range(len(scores.shape) - 1)))
        )
    # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.
    mask = np.zeros(scores.shape, dtype=bool)
    mask[scores <= thresholds] = True
    return mask


def pivot_iterator(
    pivot_array_shape, base_shape, hull, core, periodicity, min_overlap=1, mask=None
):
    """
    Generator for the valid window pivots for shadowing metric evaluation

    Parameters
    ----------
    pivot_array_shape : tuple of int
        Shape of the numpy array corresponding to the padded base orbit
    base_shape : tuple of int
        Shape of the numpy array corresponding to the base orbit
    hull : tuple of int
        Shape of the convex hull of a set of windows (largest size in each dimension).
    core : tuple of int
        Shape of smallest discretization sizes in each dimension.
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

    The masking that occurs ensures that every pivot which is scored, is scored with respect to ALL windows

    """
    if mask is None:
        mask = np.zeros(pivot_array_shape, dtype=bool)

    pivot_coordinates = np.indices(pivot_array_shape)
    for axis_coord, base_size, hull_size, core_size, periodic in zip(
        pivot_coordinates, base_shape, hull, core, periodicity
    ):
        pad_size = hull_size - 1
        if periodic:
            mask = np.logical_or(mask, axis_coord < (pad_size - core_size))
            mask = np.logical_or(mask, axis_coord >= base_size)
        else:
            # The minimum pivot can't be further from boundary than window size, else some windows will be
            # completely out of bounds.
            mask = np.logical_or(
                mask, axis_coord < pad_size - int((1 - min_overlap) * (core_size - 1)),
            )
            # The maximum can't be determined by window size, as smaller windows can get "closer" to the edge;
            # this leads to results where only a subset of windows can provide scores, which is undesirable.
            mask = np.logical_or(
                mask,
                axis_coord
                >= (base_size + pad_size) - int(min_overlap * (hull_size - 1)),
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


def _subdomain_slices(
    pivot, base_shape, window_shape, hull, periodicity, coordinate_map=None
):
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
        pivot, base_shape, window_shape, hull, periodicity
    ):
        pad_size = hull_size - 1
        # This allows generalization of rectangular to more nonlinear domains via functional mappings
        if coordinate_map is None and not periodic:
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
    base_shape,
    window_shape,
    window_grid,
    hull,
    periodicity,
    coordinate_map=None,
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
    base_slices, window_slices = _subdomain_slices(
        pivot,
        base_shape,
        window_shape,
        hull,
        periodicity,
        coordinate_map=coordinate_map,
    )

    if coordinate_map is not None:
        broadcasting_shaped_pivot = np.array(pivot).reshape(
            len(pivot), *tuple(len(pivot) * [1])
        )
        window_grid = window_grid[(slice(None), *window_slices)]
        subdomain_grid = window_grid + broadcasting_shaped_pivot
        # Add the pivot to translate the pivot; for proper broadcasting this requires reshaping
        mapped_subdomain_grid = coordinate_map(subdomain_grid, **kwargs)
        mapped_window_grid = coordinate_map(window_grid, **kwargs)
        submask = np.zeros(mapped_subdomain_grid.shape[1:], dtype=bool)
        for sub_coords, base_size, window_size, hull_size, periodic in zip(
            mapped_subdomain_grid, base_shape, window_shape, hull, periodicity
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
        base_indexer = subdomain_coordinates
        window_indexer = window_coordinates
    else:
        base_indexer = base_slices
        window_indexer = window_slices

    return base_indexer, window_indexer


def _subdomain_coordinates(
    pivot,
    base_shape,
    window_shape,
    hull_grid,
    hull,
    periodicity,
    coordinate_map=None,
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

    """
    base_slices, window_slices = _subdomain_slices(
        pivot,
        base_shape,
        window_shape,
        hull,
        periodicity,
        coordinate_map=coordinate_map,
    )
    broadcasting_shaped_pivot = np.array(pivot).reshape(
        len(pivot), *tuple(len(pivot) * [1])
    )

    # Get the subdomain's coordinates with respect to the padded array, if returning more than the pivot scores
    if coordinate_map is not None:
        broadcasting_shaped_pivot = np.array(pivot).reshape(
            len(pivot), *tuple(len(pivot) * [1])
        )
        window_grid = hull_grid[(slice(None), *window_slices)]
        subdomain_grid = window_grid + broadcasting_shaped_pivot
        # Add the pivot to translate the pivot; for proper broadcasting this requires reshaping
        mapped_subdomain_grid = coordinate_map(subdomain_grid, **kwargs)
        submask = np.zeros(mapped_subdomain_grid.shape[1:], dtype=bool)
        for sub_coords, base_size, window_size, hull_size, periodic in zip(
            mapped_subdomain_grid, base_shape, window_shape, hull, periodicity,
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
                subdomain_grid, base_shape, window_shape, hull, periodicity,
            ):
                pad_size = hull_size - 1
                if periodic:
                    sub_coords[sub_coords >= (base_size + pad_size)] -= base_size
                    sub_coords[sub_coords < pad_size] += base_size
        # unravel and cast as tuples for basic and not advanced indexing
        subdomain_coordinates = tuple(c.ravel() for c in subdomain_grid)

    return subdomain_coordinates


def _pad_orbit(base_orbit, padding_dim, periodicity, aperiodic_mode="constant"):
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
        (pad - 1, pad - 1) if bc else (0, 0)
        for bc, pad in zip(periodicity, padding_dim)
    )
    aperiodic_padding = tuple(
        (pad - 1, pad - 1) if not bc else (0, 0)
        for bc, pad in zip(periodicity, padding_dim)
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


def cover(orbit_cover, verbose=False, **kwargs):
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
    thresholds = np.array(orbit_cover.thresholds)
    window_orbits = np.array(orbit_cover.windows)
    base_orbit = orbit_cover.base

    # Only have to iterate once per unique discretization shape, take advantage of this.
    all_window_shapes = [tuple(w.discretization) for w in window_orbits]
    # array_of_window_shapes = np.array(all_window_shapes)
    unique_window_shapes = set(all_window_shapes)
    # Masking array is used for very specific or multi-part computations.
    masking_function = kwargs.get("masking_function", absolute_threshold)
    periodicity = orbit_cover.periodicity

    if orbit_cover.padded_orbit is None:
        padded_orbit = _pad_orbit(base_orbit, orbit_cover.hull, periodicity)
        orbit_cover.padded_orbit = padded_orbit
    else:
        padded_orbit = orbit_cover.padded_orbit

    if orbit_cover.mask is not None:
        mask = orbit_cover.mask
    else:
        mask = kwargs.get("mask", np.zeros(padded_orbit.shape, dtype=bool))

    original_masked_pivots = mask.sum()
    orbit_cover.scores = np.full_like(
        np.zeros([len(window_orbits), *padded_orbit.shape]), np.inf
    )

    oob_pivots = []

    if padded_orbit is None:
        padded_orbit = _pad_orbit(
            base_orbit,
            orbit_cover.hull,
            kwargs.get("periodicity", tuple(len(orbit_cover.hull) * [False])),
        )

    for index, window_shape in enumerate(unique_window_shapes):
        where_this_shape = tuple(
            i for i, shape in enumerate(all_window_shapes) if shape == window_shape
        )
        relevant_window_states = np.array(
            [window_orbits[i].state for i in where_this_shape]
        )
        # retrieve the function for scoring, using a default function based on the l2 difference.
        scoring_function = kwargs.get(
            "scoring_function", l2_difference_mean_flow_correction
        )
        # Sometimes there are metrics, like bottleneck distance between persistence diagrams,
        # which need only be computed once per window. To avoid redundant calculations, cache this result.
        # if orbit_cover.window_caching_function is not None:
        ordered_pivots = pivot_iterator(
            orbit_cover.scores.shape[1:],
            base_orbit.shape,
            orbit_cover.hull,
            orbit_cover.core,
            periodicity,
            min_overlap=orbit_cover.min_overlap,
            mask=mask,
        )
        for w_dim, b_dim in zip(window_shape, base_orbit.shape):
            assert (
                w_dim < b_dim
            ), "Shadowing window discretization is larger than the base orbit. resize first. "

        # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
        periodicity = kwargs.get(
            "base_orbit_periodicity", tuple(len(base_orbit.dimensions()) * [False])
        )

        window_grid = np.indices(window_shape)
        window_oob_pivots = []
        for i, each_pivot in enumerate(ordered_pivots):
            each_pivot = tuple(each_pivot)

            if verbose:
                if i != 0 and i % max([1, len(ordered_pivots) // 10]) == 0:
                    print("-", end="")

            base_indexer, window_indexer = _subdomain_windows(
                each_pivot,
                base_orbit.shape,
                window_shape,
                window_grid,
                orbit_cover.hull,
                periodicity,
                coordinate_map=orbit_cover.coordinate_map,
                **kwargs,
            )
            base_subdomain = padded_orbit.state[base_indexer]
            window_subdomains = relevant_window_states[(slice(None), *window_indexer)]

            # if subdomains are empty, there is nothing to score.
            if not base_subdomain.size > 0 or not window_subdomains[0].size > 0:
                if orbit_cover.return_oob:
                    window_oob_pivots.append(each_pivot)
            else:
                orbit_cover.scores[
                    (np.array(where_this_shape), *each_pivot)
                ] = scoring_function(base_subdomain, window_subdomains, **kwargs)

        if len(window_oob_pivots) > 0 and not orbit_cover.ignore_oob:
            warn_str = " ".join(
                [
                    f"\n{len(window_oob_pivots)} shadowing pivot unable to be scored for one or more windows because all",
                    f"subdomain coordinates were mapped out-of-bounds. To disable this message and return the"
                    f" out-of-bounds pivots along with the scores, oob_pivots, set return_oob=True or ignore_oob=True",
                ]
            )
            warnings.warn(warn_str, RuntimeWarning)

        if not orbit_cover.replacement:
            mask = np.logical_or(
                np.any(
                    masking_function(
                        orbit_cover.scores[where_this_shape, :],
                        thresholds[np.array(where_this_shape)],
                    ),
                    axis=0,
                ),
                mask,
            )

            # If the proportion of pivots exceeds our target, we can stop. This is based on the number ORIGINALLY
            # UNMASKED pivots, not the mask size.
            if (
                (mask.sum() - original_masked_pivots)
                / (mask.size - original_masked_pivots)
                >= orbit_cover.cover_proportion
            ) and index != len(unique_window_shapes) - 1:
                print(
                    f"Covering without replacement finished early; only a subset of the provided orbits were needed"
                    f" to cover {100*orbit_cover.cover_proportion}% of the set of unmasked pivots provided."
                )
                break
    orbit_cover.mask = mask
    if orbit_cover.return_oob:
        orbit_cover.oob_pivots = oob_pivots
        return orbit_cover, oob_pivots
    else:
        return orbit_cover
