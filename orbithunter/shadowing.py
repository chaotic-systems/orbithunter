import numpy as np
import warnings
from json import dumps

__all__ = ["cover", "scoring_functions", "OrbitCover"]


class OrbitCover:
    """
    Class which bundles all information and data relevant for covering/shadowing computations.

    Parameters
    ----------

    base : Orbit
        The Orbit instance whose state is to be scanned.
    windows: list, tuple or array
        An iterable of orbits which we are looking for in the base state; the orbits shadowed.
    thresholds : list, tuple, array
        The values used to judge whether a shadowed orbit has been detected or not. Order respects the order of
        window orbits.
    hull : tuple, default None
        Tuple whose elements are the maximum of the discretization in each dimension; see Notes for details
    core : tuple, default None
        Tuple whose elements are the minima of the discretization in each dimension; see Notes for details
    mask : array
        Masking array for the set of pivots to evaluate scoring function at; if True then a pivot is masked, i.e.
         if an element of the array is True then the scoring function is NOT evaluated there.
    padded_orbit : Orbit, default None
        The padded version of `base`, it is allowed to be passed because it can be computationally expensive to
        repeatedly produce.
    coordinate_map : function, callable, default None
        Maps an array of indices (i.e. that which would slice a describes a hypercube/rectangle/orthotope into
        an arbitrary selection of indices. Allows for arbitrarily complex slicing of base fields. Common application
        would be to map a hypercube into a parallelogram.
    window_caching_function : function, callable
        If the scoring function requires a calculation which is expensive but only needs to be performed once per
        window (i.e. persistent homology of an orbit) then this argument can be used to propagate the result through
        the covering/shadowing process.
    scoring_method : function, callable or str, default 'l2_density_mfc'
        Takes in slices of base orbit and window orbit fields and returns a scalar value.
    return_oob : bool
        If coordinate_map is used, then it is possible to have the computational domain be mapped completely outside
        of the base orbit; to keep track of these points they are allowed to be returned as a list of tuples.
    ignore_oob : bool, default True
        If False, then warns the user that pivots are being mapped out of bounds.
    replacement : bool, default False
        If True then allows pivots to be scored more than once, even after a "detection" has been recorded.
    min_overlap : float, default 1.
        Defines the fraction of the window that is required to overlap the base orbit, required to be a value
        within interval (0, 1].
    cover_proportion :
        The fraction of unmasked pivots that need to have shadowing detections recorded before the computation
        is potentially terminated midway. Value should be in (0, 1] e.g. if cover_proportion is 0.9, then computation
        is terminated after 90% of the pivots have orbits detected at them.
    periodicity : tuple of bool
        Indicates whether a dimension of the base orbit's field is periodic.

    """

    def __init__(
        self,
        base,
        windows,
        thresholds,
        hull=None,
        core=None,
        mask=None,
        padded_orbit=None,
        coordinate_map=None,
        window_caching_function=None,
        scoring_method='l2_density_mfc',
        return_oob=False,
        ignore_oob=True,
        replacement=True,
        min_overlap=1,
        cover_proportion=1,
        periodicity=(),
            **kwargs
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
        self.thresholds = np.array(thresholds).reshape(
            (-1, *(len(base.shape)*(1,))))
        self.cover_proportion = cover_proportion
        self.hull = hull or tuple(
            max([window.discretization[i] for window in windows])
            for i in range(len(base.discretization))
        )
        self.core = core or tuple(
            min([window.discretization[i] for window in windows])
            for i in range(len(base.discretization))
        )
        if not periodicity:
            self.periodicity = tuple([False] * len(base.discretization))
        else:
            self.periodicity = periodicity
        self._scores = kwargs.get("scores", kwargs.get("_scores", None))

        if padded_orbit is None:
            self.padded_orbit = _pad_orbit(base, self.hull, self.periodicity)
        else:
            self.padded_orbit = padded_orbit
        self.coordinate_map = coordinate_map
        self.window_caching_function = window_caching_function
        self.min_overlap = min_overlap
        self.scoring_function = scoring_functions(scoring_method)
        self.return_oob = return_oob
        self.ignore_oob = ignore_oob
        self.replacement = replacement
        if not self.replacement and (len(self.thresholds) != len(self.windows)):
            raise ValueError('If scoring without replacement, need to have a threshold for each window orbit.')

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

    def trim(self, non_inf_only=True):
        """
        Remove all unscored pivots (i.e. padding) from the score arrays. NOT necessarily the same shape as base orbit.

        """
        assert self.scores is not None, "cannot trim an empty set of scores."
        if len(self.scores.shape) != self.base.ndim:
            pivot_array_shape = self.scores.shape[-len(self.base.shape) :]
        else:
            pivot_array_shape = self.scores.shape
            self.scores = self.scores[np.newaxis, ...]

        if non_inf_only:
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
        else:
            trimmed_orbit_scores = self.scores[(slice(None),
                *tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in self.hull))
            ]
            return trimmed_orbit_scores

    def map(self, verbose=False):
        """ Map scores representing detections back onto the spatiotemporal tile from the original base orbit.

        Parameters
        ----------
        verbose : bool
            Whether to print '-' as a form of crude progress bar



        Returns
        -------
        ndarray :
            The scores at each pivot and their mapping onto an array the same shape as `base_orbit.state`

        Notes
        -----
        This takes the scores returned by :func:`cover` and converts them to the appropriate orbit sized array)

        """
        if len(self.scores.shape) == self.base.ndim:
            # If a single score array, create a single index along axis=0 for iteration purposes.
            scores = self.scores[np.newaxis, ...]
        else:
            scores = self.scores

        oob_pivots = []

        # Only have to iterate once per unique discretization shape, take advantage of this.
        all_window_shapes = [tuple(w.discretization) for w in self.windows]

        # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
        orbit_scores = np.full_like(
            np.zeros([len(all_window_shapes), *_pad_orbit(self.base, self.hull, self.periodicity).state.shape]), np.inf
        )

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
                        # for window_idx in where_this_shape:
                        best_score_idx = np.argmin(self.scores[(slice(None), *each_pivot)])
                        pivot_score = self.scores[(best_score_idx, *each_pivot)]
                        #  easier to just check scores instead of masks.
                        if pivot_score <= self.thresholds[best_score_idx]:
                            filling_window = orbit_scores[(best_score_idx, *orbit_coordinates)]
                            filling_window[filling_window > pivot_score] = pivot_score
                            orbit_scores[(best_score_idx, *orbit_coordinates)] = filling_window

        if len(oob_pivots) > 0 and not self.ignore_oob:
            warn_str = " ".join(
                [
                    f"\n{len(oob_pivots)} shadowing pivots unable to be scored for one or more windows because all",
                    f"subdomain coordinates were mapped out-of-bounds. To disable this message and return the"
                    f" out-of-bounds pivots along with the scores, oob_pivots, set return_oob=True or ignore_oob=True",
                ]
            )
            warnings.warn(warn_str, RuntimeWarning)

        trimmed_orbit_scores = orbit_scores[(slice(None),
            *tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in self.hull))
        ]
        return trimmed_orbit_scores

    def threshold(self):
        """
        Masks scores which do not satisfy the provided threshold constraint.

        """
        masked_scores = np.ma.masked_array(
            self.scores,
            mask=(
                self.scores > self.thresholds
            ),
        )
        self.masked_scores = masked_scores
        trimmed_masked_scores = masked_scores[(slice(None),
            *tuple(slice(hull_size - 1, -(hull_size - 1)) for hull_size in self.hull))
        ]
        return trimmed_masked_scores


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
            f"name {method} of scoring function not in methods provided by orbithunter; define\n"
            f"callable externally if still desired to be passed to shadowing functions."
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
    return np.sum(np.abs(base_slice ** 2 - window ** 2),
        axis=tuple(range(1, len(base_slice.shape) + 1)))


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
    return np.linalg.norm(base_slice - window,
        axis=tuple(range(1, len(base_slice.shape) + 1)))


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
    norm = np.linalg.norm(base_slice - window,
        axis=tuple(range(1, len(base_slice.shape) + 1)))
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
    norm = np.linalg.norm(base_slice[base_mask] - window[base_mask],
        axis=tuple(range(1, len(base_slice.shape) + 1)))
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
        (base_slice[base_mask] - base_slice[base_mask].mean()) - window[base_mask],
        axis=tuple(range(1, len(base_slice.shape) + 1)))
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
    norm = np.linalg.norm((base_slice - base_slice.mean()) - window,
                                  axis=tuple(range(1, len(base_slice.shape) + 1)))
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
            mask = np.logical_or(mask, axis_coord < pad_size)
            mask = np.logical_or(mask, axis_coord >= base_size + pad_size)
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
    if np.array(truncated_pivots).flatten().size > 0:
        # order the pivots so they follow an inside-out approach.
        truncated_pivots = np.concatenate(truncated_pivots, axis=-1)
        approximate_centroid = truncated_pivots.max(axis=0) // 2
        ordered_pivots = truncated_pivots[
            np.argsort(np.sum(np.abs(approximate_centroid - truncated_pivots), axis=-1))
        ]
        return ordered_pivots
    else:
        # so return type are identical, cast as array
        return np.array(truncated_pivots).flatten()


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
        # Add the pivot to translate the pivot; for proper broadcasting this requires reshaping
        subdomain_grid = window_grid + broadcasting_shaped_pivot
        # Specify which type of coordinates are manipulated explicitly, may be unused, depending on coordinate_map
        mapped_subdomain_grid = coordinate_map(subdomain_grid, base=True, **kwargs)
        mapped_window_grid = coordinate_map(window_grid, window=True, **kwargs)
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
        mapped_subdomain_grid = coordinate_map(subdomain_grid, base=True, **kwargs)
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
        # if all boundaries are aperiodic, then we are done.
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
    orbit_cover : OrbitCover
        Object which contains the base orbit, window orbits, thresholds, and everything else.

    Returns
    -------
    tuple : (orbit_cover, np.ndarray, [np.ndarray, np.ndarray])
        NumPy arrays whose indices along the first axis correspond to the window orbits' position in ```window_orbits```
        and whose other dimensions consist of scores or masking values. If return_pivot_arrays==True then also
        return the score and masking arrays for the pivots in addition to the orbit scores and pivots.

    Notes
    -----
    If replacement == False, mask will be overwritten to reflect that a pivot has been "filled".

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
            periodicity
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
            "scoring_function", l2_difference_mean_flow_correction_density
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
            ):
                print(
                    f"Covering without replacement finished early; only a subset of the provided orbits were needed"
                    f" to cover {100*orbit_cover.cover_proportion}% of the set of unmasked pivots provided."
                )
                break

    orbit_cover.mask = mask
    if orbit_cover.return_oob:
        orbit_cover.oob_pivots = oob_pivots
    return orbit_cover
