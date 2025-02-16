import numpy as np
import warnings
import tqdm
from joblib import Parallel, delayed
from json import dumps

__all__ = ["scoring_functions", "OrbitCovering"]


class OrbitCovering:

    def __init__(self, reference_orbit, windows, scores=None, periodicity=None):
        """
        Class which bundles all information and data relevant for covering/shadowing computations.

        Parameters
        ----------

        reference_orbit : Orbit
            The Orbit instance whose field state is to be scanned.
        windows: list, tuple or array of Orbit instances
            An iterable of Orbits which we are looking for in the base state; the orbits shadowed.
        scores: np.array or np.ma.masked_array
            An array that serves as a container for all scores from all windows. Useful to pass if scores
            are being re-used from past calculations.
        periodicity : tuple of bool
            Indicators for whether dimensions of reference orbit's field are to be taken as periodic or not.
            Will affect scoring condition as aperiodic dimensions are not wrapped around during scoring.
        thresholds: np.array
            Array of threshold values, ordered respectively of windows. Typically will simply be passed
            to function OrbitCovering.threshold_scores instead, allows to keep track of what values
            were used to threshold a covering.


        """
        self.reference_orbit = reference_orbit

        if not periodicity:
            self.periodicity = tuple([False] * len(reference_orbit.discretization))
        else:
            self.periodicity = periodicity

        if (
            isinstance(windows, list)
            or isinstance(windows, tuple)
            or isinstance(windows, np.ndarray)
        ):
            self.windows = np.array(windows)
        else:
            self.windows = np.array((windows,))

        self.hull = tuple(
            max([window.discretization[i] for window in windows])
            for i in range(len(windows[0].discretization))
        )
        self.core = tuple(
            min([window.discretization[i] for window in windows])
            for i in range(len(windows[0].discretization))
        )

        self.padded_orbit = _pad_orbit(reference_orbit, self.hull, self.periodicity)

        # Only have to iterate once per unique discretization shape, take advantage of this.
        self.all_window_shapes = [tuple(w.discretization) for w in windows]
        # array_of_window_shapes = np.array(all_window_shapes)
        self.unique_window_shapes = set(self.all_window_shapes)

        if scores is None:
            scores = np.full((len(self.windows), *self.padded_orbit.shape), np.inf)
            self._scores = np.ma.masked_array(
                scores, mask=np.ones_like(scores, dtype=bool)
            )
        elif isinstance(scores, np.ma.masked_array):
            self._scores = scores
        else:
            raise TypeError(
                f"If provided, scores must be of type {type(np.ma.masked_array())} or {type(np.array)}"
            )

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        dict_ = {"base shape": self.reference_orbit.shape,
                 "scores_shape": self.scores.shape,
                 "windows": len(self.windows)}
        # convert the dictionary to a string via json.dumps
        dictstr = dumps(dict_)
        return self.__class__.__name__ + "(" + dictstr + ")"

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, array):
        self._scores = array

    def threshold_scores(self, thresholds):
        """
        Experimental shadowing metric

        Parameters
        ----------
        thresholds : np.ndarray
            Upper bound for the shadowing metric to be labeled a 'detection'

        Returns
        -------
        OrbitCovering:
            An orbit covering whose scores have been masked if not satisfying respective thresholds.

        """
        scores = self.scores.copy()
        if len(thresholds) == 1 and isinstance(*thresholds, tuple):
            thresholds = tuple(*thresholds)

        thresholds = np.array(thresholds)

        if thresholds.size == scores.shape[0]:
            # if more than one threshold, assume that each subset of the score array is to be scored with
            # a different threshold, i.e. score array shape (5, 32, 32) and 5 thresholds, can use broadcasting to score
            # each (32, 32) slice independently.
            thresholds = np.array(thresholds).reshape(
                (-1, *tuple(1 for i in range(len(scores.shape) - 1)))
            )
        # Depending on the pivots supplied, some places in the scoring array may be empty; do not threshold these elements.

        scores.mask = scores.data > thresholds
        return self.__class__(
            **{
                "reference_orbit": self.reference_orbit,
                "scores": scores,
                "windows": self.windows,
                "periodicity": self.periodicity,
            }
        )

    def score(
        self,
        pivot_mask=None,
        coordinate_map=None,
        scoring_method="l2_density_mfc",
        ignore_oob=True,
        min_overlap=1.0,
        **kwargs,
    ):
        """
        Function which computes a metric function between reference orbit and window orbits.

        Parameters
        ----------

        scores: np.array or np.ma.masked_array
            An array that serves as a container for all scores from all windows. Useful to pass if scores
            are being re-used from past calculations.
        coordinate_map : function, callable, default None
            Maps an array of indices (i.e. that which would slice a describes a hypercube/rectangle/orthotope into
            an arbitrary selection of indices. Allows for arbitrarily complex slicing of base fields. Common application
            would be to map a hypercube into a parallelogram.
        scoring_method : function, callable or str, default 'l2_density_mfc'
            A function which maps (reference window, window) -> scalar
        ignore_oob : bool, default True
            If False, then warns the user when pivots are being mapped out of bounds.
        min_overlap : float, default 1.
            Defines the fraction of the window that is required to overlap the reference orbit, required to be a value
            within interval (0, 1]. For example, if a window had a rectangular discretization 32 x 32 = 1024 lattice sites
            and min_overlap was 0.5, then at least 512 lattice sites would be required to receive a score. Only needed
            for aperiodic dimensions, and should only really be used if scoring metric is a density / insensitive to
            discretization size.
        pivot_mask: np.ndarray
            Boolean array required to be same shape as scores array (for a single window). Values of "True"
            will result in a pivot point NOT being scored.

        """

        """
        Function to perform multiple shadowing computations given a collection of orbits.

        Parameters
        ----------
        orbit_cover : OrbitCovering
            Object which contains the base orbit, window orbits, thresholds, and everything else.

        Returns
        -------
        tuple : (orbit_cover, np.ndarray, [np.ndarray, np.ndarray])
            NumPy arrays whose indices along the first axis correspond to the window orbits' position in ```window_orbits```
            and whose other dimensions consist of scores or masking values. If return_pivot_arrays==True then also
            return the score and masking arrays for the pivots in addition to the orbit scores and pivots.

        Notes
        -----
        If a pivot mask is not provided, then any elements of the score array which are beneath threshold are not recalculated.


        """
        # Get reference orbit/OrbitScorer attributes prepped
        reference_orbit = self.reference_orbit
        # if padded_orbit is None:

        window_orbits = np.array(self.windows)

        # Masking array is used for very specific or multi-part computations.
        periodicity = self.periodicity

        oob_pivots = []
        # Score mask is True if scores are INVALID. If they are ALL invalid, then we do NOT want to mask them
        # in the scoring process, hence the pivot should be False

        if pivot_mask is None:
            pivot_mask = np.zeros_like(self.scores[0, :, :])

        if pivot_mask.shape != self.padded_orbit.shape:
            raise ValueError("Pivot mask must be same shape are padded orbit array.")

        # By selecting all windows
        for index, window_shape in enumerate(self.unique_window_shapes):
            where_this_shape = tuple(
                i
                for i, shape in enumerate(self.all_window_shapes)
                if shape == window_shape
            )
            relevant_window_states = np.array(
                [window_orbits[i].state for i in where_this_shape]
            )
            # retrieve the function for scoring, using a default function based on the l2 difference.
            scoring_function = kwargs.get(
                "scoring_function", scoring_functions(scoring_method)
            )

            ordered_pivots = pivot_iterator(
                self.scores.shape[1:],
                reference_orbit.shape,
                self.hull,
                self.core,
                periodicity,
                pivot_mask,
                min_overlap=min_overlap,
            )

            for w_dim, b_dim in zip(window_shape, reference_orbit.shape):
                assert (
                    w_dim < b_dim
                ), "Shadowing window discretization is larger than the base orbit in at least 1 dimension; resize first. "

            window_grid = np.indices(window_shape)
            window_oob_pivots = []
            # break

            for i, each_pivot in tqdm.tqdm(enumerate(ordered_pivots),
                                            desc=f"Scoring pivots for windows of size {window_shape}",
                                            ncols=100,
                                            position=0,
                                            leave=True,
                                        ):
                each_pivot = tuple(each_pivot)
                score_mask_slicer = (np.array(where_this_shape), *each_pivot)

                base_indexer, window_indexer = _subdomain_windows(
                    each_pivot,
                    reference_orbit.shape,
                    window_shape,
                    window_grid,
                    self.hull,
                    periodicity,
                    coordinate_map=coordinate_map,
                    **kwargs,
                )
                base_subdomain = self.padded_orbit.state[base_indexer]
                window_subdomains = relevant_window_states[
                    (slice(None), *window_indexer)
                ]

                # if subdomains are empty, there is nothing to score.
                if not base_subdomain.size > 0 or not window_subdomains[0].size > 0:
                    if self.return_oob:
                        window_oob_pivots.append(each_pivot)
                else:
                    self.scores[score_mask_slicer] = scoring_function(
                        base_subdomain, window_subdomains, **kwargs
                    )

            if len(window_oob_pivots) > 0 and not ignore_oob:
                error_msg = f"""\n{len(window_oob_pivots)} shadowing pivot unable to be scored for one or more windows because all
                        subdomain coordinates were mapped out-of-bounds, typically as a result of a coordinate_map. 
                        This is disallowed unless ignore_oob=True is explicitly specified to
                        avoid unintentionally returning a score array with NaN values"""
                raise ValueError(error_msg)

        self.oob_pivots = oob_pivots
        return self

    def trim(self, remove_hull_only=True, min_overlap=1):
        """
        Remove all unscored pivots (i.e. padding) from the score arrays. This does NOT necessarily
        have the same shape as the reference orbit as it depends on min_overlap and periodicity
        of different dimensions.
        """

        assert self.scores is not None, "Cannot trim an empty set of scores."

        if remove_hull_only:
            trimmed_scores = self.scores[
                (
                    slice(None),
                    *tuple(
                        slice(hull_size - 1, -(hull_size - 1))
                        for hull_size in self.hull
                    ),
                )
            ]

        else:
            # If a mask was applied on a subdomain, return the region of spacetime scores which have finite values.
            maximal_set_of_pivots = pivot_iterator(
                self.padded_orbit.shape,
                self.reference_orbit.shape,
                self.hull,
                self.core,
                self.periodicity,
                False,
                min_overlap=min_overlap,
            )
            maximal_pivot_slices = tuple(
                slice(axis.min(), axis.max() + 1) for axis in maximal_set_of_pivots.T
            )
            trimmed_scores = self.scores[(slice(None), *maximal_pivot_slices)]

        return self.__class__(
            **{
                "reference_orbit": self.reference_orbit,
                "scores": trimmed_scores,
                "windows": self.windows,
                "periodicity": self.periodicity,
            }
        )

    def map(
        self, min_overlap=1, coordinate_map=None, n_cores=1
    ):
        """
        Return the cumulative minimum at each space-time lattice site when mapping
        scores at each pivot back onto the spacetime.

        Parameters
        ----------
        scores: np.ma.masked_array
            (Untrimmed) array of scores returned by OrbitCovering.score

        coordinate_map: function
            Function used in scoring which manipulates the "window shape".

        Returns
        -------
        ndarray :
            Cumulative minimum of score array after mapping scores back onto space-time lattice.

        Notes
        -----
        This takes the scores returned by :func:`cover` and converts them to the appropriate orbit sized array)

        I found this hard to vectorize because of the inequality condition and inplace operations, so parallelized
        via joblib instead.

        """

        if (
            int(
                np.invert(self.scores.mask)
                .sum(axis=tuple(range(1, len(self.scores.shape))))
                .max()
            )
            >= 0.5 * self.reference_orbit.size
        ):
            warnings.warn(
                f"""POTENTIALLY UNFILTERED SCORES DETECTED: 
                            Mapping scores which have not had thresholds applied can take a long time due to having to map
                            every single pivot back into spacetime. If you are going to threshold the scores afterwards
                            anyway, it is highly recommended to apply thresholding BEFORE mapping. If scores have had
                            thresholds applied then ignore this message."""
            )

        # The bases orbit periodicity has to do with scoring and whether or not to wrap windows around.
        def _joblib_wrapper(
            single_window,
            single_window_scores,
            reference_orbit_shape,
            hull,
            core,
            periodicity,
            min_overlap,
            coordinate_map,
        ):
            score_array = np.full_like(single_window_scores, np.inf)
            window_shape = single_window.shape
            window_grid = np.indices(window_shape)

            ordered_pivots = pivot_iterator(
                single_window_scores.shape,
                reference_orbit_shape,
                hull,
                core,
                periodicity,
                single_window_scores.mask,
                min_overlap=min_overlap,
            )
            for each_pivot in ordered_pivots:
                each_pivot = tuple(each_pivot)
                orbit_coordinates = _subdomain_coordinates(
                    each_pivot,
                    reference_orbit_shape,
                    window_shape,
                    window_grid,
                    hull,
                    periodicity,
                    coordinate_map=coordinate_map,
                )

                if np.size(orbit_coordinates) > 0:
                    pivot_score = single_window_scores[*each_pivot]
                    filling_window = score_array[*orbit_coordinates]
                    filling_window[filling_window > pivot_score] = pivot_score
                    score_array[*orbit_coordinates] = filling_window

            return score_array

        with Parallel(n_jobs=n_cores) as parallel:
            mapped_scores_per_window = parallel(
                delayed(_joblib_wrapper)(
                    window,
                    self.scores[index],
                    self.reference_orbit.shape,
                    self.hull,
                    self.core,
                    self.periodicity,
                    min_overlap,
                    coordinate_map,
                )
                for index, window in tqdm.tqdm(
                    enumerate(self.windows),
                    desc="Mapping scores back onto windows",
                    ncols=100,
                    leave=True,
                    position=0,
                )
            )

        mapped_scores = np.concatenate(
            [x[None, :, :] for x in mapped_scores_per_window], axis=0
        )
        mapped_orbit_covering = self.__class__(
            **{
                "reference_orbit": self.reference_orbit,
                "scores": mapped_scores,
                "windows": self.windows,
                "periodicity": self.periodicity,
            }
        )
        return mapped_orbit_covering

    def minimal_covering_set(self, cover_threshold=0.99):
        """
        Find the smallest number of masks which cover a specified proportion of the total space-time area.
        If 100 windows were used for scoring, but only 5 windows are needed to cover space-time, this
        will return an OrbitCovering with 5 windows. Ordered by % of space-time covered.

        Parameters
        ----------
        cover_threshold: float
            float in (0, 1) which is the target % of spacetime covered.

        Returns
        -------
        minimal_orbitcover: OrbitCovering
            OrbitCovering with a number of windows required to cover `cover_threshold` percentage
            of space-time

        """
        thresheld_scores = self.scores
        assert (
            isinstance(cover_threshold, float) and cover_threshold < 1
        ), "cover threshold must be provided as a float in interval (0, 1)."

        cover_percentages = {}

        assert (
            thresheld_scores.mask.sum() != 0
        ), "Scores have not had thresholds applied or thresholds are too lax."

        # Find the orbit mask with the largest covering.
        detections = np.invert(thresheld_scores.mask).sum(
            axis=tuple(range(1, len(thresheld_scores.mask.shape)))
        )
        next_best_orbit_idx = np.argmax(detections)
        minimal_cover_inclusion_flags = np.zeros(
            thresheld_scores.mask.shape[0], dtype=bool
        )
        minimal_cover_inclusion_flags[next_best_orbit_idx] = True
        # percentage of unmasked points accounted for.
        valid_pivots = np.invert(thresheld_scores.mask)
        total_valid_pivots = np.any(valid_pivots, axis=0).astype(bool).sum()
        minimal_cover = valid_pivots[minimal_cover_inclusion_flags, ...]
        cover_percentages[next_best_orbit_idx] = (
            valid_pivots[next_best_orbit_idx, ...].sum() / total_valid_pivots
        )

        total_cover_percentage = (
            np.any(minimal_cover, axis=0).astype(bool).sum() / total_valid_pivots
        )
        while total_cover_percentage < cover_threshold:
            # The next-best orbit cover to add is the one which covers the most uncovered area.
            covered = np.any(minimal_cover, axis=0)
            currently_uncovered = np.logical_and(
                valid_pivots, np.invert(covered).reshape((1, *covered.shape))
            )
            # Find the next best mask to add by totaling the number of points it covers which are not in minimal cover yet.
            next_greatest_contribution_index = np.argmax(
                currently_uncovered.sum(
                    axis=tuple(range(1, len(self.reference_orbit.shape) + 1))
                )
            )
            # demarcate the newer member of the cover.
            minimal_cover_inclusion_flags[next_greatest_contribution_index] = True
            minimal_cover = valid_pivots[minimal_cover_inclusion_flags, ...]
            cover_percentages[next_greatest_contribution_index] = (
                valid_pivots[next_greatest_contribution_index, ...].sum()
                / total_valid_pivots
            )
            total_cover_percentage = (
                np.any(minimal_cover, axis=0).astype(bool).sum() / total_valid_pivots
            )

        minimal_orbitcover = self.__class__(
            **{
                "reference_orbit": self.reference_orbit,
                "scores": thresheld_scores[minimal_cover_inclusion_flags, ...],
                "windows": self.windows[minimal_cover_inclusion_flags, ...],
                "periodicity": self.periodicity,
            }
        )
        return (
            minimal_orbitcover,
            cover_percentages,
            np.arange(len(self.windows))[minimal_cover_inclusion_flags],
        )

    def best_subset(self, n):
        """
        Similar to minimal_covering_set, except this takes specified number of
        windows to keep as opposed to filling until a threshold is met.

        Parameters
        ----------
        n: int
            The number of windows to reduce to; i.e. the "top n" windows.

        Returns
        -------
        cover_subset: OrbitCovering
            OrbitCovering with a subset of "n" windows.

        """
        thresheld_scores = self.scores
        assert (
            thresheld_scores.mask.sum() != 0
        ), "Scores have not had thresholds applied or thresholds are too lax."

        detections = np.invert(thresheld_scores.mask).sum(
            axis=tuple(range(1, len(thresheld_scores.shape)))
        )
        best_n = np.argsort(detections)[-n:]
        best_n_masked_scores = thresheld_scores[best_n, ...]

        cover_subset = self.__class__(
            **{
                "reference_orbit": self.reference_orbit,
                "scores": best_n_masked_scores,
                "windows": self.windows[best_n, ...],
                "periodicity": self.periodicity,
            }
        )
        return cover_subset, best_n


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
    return np.sum(
        np.abs(base_slice**2 - window**2),
        axis=tuple(range(1, len(base_slice.shape) + 1)),
    )


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
    return np.linalg.norm(
        base_slice - window, axis=tuple(range(1, len(base_slice.shape) + 1))
    )


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
    norm = np.linalg.norm(
        base_slice - window, axis=tuple(range(1, len(base_slice.shape) + 1))
    )
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
    norm = np.linalg.norm(
        base_slice[base_mask] - window[base_mask],
        axis=tuple(range(1, len(base_slice.shape) + 1)),
    )
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
        axis=tuple(range(1, len(base_slice.shape) + 1)),
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
    norm = np.linalg.norm(
        (base_slice - base_slice.mean()) - window,
        axis=tuple(range(1, len(base_slice.shape) + 1)),
    )
    norm_density = norm / base_slice.size
    return norm_density


def pivot_iterator(
    pivot_array_shape,
    base_shape,
    hull,
    core,
    periodicity,
    mask,
    min_overlap=1,
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
                mask, axis_coord < pad_size - int((1 - min_overlap) * (core_size - 1))
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
    /,
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
    /,
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
            mapped_subdomain_grid,
            base_shape,
            window_shape,
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
        # if all boundaries are aperiodic, then we are done.
        if True in periodicity:
            for sub_coords, base_size, window_size, hull_size, periodic in zip(
                subdomain_grid,
                base_shape,
                window_shape,
                hull,
                periodicity,
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
