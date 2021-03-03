import numpy as np
from .persistent_homology import persistence_array_converter
__all__ = ['scan', 'shadow', 'fill', 'cover', 'scanning_mask']


def amplitude_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice**2 - window**2)


def l2_difference(base_slice, window, *args, **kwargs):
    return np.linalg.norm(base_slice - window)


def masked_l2_difference(base_slice, window, *args, **kwargs):
    base_mask = base_slice.copy().astype(bool)
    norm = np.linalg.norm(base_slice[base_mask] - window[base_mask])
    norm_density = norm / max([1, base_mask.sum()])
    return norm_density


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
    Because of the wrap technique used in the shadowing function, need a function that converts
    the score array (whose shape is determined by the pivots) into a mask of the same shape as the underlying base orbit.
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


def pivot_iterator(scan_shape, strides=(1, 1)):
    pivots = np.ndindex(scan_shape)
    for pivot_tuple in pivots:
        yield tuple(stride*p for p, stride in zip(pivot_tuple, strides))


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
    cache = kwargs.get('cache', None)
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
    base_orbit_periodicity = kwargs.get('base_orbit_periodicity', tuple(len(base_orbit.dimensions())*[False]))

    score_array_shape, pad_shape = scanning_dimensions(base.shape, window.shape, strides, base_orbit_periodicity)
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
        # the periodic_dimensions key here determines the periodic dimensions in the gudhi.PeriodicCubicalComplex
        # See notes for why this is set to True even though slices are expected to be aperiodic.
        gudhikwargs = kwargs.get('gudhi_kwargs', {'periodic_dimensions': tuple(len(window.shape)*[True]),
                                                  'min_persistence': 0.01})
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
        for i, pivot_tuple in enumerate(iterator):
            # Required to cache the persistence with the correct key.
            base_pivot_tuple = tuple(stride*p for p, stride in zip(pivot_tuple, strides))
            # If the current pivot doesn't have a stored value, then calculate it and add it to the cache.
            # Pivot tuples iterate over the score_array, not the actual base orbit, need to convert.
            if cache is None or len(persistence_array_converter(cache, base_pivot_tuple)) == 0:
                if verbose and i % max([1, n_pivots//10]) == 0:
                    print('-', end='')
                # by definition base slices cannot be periodic unless w_dim == b_dim and that dimension is periodic.
                # To get the slice indices, find the pivot point (i.e. 'corner' of the window) and then add
                # the window's dimensions.
                window_slices = []
                # The CORNER is not given by pivot * span. That is pivot * stride. The WIDTH is +span
                for pivot, span, stride in zip(pivot_tuple, window.shape, strides):
                    window_slices.append(slice(pivot * stride, pivot * stride + span))
                base_slice_orbit = window_orbit.__class__(**{**vars(window_orbit),
                                                             'state': pbase[tuple(window_slices)]})
                base_orbit_pivot_persistence = persistence_function(base_slice_orbit, **gudhikwargs)
                persistence_array = persistence_array_converter(base_orbit_pivot_persistence, base_pivot_tuple)
                if caching:
                    if cache is None:
                        cache = persistence_array.copy()
                    else:
                        cache = np.concatenate((cache, persistence_array), axis=0)
            else:
                if verbose and i % max([1, n_pivots//10]) == 0:
                    print('+', end='')
                base_orbit_pivot_persistence = persistence_array_converter(cache, base_pivot_tuple)
            score_array[pivot_tuple] = scoring_function(base_orbit_pivot_persistence, window_persistence, **gudhikwargs)

        # Once the persistences are computed for the pivots for a given slice shape, can include them in the cache.

        # Periodicity of windows determined by the periodic_dimensions() method.
        # .ravel() returns a view, therefore can use it to broadcast.
        # Unfortunately what I want to do is only supported if the base slice and window have same periodic dimensions?
        return score_array, cache
    else:
        # Pointwise scoring can use the amplitude difference default.
        scoring_function = kwargs.get('scoring_function', l2_difference)

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
            score_array[pivot_tuple] = scoring_function(base_slice_orbit.state, window_orbit.state, *args, **kwargs)
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

    cache = kwargs.get('cache', None)
    masking_function = kwargs.get('masking_function', absolute_threshold)
    # Whether or not previous runs have "cached" the persistence information; not really caching, mind you.
    # This doesn't do anything unless either mask_type=='family' or score_type=='persistence'. One or both must be true.

    thresholds, window_orbits = np.array(thresholds), np.array(window_orbits)
    # Want to sort by "complexity"; originally was discretization, now area; maybe norm?
    window_sizes = [np.prod(w.dimensions()) for w in window_orbits]
    if not replacement:
        assrt_str = 'unequal window discretizations require replacement=True.'
        assert len(np.unique([w.size for w in window_orbits])) == 1, assrt_str

    if reorder_by_size:
        size_order = np.argsort(window_sizes)[::-1]
    else:
        size_order = list(range(len(window_orbits)))
    mask = None
    covering_masks = {}
    for index, threshold, window in zip(size_order, thresholds[size_order], window_orbits[size_order]):
        if kwargs.get('verbose', False) and index % max([1, len(thresholds)//10]) == 0:
            print('#', end='')
        # Just in case of union method
        scores, cache = scan(base_orbit, window, **{**kwargs, 'cache': cache, 'mask': mask})
        scores_bool = masking_function(scores, threshold)
        if not replacement:
            if mask is not None:
                # The "new" detections; removing the masked values
                scores_bool = np.logical_xor(scores_bool, mask)
                # all detections
                mask = np.logical_or(scores_bool, mask)
            else:
                mask = scores_bool.copy()

        # If there were no detections then do not add anything to the cover.
        if scores_bool.sum() > 0:
            if dtype is bool:
                covering_masks[index] = scores_bool.copy()
            else:
                covering_masks[index] = scores.copy().astype(dtype)
    return covering_masks


def inside_to_outside_iterator(padded_shape, base_shape, hull, verbose=False):
    if verbose:
        print('Scanning interior pivots')
    i = 0
    for pivot_position in np.ndindex(base_shape):
        i += 1
        if i % (np.prod(padded_shape)//100) == 0. and i != 0. and verbose:
            print('#', end='')
        yield tuple(p+h for p, h in zip(pivot_position, hull))
    if verbose:
        print('Scanning exterior pivots')
    # Pivot positions completely consisting of padding are redundant
    for pivot_position in np.ndindex(tuple(p - h for p, h in zip(padded_shape, hull))):
        i += 1
        if i % (np.prod(padded_shape)//100) == 0. and i != 0. and verbose:
            print('#', end='')
        # a hack for "slicing" the generator.
        exterior_checker = tuple(p < hl for p, shp, hl in zip(pivot_position, base_shape, hull))
        if True in exterior_checker:
            yield pivot_position
        else:
            continue


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
    # Determine the pivots to iterate over and score, as well as the array to do so.
    # For the periodic dimensions, pad with wrapped values.
    periodic_padding = tuple((pad, pad) if bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    aperiodic_padding = tuple((pad, pad) if not bc else (0, 0) for bc, pad in zip(base_orbit_periodicity, hull))
    # If padded then
    padded_base_orbit = base_orbit.__class__(**{**vars(base_orbit),
                                                'state': np.pad(np.pad(base_orbit.state, periodic_padding, mode='wrap'),
                                                                aperiodic_padding)})

    # Number of discrete points will be the discretized spacetime.
    base_size = np.prod(base_orbit.discretization)
    # Allow for storage of floats, might want to plug in physical observable value.
    padded_base_orbit_mask = np.zeros(padded_base_orbit.shape, dtype=float)
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
            detected_orbit_index = int(np.argmin(pivot_scores))
            if pivot_scores[detected_orbit_index] <= thresholds[detected_orbit_index]:
                grid = np.indices(window_orbits[detected_orbit_index].shape)
                pivot_grid = tuple(g + p * s for g, p, s in zip(grid, pivot, strides))
                # To account for wrapping, fold indices which extend beyond the base orbit's extent using periodicity.
                # If aperiodic then this has no effect, as the coordinates are dependent upon the *padded* base state
                # If one of the dimensions is not periodic, then need to ensure that all coordinates of the
                # window slice are within the boundaries of the base orbit.
                # if False in base_orbit_periodicity:
                coord_mask = np.ones(window_orbits[detected_orbit_index].shape, dtype=bool)
                wrap_coord_mask = np.ones(coord_mask.shape, dtype=bool)
                base_coordinates = []
                wrapped_periodic_coordinates = []
                periodic_coordinates = []
                for coordinates, base_extent, periodic, h in zip(pivot_grid, base_orbit.shape,
                                                                 base_orbit_periodicity, hull):
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
                        wrap_coord_mask = np.logical_and(wrap_coord_mask, (wrap < (base_extent+2*h)))
                        wrap_coord_mask = np.logical_and(wrap_coord_mask, (wrap > 0))

                        # If periodic then all coordinates in this dimension are valid.
                        coord[coord >= (base_extent+h)] = coord[coord >= (base_extent+h)] - base_extent
                        coord[coord < h] = coord[coord < h] + base_extent
                    else:
                        if True in base_orbit_periodicity:
                            # only need to add to the
                            wrapped_periodic_coordinates.append(coordinates)
                            periodic_coordinates.append(coordinates)
                        # If out of bounds in any of the dimensions, need to exclude.
                        coord_mask = np.logical_and(coord_mask, (coord < base_extent+h) & (coord >= h))
                    base_coordinates.append(coord)

                pivot_grid = tuple(coordinates[coord_mask] for coordinates in base_coordinates)
                # Get the correct amount of space-time which is *about* to be filled; this is the proportion added to
                # weights. The "filled" qualifier at the beginning of the loop just checked whether the
                # *pivot* was filled, NOT the entirety of the window.
                filling_window = padded_base_orbit_mask[pivot_grid]
                # This is the number of unfilled points divided by total number of points in base discretization
                unfilled_spacetime_within_window = filling_window[filling_window == 0.].size / base_size
                # use the window_keys list and not the actual dict keys because repeat keys are allowed for families.
                weights[window_keys[detected_orbit_index]] += unfilled_spacetime_within_window
                # Once we count what the compute the unfilled space-time, can then fill it in with the correct value.
                filling_window[filling_window == 0.] = window_keys[detected_orbit_index]
                # Finally, need to fill in the actual base orbit mask the window came from.
                # This fills the correct amount of space-time with a key related to the "best" fitting window orbit
                padded_base_orbit_mask[pivot_grid] = filling_window
                # Subtract the orbit state from the padded base orbit used to compute the score.
                if kwargs.get('subtract_field', False):
                    padded_base_orbit.state[pivot_grid] = 0
                if True in base_orbit_periodicity:
                    wrapped_pivot_grid = tuple(coordinates[wrap_coord_mask] for coordinates
                                               in wrapped_periodic_coordinates)
                    padded_pivot_grid = tuple(c for c in periodic_coordinates)
                    # The mask is only affected by the original, internal points, but the padded field needs
                    # to set padding equal to zero if it is nonzero (periodic padding).
                    if kwargs.get('subtract_field', False):
                        padded_base_orbit.state[wrapped_pivot_grid] = 0
                        padded_base_orbit.state[padded_pivot_grid] = 0
    # Only want to return the filling of the original orbit, padding is simply a tool
    base_orbit_mask = padded_base_orbit_mask[tuple(slice(pad, -pad) for pad in hull)]
    filled_base_orbit = padded_base_orbit[tuple(slice(pad, -pad) for pad in hull)]
    return filled_base_orbit, base_orbit_mask, weights
