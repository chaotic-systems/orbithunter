from .io import to_symbol_string
import numpy as np
import itertools
from collections import Counter

__all__ = [
    "tile",
    "glue",
    "generate_symbol_arrays",
    "rediscretize_tileset",
    "expensive_pairwise_glue",
    "aspect_ratio_correction",
]


def aspect_ratio_correction(orbit_array, axis=0, conserve_parity=True):
    """
    Resize a collection of Orbits' discretizations according to their sizes in the dimension specified by axis.

    Parameters
    ----------
    orbit_array : ndarray
        An array of orbits to resize along 'axis'
    axis : int
        ndarray axis along which to resize with respect to.
    conserve_parity : bool
        Whether or not to maintain parity of the discretization size for each orbit. This is relevant when
        certain bases require either odd or even numbered discretization sizes.

    Returns
    -------
    ndarray :
        An array of resized orbits, same shape as `orbit_array`.

    Notes
    -----
    Note that this *will* allow equilibria tiles to become very distorted; this can be managed by gluing
    order but typically when including equilibria, strip-wise corrections cause too much distortion to be useful.
    This is not an issue, however, as this method should never really be used for dramatically different sized Orbits
    unless the distortions are permitted.

    """
    # Get the dimensions and corresponding discretization sizes.
    dims = np.array([o.dimensions()[axis] for o in orbit_array])
    sizes = np.array([o.shapes()[0][axis] for o in orbit_array])
    disc_total = np.sum(sizes)
    dim_total = np.sum(dims)

    # The absolute minimum is set to the smallest number of points which doesn't completely "contract" the dimension.
    # Whether or not this takes an even (2) or odd (3) value is inferred from the total number of discrete points
    # along the given axis for the array of orbits provided. To avoid orbits with small dimension being
    # undersized, use some minimum fraction based upon the strip length.

    # If dimensions are all zero (dimensions taken to be non-negative), then no resizing can be inferred.
    # Also, if orbit array being glued is of shape (N,1) then orbit array may contain single orbit.
    if dim_total == 0.0 or len(orbit_array.ravel()) == 1:
        return orbit_array

    new_discretization_sizes = np.zeros(len(orbit_array))

    if conserve_parity:
        # the values returned by minimal shape are the absolute minimum. If the parity is wrong, then the practical
        # minimum must be at least one greater.
        min_disc_sizes = []
        for i, min_size in enumerate(o.minimal_shape()[axis] for o in orbit_array):
            if (min_size % 2) != (sizes[i] % 2):
                min_disc_sizes.append(min_size + 1)
            else:
                min_disc_sizes.append(min_size)
        min_disc_sizes = np.array(min_disc_sizes)
        # The parity is built into the minimum discretization sizes; therefore, the total sizes are these minimum
        # sizes plus an even number if parity is to be maintained. The best way (that I've found) to maintain
        # this even parity in the remainder is to distribute
        disc_remainder = disc_total - np.sum(min_disc_sizes)
        disc_remainder //= 2
        dim_remainder = dim_total
        for sorted_index in np.argsort(dims)[:-1]:
            orbits_share = int(disc_remainder * (dims[sorted_index] / dim_remainder))
            new_discretization_sizes[sorted_index] = min_disc_sizes[sorted_index] + (
                2 * orbits_share
            )
            dim_remainder -= dims[sorted_index]
            disc_remainder -= orbits_share
        # Whatever the last orbit is, give it the remainder of the discretization.
        new_discretization_sizes[np.argsort(dims)[-1]] = min_disc_sizes[
            np.argsort(dims)[-1]
        ] + (2 * disc_remainder)
    else:
        min_disc_sizes = np.array([o.minimal_shape()[axis] for o in orbit_array])
        disc_remainder = disc_total - np.sum(min_disc_sizes)
        dim_remainder = dim_total

        for sorted_index in np.argsort(dims)[:-1]:
            orbits_share = int(disc_remainder * (dims[sorted_index] / dim_remainder))
            new_discretization_sizes[sorted_index] = (
                min_disc_sizes[sorted_index] + orbits_share
            )
            dim_remainder -= dims[sorted_index]
            disc_remainder -= orbits_share
        # Whatever the last orbit is, give it the remainder of the discretization.
        new_discretization_sizes[np.argsort(dims)[-1]] = min_disc_sizes[
            np.argsort(dims)[-1]
        ] + (2 * disc_remainder)

    # from smallest to largest, increase the share of the discretization
    new_shapes = [
        tuple(
            int(new_discretization_sizes[j]) if i == axis else o.shapes()[0][i]
            for i in range(len(o.shape))
        )
        for j, o in enumerate(orbit_array)
    ]

    # Return the strip of orbits with corrected proportions.
    return np.array([o.resize(shp) for o, shp in zip(orbit_array, new_shapes)])


def glue(orbit_array, orbit_type, strip_wise=False, **kwargs):
    """
    Combines the state arrays of a configuration of Orbits

    Parameters
    ----------
    orbit_array : ndarray of Orbit instances
        A NumPy array wherein each element is an orbit. i.e. a tensor of Orbit instances. The shape should be
        representative to how the orbits are going to be glued. See notes for more details. The orbits must all
        have the same discretization size if gluing is occuring along more than one axis. The orbits should
        all be in the physical field basis.
    orbit_type : Orbit type
        The class that the final result will be returned as.
    strip_wise : bool
        If True, then the "strip-wise aspect ratio correction" is applied. See :func:`aspect_ratio_correction`.

    Returns
    -------
    glued_orbit : Orbit
        Instance of type orbit_type, whose state and dimensions are the combination of the original array of orbits.

    Notes
    -----
    Because of how the concatenation of fields works, wherein the discretization must match along the boundaries. It
    is quite complicated to write a generalized code that glues all dimensions together at once for differently sized,
    orbit, so instead this is designed to iterate through the axes of the orbit_array.

    To prevent confusion, there are many different notions of 'shape' and 'discretization' that are relevant.
    There are three main array shapes or dimensions that are involved in this function. The first is the array
    of orbits, which represents a spatiotemporal symbolic "dynamics" block. This array can have as many dimensions
    as the solutions to the equation have. An array of orbits of shape (2,1,1,1) means that the fields have four
    continuous dimensions. The specific shape means that two such fields
    are being concatenated in time (because the first axis should always be time) by orbithunter convention.

    Example for the spatiotemporal Navier-stokes equation. The spacetime is (1+3) dimensional. Let's assume we're gluing
    two vector fields with the same discretization size, (N, X, Y, Z). We can think of this discretization as a
    collection of 3D vector field snapshots in time. Therefore, there are actually (N, X, Y, Z, 3) degrees of freedom.
    Therefore the actually tensor before the gluing will be of the shape (2, 1, 1, 1, N, X, Y, Z, 3).
    Because we are gluing the orbits along the time axis, The final shape will be (1, 1, 1, 1, 2*N, X, Y, Z, 3),
    given that the original X, Y, Z, are all the same size. Being forced to have everything the same size is what
    makes this difficult, because this dramatically complicates things for a multi-dimensional symbol array.

    For a symbol array of shape (a, b, c, d) and orbit field with shape (N, X, Y, Z, 3) the final dimensions
    would be (a*N, b*X, c*Y, d*Z, 3). This is achieved by repeated concatenation along the
    axis corresponding to the last axis of the symbol array. i.e. for (a,b,c,d) this would be concatenation along
    axis=3, 4 times in a row. I believe that this generalizes for all equations but it has not been tested yet.

    """
    glue_shape = orbit_array.shape
    tile_shape = orbit_array.ravel()[0].shapes()[0]
    gluing_order = kwargs.get("gluing_order", np.argsort(glue_shape))
    conserve_parity = kwargs.get("conserve_parity", True)
    nzero = kwargs.get("exclude_nonpositive", True)
    gluing_basis = kwargs.get("basis", orbit_type.bases_labels()[0])
    # This joins the dictionary of all orbits' dimensions by zipping the values together. i.e.
    # (T_1, L_1, ...), (T_2, L_2, ...) transforms into  ((T_1, T_2, ...) , (L_1, L_2, ...))

    if strip_wise:
        glued_orbit = None
        for gluing_axis in gluing_order:
            # Produce the slices that will select the strips of orbits so we can iterate and work with strips.
            gluing_slices = itertools.product(
                *(
                    range(g) if i != gluing_axis else [slice(None)]
                    for i, g in enumerate(glue_shape)
                )
            )
            # Each strip will have a resulting glued orbit, collect these via appending to list.
            glued_orbit_strips = []
            for gs in gluing_slices:
                # The strip shape is 1-d but need a d-dimensional tuple filled with 1's to keep track of axes.
                # i.e. (3,1,1,1) instead of just (3,).
                strip_shape = tuple(
                    len(orbit_array[gs]) if n == gluing_axis else 1
                    for n in range(len(orbit_array.shape))
                )

                # For each strip, need to know how to combine the dimensions of the orbits. Bundle, then combine.
                tuple_of_zipped_dimensions = tuple(
                    zip(*(o.dimensions() for o in orbit_array[gs].ravel()))
                )
                strip_parameters = orbit_type.glue_dimensions(
                    tuple_of_zipped_dimensions,
                    glue_shape=strip_shape,
                    exclude_nonpositive=nzero,
                )
                # Slice the orbit array to get the strip, reshape to maintain its d-dimensional form.
                strip_of_orbits = orbit_array[gs].ravel()

                # Correct the proportions of the dimensions along the current gluing axis.
                orbit_array_corrected = aspect_ratio_correction(
                    strip_of_orbits, axis=gluing_axis, conserve_parity=conserve_parity
                )
                # Concatenate the states with corrected proportions.
                glued_strip_state = np.concatenate(
                    tuple(x.state for x in orbit_array_corrected), axis=gluing_axis
                )
                # Put the glued strip's state back into a class instance.
                glued_strip_orbit = orbit_type(
                    state=glued_strip_state,
                    basis=gluing_basis,
                    parameters=strip_parameters,
                    **kwargs
                )

                # Take the result and store it for futher gluings.
                glued_orbit_strips.append(glued_strip_orbit)
            # We combined along the gluing axis meaning that that axis has a new shape of 1. For symbol arrays
            # with more dimensions, we need to repeat the gluing along each axis, update with the newly glued strips.
            glue_shape = tuple(
                glue_shape[i] if i != gluing_axis else 1 for i in range(len(glue_shape))
            )
            orbit_array = np.array(glued_orbit_strips).reshape(glue_shape)
            if orbit_array.size == 1:
                glued_orbit = orbit_array.ravel()[0]
        if glued_orbit is None:
            raise ValueError(
                "strip-wise gluing did not produce a result; verify gluing_shape and gluing_order "
                "are not empty or disable the strip-wise correction."
            )
    else:
        # Bundle all of the parameters at once, instead of "stripwise"
        zipped_dimensions = tuple(zip(*(o.dimensions() for o in orbit_array.ravel())))
        # Default parameter gluing strategy is to average all tile dimensions
        glued_dimensions = orbit_type.glue_dimensions(
            zipped_dimensions, glue_shape=glue_shape, exclude_nonpositive=nzero
        )
        # If we want a much simpler method of gluing, we can do "arraywise" which simply concatenates everything at
        # once. I would say this is the better option if all orbits in the tile dictionary are approximately equal
        # in size.
        # the "gluing axis" in this case is always the same, we are just iterating through the gluing shape one
        # axis at a time.
        gluing_axis = len(glue_shape) - 1

        # arrange the orbit states into an array of the same shape as the symbol array.
        orbit_field_list = np.array(
            [
                o.transform(to=orbit_type.bases_labels()[0]).state
                for o in orbit_array.ravel()
            ]
        )
        glued_orbit_state = np.array(orbit_field_list).reshape(
            *orbit_array.shape, *tile_shape
        )
        # iterate through and combine all of the axes.
        while len(glued_orbit_state.shape) > len(tile_shape):
            glued_orbit_state = np.concatenate(glued_orbit_state, axis=gluing_axis)

        glued_orbit = orbit_type(
            state=glued_orbit_state,
            basis=gluing_basis,
            parameters=glued_dimensions,
            **kwargs
        )
    return glued_orbit


def tile(symbol_array, tiling_dictionary, orbit_type, **kwargs):
    """
    Wraps the glue function so that configurations of symbols may be provided instead of configurations of Orbits.

    Parameters
    ----------
    symbol_array : numpy.ndarray
        An array of dictionary keys which exist in tiling_dictionary
    tiling_dictionary : dict
        A dictionary whose values are Orbit instances.
    orbit_type : type
        The type of Orbit that will be returned.
    kwargs :
        Orbit kwargs relevant to instantiation and gluing. See :func:`glue` for details.

    Returns
    -------
    Orbit : `orbit_type`
        An instance containing the glued state

    """
    symbol_array_shape = symbol_array.shape
    orbit_array = np.array(
        [tiling_dictionary[symbol] for symbol in symbol_array.ravel()]
    ).reshape(*symbol_array_shape)
    glued_orbit = glue(orbit_array, orbit_type, **kwargs)
    return glued_orbit


def generate_symbol_arrays(tiling_dictionary, glue_shape, unique=True):
    """
    Produce all d-dimensional symbol arrays for a given dictionary and shape.

    Parameters
    ----------
    tiling_dictionary : dict
        Dictionary whose keys are the orbit symbols and whose values are Orbits
    glue_shape : tuple
        The shape of the gluing configuration (i.e. symbol array)
    unique : bool
        If True, then rotations of symbol arrays are treated as redundant.

    Returns
    -------
    list of ndarray :
        A list of numpy arrays containing configurations of symbols (`tiling_dictionary` keys).

    Notes
    -----
    If unique = False then this produces a list of d^N elements, d being the dimension and N being the number
    of symbols in the dictionary. Clearly this can be a huge drain on resources in certain cases. If possible,
    it is better to take the group orbit of the glued orbits resulting from a symbol array than all possible symbol
    arrays.

    """
    symbol_array_generator = itertools.product(
        list(tiling_dictionary.keys()), repeat=np.product(glue_shape)
    )
    if unique:
        axes = tuple(range(len(glue_shape)))
        cumulative_equivariants = []
        unique_symbol_arrays = []
        for symbol_combination in symbol_array_generator:
            for rotation in itertools.product(*(list(range(a)) for a in glue_shape)):
                equivariant_combination = to_symbol_string(
                    np.roll(
                        np.reshape(symbol_combination, glue_shape), rotation, axis=axes
                    )
                )
                if equivariant_combination in cumulative_equivariants:
                    break
                else:
                    cumulative_equivariants.append(equivariant_combination)
            else:
                unique_symbol_arrays.append(np.reshape(symbol_combination, glue_shape))
        return unique_symbol_arrays
    else:
        return [np.reshape(x, glue_shape) for x in symbol_array_generator]


def rediscretize_tileset(tiling_dictionary, new_shape=None, **kwargs):
    """
    Convenience tool for resizing all orbits in a tiling dictionary in a single function call

    Parameters
    ----------
    tiling_dictionary : dict
        Keys are symbol alphabet, values are Orbits.
    new_shape : tuple, optional, default None
        If provided as a shape tuple then all orbits in the tiling dictionary will be
        resized to this shape.
    kwargs : dict
        Keyword arguments for :meth:`Orbit.dimension_based_discretization` method
        
    Returns
    -------
    dict :
        Tiling dictionary whose values (Orbits) have been resized.

    """
    orbits = list(tiling_dictionary.values())
    if new_shape is None:
        # If the user is really lazy this will make the dictionary uniform by
        # changing the discretization sizes based on averages (and class type).
        average_dimensions = tuple(
            np.mean(x) for x in tuple(zip(*(o.dimensions() for o in orbits)))
        )
        # The new shape will be inferred from the most common class in the tiling dictionary,
        # as to produce the "best" approximation for the provided dictionary.
        most_common_orbit_class = max(Counter([o.__class__ for o in orbits]))
        new_shape = most_common_orbit_class.dimension_based_discretization(
            average_dimensions, **kwargs
        )

    return {
        td_key: td_val.resize(*new_shape)
        for td_key, td_val in tiling_dictionary.items()
    }


def pairwise_group_orbit(orbit_pair, **kwargs):
    """
    Generate all pairs of elements from two group orbits.

    Parameters
    ----------
    orbit_pair : array-like
        Two orbits whose group orbit product is to be computed; all possible pairs of group
        orbit members.
    kwargs :
        keyword arguments relevant for group orbit generators; specific to symmetries of governing equation.

    Yields
    ------
    tuple :
        Pairs of Orbits 
    
    Notes
    -----
    Group orbits are the sets of Orbits resulting from applying a group of equivariant symmetry operations to an Orbit.

    """
    orbit_pair = np.array(orbit_pair)
    yield from itertools.product(
        orbit_pair.ravel()[0].group_orbit(**kwargs),
        orbit_pair.ravel()[1].group_orbit(**kwargs),
    )


def expensive_pairwise_glue(orbit_pair, objective="cost", axis=0, **kwargs):
    """
    Gluing that searches pairs of group orbit members for the best combination.
    
    Parameters
    ----------
    orbit_pair : np.ndarray
        An array with the same number of dimensions as the orbits within them; i.e. (2, 1, 1, 1), (1, 2, 1, 1), ...
        for Orbits with 4 dimensions.
    orbit_type : type
        The Orbit type to return the result as.
    objective : str
        The manner that orbit combinations are graded; options are 'cost' or 'boundary_cost'.
        The former calls the Orbit's built in cost function, the latter computes the
        L2 difference of the boundaries that are joined together.

    Returns
    -------
    best_glued_orbit_so_far : Orbit
        The orbit combination in the pairwise group orbit that had the smallest objective function value.
    
    Notes
    -----
    Expensive gluing only supported for pairs of Orbits. For larger arrays of orbits, apply this function in
    an iterative manner if so desired.

    This function can not only be expensive, it can be VERY expensive depending on the generator x.group_orbit().
    It is highly advised to have some way of controlling how many members of each group orbit are being used.
    For example, for the K-S equation and its translational symmetries, the field arrays are typically being rolled; the
    option exists to roll by an integer "stride" value s.t. instead of shifting by one unit, a number of units
    equal to stride is shifted. Note that for group orbits of size 1000, this would construct and search over 1000000
    combinations, unless a subset of the group orbit is specified.

    """
    # Aspect ratio correction prior to gluing means that it does not have to be done for each combination.
    orbit_return_type = kwargs.get("return_type", type(np.array(orbit_pair).ravel()[0]))
    smallest_cost_so_far = np.inf
    best_glued_orbit_so_far = None
    # iterate over all combinations of group orbit members; keyword arguments can be passed to control sampling rate.
    for ga, gb in pairwise_group_orbit(orbit_pair, **kwargs):
        if objective == "boundary_cost":
            # Way of slicing the state arrays at the boundaries. This is assuming periodic boundary conditions.
            # This method doesn't work very well for fundamental domains.
            aslice = tuple(
                (0, -1) if i == axis and periodic else -1 if i == axis else slice(None)
                for i, periodic in enumerate(ga.periodic_dimension())
            )
            bslice = tuple(
                (-1, 0) if i == axis and periodic else 0 if i == axis else slice(None)
                for i, periodic in enumerate(ga.periodic_dimension())
            )
            # Slice the state and not the orbit, as slices may not be valid (do not retain number of dimensions)
            current_cost = np.linalg.norm(ga.state[aslice] - gb.state[bslice])
            if current_cost < smallest_cost_so_far:
                best_glued_orbit_so_far = ga.concat(gb, axis=axis)
                if kwargs.get("fundamental_domain", False):
                    best_glued_orbit_so_far = (
                        best_glued_orbit_so_far.from_fundamental_domain()
                    )
                smallest_cost_so_far = current_cost
                best_glued_orbit_so_far = orbit_return_type(
                    **vars(best_glued_orbit_so_far)
                )
        else:
            glued_orbit_attempt = ga.concat(gb, axis=axis)
            if kwargs.get("fundamental_domain", False):
                glued_orbit_attempt = glued_orbit_attempt.from_fundamental_domain()
            glued_orbit_attempt = orbit_return_type(**vars(glued_orbit_attempt))
            current_cost = glued_orbit_attempt.cost()
            if current_cost < smallest_cost_so_far:
                smallest_cost_so_far = current_cost
                best_glued_orbit_so_far = glued_orbit_attempt
    return best_glued_orbit_so_far
