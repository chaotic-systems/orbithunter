from .io import to_symbol_string
import numpy as np
import itertools
from collections import Counter

__all__ = ['tile', 'glue', 'generate_symbol_arrays', 'rediscretize_tileset']


def _correct_aspect_ratios(orbit_array, axis=0, conserve_parity=True):
    """ Correct aspect ratios of a one-dimensional strip of orbits.

    Parameters
    ----------
    orbit_array
    axis

    Returns
    -------

    Notes
    -----
    Note that this *will* allow equilibria tiles to become very distorted; this can be managed by gluing
    order but typically when including equilibria, strip-wise corrections cause too much distortion to be useful.

    """
    dims = np.array([o.dimensions()[axis] for o in orbit_array])
    sizes = np.array([o.shapes()[0][axis] for o in orbit_array])

    disc_total = np.sum(sizes)
    dim_total = np.sum(dims)

    # The absolute minimum is set to the smallest number of points which doesn't completely "contract" the dimension.
    # Whether or not this takes an even (2) or odd (3) value is inferred from the total number of discrete points
    # along the given axis for the array of orbits provided. To avoid orbits with small dimension being
    # undersized, use some minimum fraction based upon the strip length.

    # If dimensions are all zero (dimensions taken to be non-negative), then no resizing can be inferred.
    if dim_total == 0.:
        return orbit_array

    new_discretization_sizes = np.zeros(len(orbit_array))

    if conserve_parity:
        # the values returned by minimal shape are the absolute minimum. If the parity is wrong, then the practical
        # minimum must be one greater.
        min_disc_sizes = []
        for i, min_size in enumerate(o.minimal_shape()[axis] for o in orbit_array):
            if (min_size % 2) != (sizes[i] % 2):
                min_disc_sizes.append(min_size+1)
            else:
                min_disc_sizes.append(min_size)
        min_disc_sizes = np.array(min_disc_sizes)
        # Once the minimum discretization sizes are stored with the correct parity, then to maintain the parity
        # we need to add even numbers only; can accomplish this by using units of 2.
        disc_remainder = disc_total - np.sum(min_disc_sizes)
        disc_remainder //= 2
        dim_remainder = dim_total
        for sorted_index in np.argsort(dims):
            orbits_share = int(disc_remainder * (dims[sorted_index] / dim_remainder))
            new_discretization_sizes[sorted_index] = min_disc_sizes[sorted_index] + (2 * orbits_share)
            dim_remainder -= dims[sorted_index]
            disc_remainder -= orbits_share
    else:
        min_disc_sizes = np.array([o.minimal_shape()[axis] for o in orbit_array])
        disc_remainder = disc_total - np.sum(min_disc_sizes)
        dim_remainder = dim_total

        for sorted_index in np.argsort(dims):
            orbits_share = int(disc_remainder * (dims[sorted_index] / dim_remainder))
            new_discretization_sizes[sorted_index] = min_disc_sizes[sorted_index] + orbits_share
            dim_remainder -= dims[sorted_index]
            disc_remainder -= orbits_share
    # from smallest to largest, increase the share of the discretization
    new_shapes = [tuple(int(new_discretization_sizes[j]) if i == axis else o.shapes()[0][i]
                  for i in range(len(o.shape))) for j, o in enumerate(orbit_array)]

    # Return the strip of orbits with corrected proportions.
    return np.array([o.resize(shp) for o, shp in zip(orbit_array, new_shapes)])


def glue(orbit_array, class_constructor, strip_wise=False, **kwargs):
    """ Function for combining spatiotemporal fields

    Parameters
    ----------
    orbit_array : ndarray of Orbit instances
        A NumPy array wherein each element is an orbit. i.e. a tensor of Orbit instances. The shape should be
        representative to how the orbits are going to be glued. See notes for more details. The orbits must all
        have the same discretization size if gluing is occuring along more than one axis. The orbits should
        all be in the physical field basis.
    class_constructor : Orbit class
        i.e. OrbitKS without parentheses
    strip_wise : bool
        If True, then "strip-wise aspect ratio correction" is applied.
    Returns
    -------
    glued_orbit : Orbit instance
        Instance of type class_constructor

    Notes
    -----
    Assumes that each orbit in the array has identical dimensions in all other axes other than the one specified.
    Assumes no symmetries of the different orbits in the array. There are too many complications if handled otherwise.

    Because of how the concatenation of fields works, wherein the discretization must match along the boundaries. It
    is quite complicated to write a generalized code that glues all dimensions together at once, so instead this is
    designed to iterate through the axes of the orbit_array.

    To prevent confusion, there are many different "shapes" and discretizations that are possibly floating around.
    There are three main array shapes or dimensions that are involved in this function. The first is the array
    of orbits, which represents a spatiotemporal symbolic "dynamics" block. This array can have as many dimensions
    as the solutions to the equation have. An array of orbits of shape (2,1,1,1) means that the fields have four
    continuous dimensions; they need not be scalar fields either. The specific shape means that two such fields
    are being concatenated in time (because the first axis should always be time).

    Example for the spatiotemporal Navier-stokes equation. The spacetime is (1+3) dimensional. Let's assume we're gluing
    two vector fields with the same discretization size, (N, X, Y, Z). We can think of this discretization as a
    collection of 3D vector field snapshots in time. Therefore, there are actually (N, X, Y, Z, 3) degrees of freedom.
    Therefore
    the actually tensor before the gluing will be of the shape (2, 1, 1, 1, N, X, Y, Z, 3). Because we are gluing the
    orbits along the time axis, The final shape will be (1, 1, 1, 1, 2*N, X, Y, Z, 3), given that the original
    X, Y, Z, are all the same size. Being forced to have everything the same size is what makes this difficult, because
    this dramatically complicates things for a multi-dimensional symbol array.

    It's so complicated that for gluing along more than one axis its only really viable to start with tiles with the
    same shape.

    For a symbol array of shape (a, b, c, d) and orbit field with shape (N, X, Y, Z, 3) the final dimensions
    would be (a*N, b*X, c*Y, d*Z, 3). I believe that this can be achieved by repeated concatenation along the
    axis corresponding to the last axis of the symbol array. i.e. for (a,b,c,d) this would be concatenation along
    axis=3, 4 times in a row. I believe that this generalizes for all equations but it's hard to test

    """
    glue_shape = orbit_array.shape
    tiling_shape = orbit_array.ravel()[0].shapes()[0]
    gluing_order = kwargs.get('gluing_order', np.argsort(glue_shape))
    # This joins the dictionary of all orbits' dimensions by zipping the values together. i.e.
    #(T_1, L_1, ...), (T_2, L_2, ...) transforms into  ((T_1, T_2, ...) , (L_1, L_2, ...))

    # Bundle all of the parameters at once, instead of "stripwise"
    zipped_dimensions = tuple(zip(*(o.dimensions() for o in orbit_array.ravel())))
    # Average tile dimensions
    glued_parameters = class_constructor.glue_parameters(zipped_dimensions, glue_shape=glue_shape)

    if strip_wise:
        for gluing_axis in gluing_order:
            # Produce the slices that will select the strips of orbits so we can iterate and work with strips.
            gluing_slices = itertools.product(*(range(g) if i != gluing_axis else [slice(None)]
                                                for i, g in enumerate(glue_shape)))
            # Each strip will have a resulting glued orbit, collect these via appending to list.
            glued_orbit_strips = []
            for gs in gluing_slices:
                # The strip shape is 1-d but need a d-dimensional tuple filled with 1's to keep track of axes.
                # i.e. (3,1,1,1) instead of just (3,).
                strip_shape = tuple(len(orbit_array[gs]) if n == gluing_axis else 1
                                    for n in range(len(orbit_array.shape)))

                # For each strip, need to know how to combine the dimensions of the orbits. Bundle, then combine.
                tuple_of_zipped_dimensions = tuple(zip(*(o.dimensions() for o in orbit_array[gs].ravel())))
                strip_parameters = class_constructor.glue_parameters(tuple_of_zipped_dimensions, glue_shape=strip_shape)
                # Slice the orbit array to get the strip, reshape to maintain its d-dimensional form.
                strip_of_orbits = orbit_array[gs].ravel()

                # Correct the proportions of the dimensions along the current gluing axis.
                orbit_array_corrected = _correct_aspect_ratios(strip_of_orbits, strip_parameters, glued_parameters,
                                                               axis=gluing_axis)
                # Concatenate the states with corrected proportions.
                glued_strip_state = np.concatenate(tuple(x.state for x in orbit_array_corrected),
                                                   axis=gluing_axis)
                # Put the glued strip's state back into a class instance.
                glued_strip_orbit = class_constructor(state=glued_strip_state, basis=class_constructor.bases()[0],
                                                      parameters=strip_parameters, **kwargs)

                # Take the result and store it for futher gluings.
                glued_orbit_strips.append(glued_strip_orbit)
            # We combined along the gluing axis meaning that that axis has a new shape of 1. For symbol arrays
            # with more dimensions, we need to repeat the gluing along each axis, update with the newly glued strips.
            glue_shape = tuple(glue_shape[i] if i != gluing_axis else 1 for i in range(len(glue_shape)))
            orbit_array = np.array(glued_orbit_strips).reshape(glue_shape)
            if orbit_array.size == 1:
                glued_orbit = orbit_array.ravel()[0]
    else:
        # If we want a much simpler method of gluing, we can do "arraywise" which simply concatenates everything at
        # once. I would say this is the better option if all orbits in the tile dictionary are approximately equal
        # in size.
        # the "gluing axis" in this case is always the same, we are just iterating through the gluing shape one
        # axis at a time.
        gluing_axis = len(glue_shape) - 1

        # arrange the orbit states into an array of the same shape as the symbol array.
        orbit_field_list = np.array([o.transform(to=class_constructor.bases()[0]).state for o in orbit_array.ravel()])
        glued_orbit_state = np.array(orbit_field_list).reshape(*orbit_array.shape, *tiling_shape)
        # iterate through and combine all of the axes.
        while len(glued_orbit_state.shape) > len(tiling_shape):
            glued_orbit_state = np.concatenate(glued_orbit_state, axis=gluing_axis)

        glued_orbit = class_constructor(state=glued_orbit_state, basis=class_constructor.bases()[0],
                                        parameters=glued_parameters, **kwargs)

    return glued_orbit


def tile(symbol_array, tiling_dictionary, class_constructor,  **kwargs):
    """
    Parameters
    ----------
    symbol_array : ndarray
        An array of dictionary keys which exist in tiling_dictionary
    tiling_dictionary : dict
        A dictionary whose values are Orbit instances.
    class_constructor : Orbit generator
        i.e. 'Orbit' w/o parenthesis.
    tile_shape : tuple
        Tuple containing the field discretization to be used in tiling
    kwargs :
        Orbit kwargs relevant to instantiation
    Returns
    -------

    Notes
    -----
    This is simply a wrapper for gluing that allows the user to submit symbol arrays and dictionaries instead
    of orbit arrays.

    """
    symbol_array_shape = symbol_array.shape
    orbit_array = np.array([tiling_dictionary[symbol] for symbol in symbol_array.ravel()]
                                        ).reshape(*symbol_array_shape)
    glued_orbit = glue(orbit_array, class_constructor, **kwargs)
    return glued_orbit


def generate_symbol_arrays(fpo_dictionary, glue_shape, unique=True):
    """ Produce all d-dimensional symbol arrays for a given dictionary and shape.

    Parameters
    ----------
    fpo_dictionary
    glue_shape
    unique

    Returns
    -------

    Notes
    -----
    If unique = False then this produces a list of d^N elements, d being the dimension and N being the number
    of symbols in the dictionary.

    """
    symbol_array_generator = itertools.product(list(fpo_dictionary.keys()), repeat=np.product(glue_shape))
    if unique:
        axes = tuple(range(len(glue_shape)))
        cumulative_equivariants = []
        unique_symbol_arrays = []
        for symbol_combination in symbol_array_generator:
            for rotation in itertools.product(*(list(range(a)) for a in glue_shape)):
                equivariant_combination = to_symbol_string(np.roll(np.reshape(symbol_combination, glue_shape),
                                                                   rotation, axis=axes))
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
    orbits = list(tiling_dictionary.values())
    if new_shape is None:
        # If the user is really lazy this will make the dictionary uniform by
        # changing the discretization sizes based on averages (and class type).
        average_dimensions = tuple(np.mean(x) for x in tuple(zip(*(o.dimensions() for o in orbits))))
        # The new shape will be inferred from the most common class in the tiling dictionary,
        # as to produce the "best" approximation for the provided dictionary.
        most_common_orbit_class = max(Counter([o.__class__ for o in orbits]))
        new_shape = most_common_orbit_class.parameter_based_discretization(average_dimensions, **kwargs)

    return {td_key: td_val.resize(*new_shape) for td_key, td_val in tiling_dictionary.items()}


def pairwise_group_orbit(orbit_pair, **kwargs):
    """ Generate all pairwise elements from two group orbits. I.e. all symmetry combinations.

    Notes
    -----
    Keeping this as a generator saves a *lot* of time, hence the reason for the function instead of just
    instantiating the itertools product.
    """
    yield from itertools.product(orbit_pair.ravel()[0].group_orbit(**kwargs),
                                 orbit_pair.ravel()[1].group_orbit(**kwargs))


def expensive_glue(orbit_pair_array, class_constructor, method='residual', **kwargs):
    """ Gluing that searches group orbit for the best gluing.

    Notes
    -----
    This can not only be expensive, it can be VERY expensive depending on the generator x.group_orbit().
    It is highly advised to have some way of controlling how many members of each group orbit are being used.
    For example, for the K-S equation and its translational symmetries, the field arrays are being rolled; the
    option exists to roll by an integer "stride" value s.t. instead of shifting by one unit, a number of units
    equal to stride is shifted.

    With regards to discrete symmetries
    """
    # Aspect ratio correction prior to gluing means that it does not have to be done for each combination.
    gluing_axis = int(np.argmax(orbit_pair_array.shape))
    # The best orbit pair at the start is by default the first one.
    corrected_orbit_pair = _correct_aspect_ratios(orbit_pair_array, axis=gluing_axis)
    best_glued_orbit = glue(corrected_orbit_pair, class_constructor, **kwargs)
    smallest_residual_so_far = best_glued_orbit.residual()
    # iterate over all combinations of group orbit members; keyword arguments can be passed to control sampling rate.
    for ga, gb in pairwise_group_orbit(corrected_orbit_pair, **kwargs):
        if method == 'boundary_residual':
            # Ugly way of slicing the state arrays at the boundaries. This is assuming periodic boundary conditions.
            aslice = tuple((0, -1) if i == gluing_axis else slice(None) for i in range(corrected_orbit_pair.ndim))
            bslice = tuple((-1, 0) if i == gluing_axis else slice(None) for i in range(corrected_orbit_pair.ndim))
            boundary_residual = np.linalg.norm(ga.state[aslice]-gb.state[bslice])
            if boundary_residual < smallest_residual_so_far:
                best_glued_orbit = glue(np.array([ga, gb]).reshape(orbit_pair_array.shape), class_constructor, **kwargs)
                smallest_residual_so_far = boundary_residual
        else:
            g_orbit_array = np.array([ga, gb]).reshape(orbit_pair_array.shape)
            best_glued_orbit = glue(g_orbit_array, class_constructor, **kwargs)
            residual = best_glued_orbit.residual()
            if residual < smallest_residual_so_far:
                smallest_residual_so_far = residual

    return best_glued_orbit
