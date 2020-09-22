from .discretization import rediscretize, correct_aspect_ratios
from .io import read_h5
import numpy as np
import os
import itertools

__all__ = ['tile', 'glue', 'generate_symbol_arrays']


def best_combination(orbit, other_orbit, fundamental_domain_combinations, axis=0):
    half_combinations = list(itertools.product(half_list,repeat=2))
    residual_list = []
    glued_list = []
    for halves in half_combinations:
        orbit_domain = orbit._to_fundamental_domain(half=halves[0])
        other_orbit_domain = other_orbit._to_fundamental_domain(half=halves[1])
        glued = concat(orbit_domain, other_orbit_domain, direction=direction)
        glued_orbit = glued._from_fundamental_domain()
        glued_list.extend([glued_orbit])
        residual_list.extend([glued_orbit.residual])
    best_combi = np.array(glued_list)[np.argmin(residual_list)]
    return best_combi


def best_rotation(orbit, other_orbit, axis=0):
    field_orbit, field_other_orbit = correct_aspect_ratios(orbit, other_orbit, axis=axis)

    resmat = np.zeros([field_other_orbit.N, field_other_orbit.M])
    # The orbit only remains a converged solution if the rotations occur in
    # increments of the discretization, i.e. multiples of L / M and T / N.
    # The reason for this is because those are the only values that do not
    # actually change the field via interpolation. In other words,
    # The rotations must coincide with the collocation points.
    # for n in range(0, field_other_orbit.N):
    #     for m in range(0, field_other_orbit.M):
    #         rotated_state = np.roll(np.roll(field_other_orbit.state, m, axis=1), n, axis=0)
    #         rotated_orbit = other_orbit.__class__(state=rotated_state, state_type=field_other_orbit.state_type, T=other_orbit.T,
    #                                               L=other_orbit.L, S=other_orbit.S)
    #         resmat[n,m] = concat(field_orbit, rotated_orbit, direction=direction).residual


    bestn, bestm = np.unravel_index(np.argmin(resmat), resmat.shape)
    high_resolution_orbit = rediscretize(field_orbit, new_N=16*field_orbit.N, new_M=16*field_orbit.M)
    high_resolution_other_orbit = rediscretize(field_other_orbit, new_N=16*field_other_orbit.N, new_M=16*field_other_orbit.M)

    best_rotation_state = np.roll(np.roll(high_resolution_other_orbit.state, 16*bestm, axis=1), 16*bestn, axis=0)
    highres_rotation_orbit = other_orbit.__class__(state=best_rotation_state, state_type='field',
                                                   T=other_orbit.T, L=other_orbit.L, S=other_orbit.S)
    best_gluing = concat(high_resolution_orbit, highres_rotation_orbit, direction=direction)
    return best_gluing


def expensive_glue(pair_of_orbits_array, class_constructor, gluing_axis=0):
    """

    Parameters
    ----------
    orbit
    other_orbit
    axis

    Returns
    -------

    Notes
    -----
    This function gives the user a lot more control over the result of gluing when considering only the
    combination of two orbits. This should be used when the optimal gluing is desired for a pair of orbits.
    """
    # Converts tori to best representatives for gluing by choosing from group orbit.
    # If we want a much simpler method of gluing, we can do "arraywise" which simply concatenates everything at
    # once. I would say this is the better option if all orbits in the tile dictionary are approximately equal
    # in size.
    # the "gluing axis" in this case is always the same, we are just iterating through the gluing shape one
    # axis at a time.
    glue_shape = tuple(2 if i == gluing_axis else 1 for i in range(2))
    corrected_pair_of_orbits = correct_aspect_ratios(pair_of_orbits_array, gluing_axis=0)
    # Bundle all of the parameters at once, instead of "stripwise"
    zipped_dimensions = tuple(zip(*(o.dimensions for o in pair_of_orbits_array.ravel())))
    glued_parameters = class_constructor.glue_parameters(zipped_dimensions, glue_shape=glue_shape)

    # arrange the orbit states into an array of the same shape as the symbol array.
    orbit_field_list = [o.convert(to='field').state for o in pair_of_orbits_array.ravel()]
    glued_orbit_state = np.array(orbit_field_list).reshape(*pair_of_orbits_array.shape, *tiling_shape)
    # iterate through and combine all of the axes.
    while len(glued_orbit_state.shape) > len(tiling_shape):
        glued_orbit_state = np.concatenate(glued_orbit_state, axis=gluing_axis)

    glued_orbit = class_constructor(state=glued_orbit_state, state_type='field',
                                    orbit_parameters=glued_parameters)


def tile_dictionary_ks(padded=False, comoving=False):
    if padded:
        directory = '../data/tiles/padded/'
    else:
        directory = '../data/tiles/'

    if comoving:
        # padded merger Orbit in comoving frame
        merger = read_h5('./OrbitKS_merger.h5', directory=directory, state_type='field')
    else:
        # padded merger orbit in physical frame.
        merger = read_h5('./OrbitKS_merger_fdomain.h5', directory=directory, state_type='field')

    # padded streak orbit
    streak = read_h5('./OrbitKS_streak.h5',directory=directory, state_type='field')

    # padded wiggle orbit
    wiggle = read_h5('./OrbitKS_wiggle.h5', directory=directory, state_type='field')

    tile_dict = {0: streak, 1: merger, 2: wiggle}
    return tile_dict


def glue(array_of_orbit_instances, class_constructor, stripwise=False, **kwargs):
    """ Function for combining spatiotemporal fields

    Parameters
    ----------
    array_of_orbit_instances : ndarray of Orbit instances
        A NumPy array wherein each element is an orbit. i.e. a tensor of Orbit instances. The shape should be
        representative to how the orbits are going to be glued. See notes for more details. The orbits must all
        have the same discretization size if gluing is occuring along more than one axis. The orbits should
        all be in the physical field basis.
    class_constructor : Orbit class
        i.e. OrbitKS without parentheses

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
    designed to iterate through the axes of the array_of_orbit_instances.

    To prevent confusion, there are many different "shapes" and discretizations that are possibly floating around.
    There are three main array shapes or dimensions that are involved in this function. The first is the array
    of orbits, which represents a spatiotemporal symbolic "dynamics" block. This array can have as many dimensions
    as the solutions to the equation have. An array of orbits of shape (2,1,1,1) means that the fields have four
    continuous dimensions; they need not be scalar fields either. The specific shape means that two such fields
    are being concatenated in time (because the first axis should always be time).

    Example for the spatiotemporal Navier-stokes equation. The spacetime is (1+3) dimensional. Let's assume we're gluing
    two vector fields with the same discretization size, (N, X, Y, Z). We can think of this discretization as a collection
    of 3D vector field snapshots in time. Therefore, there are actually (N, X, Y, Z, 3) degrees of freedom. Therefore
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
    glue_shape = array_of_orbit_instances.shape
    tiling_shape = array_of_orbit_instances.ravel()[0].field_shape
    gluing_order = kwargs.get('gluing_order', np.argsort(glue_shape))
    # This joins the dictionary of all orbits' dimensions by zipping the values together. i.e.
    # {'T': T_1, 'L': L_1}, {'T': T_2, 'L': L_2}, .....  transforms into  {'T': (T_1, T_2, ...) , 'L': (L_1, L_2, ...)}

    if stripwise:
        for gluing_axis in gluing_order:
            # Produce the slices that will select the strips of orbits so we can iterate and work with strips.
            gluing_slices = list(itertools.product(*(range(g) if i != gluing_axis else [slice(None)]
                                                     for i, g in enumerate(glue_shape))))
            # Each strip will have a resulting glued orbit, collect these via appending to list.
            glued_orbit_strips = []
            for gs in gluing_slices:

                # The strip shape is 1-d but need a d-dimensional tuple filled with 1's to keep track of axes.
                strip_shape = tuple(len(array_of_orbit_instances[gs]) if n == gluing_axis else 1
                                    for n in range(len(array_of_orbit_instances.shape)))

                # For each strip, need to know how to combine the dimensions of the orbits. Bundle, then combine.
                zipped_dimensions = tuple(zip(*(o.dimensions for o in array_of_orbit_instances[gs].ravel())))
                glued_parameters = class_constructor.glue_parameters(zipped_dimensions, glue_shape=strip_shape)
                # Slice the orbit array to get the strip, reshape to maintain its d-dimensional form.
                strip_of_orbits = array_of_orbit_instances[gs].reshape(strip_shape)

                # Correct the proportions of the dimensions along the current gluing axis.
                array_of_orbit_instances_corrected = correct_aspect_ratios(strip_of_orbits, gluing_axis=gluing_axis)
                # Concatenate the states with corrected proportions.
                glued_strip_state = np.concatenate(tuple(x.state for x in array_of_orbit_instances_corrected),
                                                   axis=gluing_axis)
                # Put the glued strip's state back into a class instance.
                glued_strip_orbit = class_constructor(state=glued_strip_state, state_type='field',
                                                      orbit_parameters=glued_parameters)
                # Take the result and store it for futher gluings.
                glued_orbit_strips.append(glued_strip_orbit)
            # We combined along the gluing axis meaning that that axis has a new shape of 1. For symbol arrays
            # with more dimensions, we need to repeat the gluing along each axis, update with the newly glued strips.
            glue_shape = tuple(glue_shape[i] if i != gluing_axis else 1 for i in range(len(glue_shape)))
            array_of_orbit_instances = np.array(glued_orbit_strips).reshape(glue_shape)
            if array_of_orbit_instances.size == 1:
                glued_orbit = array_of_orbit_instances.ravel()[0]
    else:
        # If we want a much simpler method of gluing, we can do "arraywise" which simply concatenates everything at
        # once. I would say this is the better option if all orbits in the tile dictionary are approximately equal
        # in size.
        # the "gluing axis" in this case is always the same, we are just iterating through the gluing shape one
        # axis at a time.
        gluing_axis = len(glue_shape) - 1
        # Bundle all of the parameters at once, instead of "stripwise"
        zipped_dimensions = tuple(zip(*(o.dimensions for o in array_of_orbit_instances.ravel())))
        glued_parameters = class_constructor.glue_parameters(zipped_dimensions, glue_shape=glue_shape)

        # arrange the orbit states into an array of the same shape as the symbol array.
        orbit_field_list = [o.convert(to='field').state for o in array_of_orbit_instances.ravel()]
        glued_orbit_state = np.array(orbit_field_list).reshape(*array_of_orbit_instances.shape, *tiling_shape)
        # iterate through and combine all of the axes.
        while len(glued_orbit_state.shape) > len(tiling_shape):
            glued_orbit_state = np.concatenate(glued_orbit_state, axis=gluing_axis)

        glued_orbit = class_constructor(state=glued_orbit_state, state_type='field',
                                        orbit_parameters=glued_parameters)

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
        i.e. Orbit w/o parenthesis.
    tile_shape : tuple
        Tuple containing the field discretization to be used in tiling
    kwargs :
        Orbit kwargs relevant to instantiation
    Returns
    -------

    Notes
    -----
    This is simply a wrapper for gluing that allows the user to submit symbol arrays and dictionaries instead
    of orbit arrays. It also reshapes all orbits in the dictionary to a uniform size.

    """
    symbol_array_shape = symbol_array.shape
    array_of_orbit_instances = np.array([tiling_dictionary[symbol] for symbol in symbol_array.ravel()]
                                        ).reshape(*symbol_array_shape)
    glued_orbit = glue(array_of_orbit_instances, class_constructor, **kwargs)
    return glued_orbit


def generate_symbol_arrays(tile_dictionary, glue_shape, unique=True):
    symbol_array_generator = itertools.product(list(tile_dictionary.keys()), repeat=np.product(glue_shape))
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


def query_symbolic_index(symbol_array, results_csv):
    """ Check to see if a combination has already been searched for locally.

    Returns
    -------

    Notes
    -----
    Computes the equivariant equivalents of the symbolic array being searched for.
    Strings can become arbitrarily long but I think this is unavoidable unless symbolic dynamics are redefined
    to get full shift.

    This checks the records/logs as to whether an orbit or one of its equivariant arrangements converged with
    a particular method.
    """
    from pandas import read_csv

    all_rotations = itertools.product(*(list(range(a)) for a in symbol_array.shape))
    axes = tuple(range(len(symbol_array.shape)))
    equivariant_symbol_string_list = []
    for rotation in all_rotations:
        equivariant_symbol_string_list.append(to_symbol_string(np.roll(symbol_array, rotation, axis=axes)))

    results_data_frame = read_csv(results_csv, index_col=0)
    n_permutations_in_results_log = results_data_frame.index.isin(equivariant_symbol_string_list).sum()
    return n_permutations_in_results_log


def to_symbol_string(symbol_array):
    symbolic_string = symbol_array.astype(str).copy()
    shape_of_axes_to_contract = symbol_array.shape[1:]
    for i, shp in enumerate(shape_of_axes_to_contract):
        symbolic_string = [(i*'_').join(list_) for list_ in np.array(symbolic_string).reshape(-1, shp).tolist()]
    symbolic_string = ((len(shape_of_axes_to_contract))*'_').join(symbolic_string)
    return symbolic_string


def to_symbol_array(symbol_string, symbol_array_shape):
    return np.array([char for char in symbol_string.replace('_', '')]).astype(int).reshape(symbol_array_shape)


