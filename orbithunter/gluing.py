from .discretization import rediscretize
from .io import read_h5
import numpy as np
import os
import itertools

__all__ = ['tile', 'glue', 'generate_symbol_arrays']


def _correct_aspect_ratios(array_of_orbits, axis=0):

    iterable_of_dims = [o.dimensions[axis] for o in array_of_orbits.ravel()]
    iterable_of_shapes = [o.field_shape[axis] for o in array_of_orbits.ravel()]

    disc_total = np.sum(iterable_of_shapes)
    dim_total = np.sum(iterable_of_dims)

    fraction_array = np.array(iterable_of_dims) / dim_total
    new_discretization_sizes = (2 * np.round((fraction_array * disc_total) / 2)).astype(int)
    number_of_dimensionless_orbits = np.sum(fraction_array == 0)
    number_of_dimensionful_orbits = len(new_discretization_sizes) - number_of_dimensionless_orbits

    # Due to how the fractions and rounding can work out the following is a safety measure to ensure the
    # new discretization sizes add up to the original; this is a requirement in order to glue strips together.
    # The simplest way of adjusting these is to simply make every even (it already should be and I actually cannot
    # think of a case where this would not be true, but still), then add or subtract from the largest discretization
    # until we get to the original total.
    while np.sum(new_discretization_sizes) != disc_total:
        new_discretization_sizes[np.mod(np.sum(new_discretization_sizes), 2) == 1] += 1
        if np.sum(new_discretization_sizes) < disc_total:
            # Add points to the first instance of the minimum discretization size not equal to 0.
            non_zero_minimum = new_discretization_sizes[new_discretization_sizes > 0].min()
            new_discretization_sizes[np.min(np.where(new_discretization_sizes == non_zero_minimum)[0])] += 2
        else:
            # non_zero_maximum is same as maximum, just a precaution and consistency with minimum
            non_zero_maximum = new_discretization_sizes[new_discretization_sizes > 0].max()
            # Using np.argmin here would return the index relative to the sliced disc_size array, not the original as
            # we desire.
            new_discretization_sizes[np.min(np.where(new_discretization_sizes == non_zero_maximum)[0])] -= 2

    fraction_array = np.array(new_discretization_sizes) / np.sum(new_discretization_sizes)

    # Can't have orbits with 0 discretization but don't want them too large because they have no inherent scale.
    # If there are only orbits of this type however we need to return the original sizes.
    if number_of_dimensionless_orbits == len(iterable_of_shapes):
        # if gluing all equilibria in time, don't change anything.
        new_shapes = [o.shape for o in array_of_orbits.ravel()]

    elif number_of_dimensionless_orbits > 0 and number_of_dimensionful_orbits >= 1:
        # The following is the most convoluted piece of code perhaps in the entire package. It attempts to find the
        # best minimum discretization size for the dimensionless orbits; simply having them take the literal minimum
        # size is as good as throwing them out.
        # If the number of elements in the strip is large, then it is possible that the previous defaults will result
        # in 0's.
        half_dimless_disc = np.min([np.max([2 * (iterable_of_shapes[0] // (2*number_of_dimensionless_orbits
                                                                           * number_of_dimensionful_orbits)),
                                    disc_total // (len(iterable_of_shapes)*number_of_dimensionless_orbits),
                                    2*int((np.round(1.0 / np.min(fraction_array[fraction_array != 0]))+1) // 2)]),
                                    iterable_of_shapes[0]//2])

        # Find the number of points to take from each orbit. Multiply by two to get an even total.
        how_much_to_take_away_for_each = 2 * np.round(half_dimless_disc * fraction_array)
        # Subtract the number of points to take away from each per dimensionless
        new_discretization_sizes = (new_discretization_sizes
                                             - (number_of_dimensionless_orbits * how_much_to_take_away_for_each))
        # The size of the dimensionless orbits is of course how much we took from the other orbits.
        new_discretization_sizes[new_discretization_sizes == 0] = np.sum(how_much_to_take_away_for_each).astype(int)

        # The new shapes of each orbit are the new sizes along the gluing axis, and the originals for axes not being
        # glued.
        new_shapes = [tuple(int(new_discretization_sizes[j]) if i == axis else o.field_shape[i]
                        for i in range(len(o.shape))) for j, o in enumerate(array_of_orbits.ravel())]

    else:
        new_shapes = [tuple(int(new_discretization_sizes[j]) if i == axis else o.field_shape[i]
                      for i in range(len(o.shape))) for j, o in enumerate(array_of_orbits.ravel())]

    # Return the strip of orbits with corrected proportions.
    return np.array([rediscretize(o, new_shape=shp) for o, shp in zip(array_of_orbits.ravel(), new_shapes)])


def to_symbol_string(symbol_array):
    symbolic_string = symbol_array.astype(str).copy()
    shape_of_axes_to_contract = symbol_array.shape[1:]
    for i, shp in enumerate(shape_of_axes_to_contract):
        symbolic_string = [(i*'_').join(list_) for list_ in np.array(symbolic_string).reshape(-1, shp).tolist()]
    symbolic_string = ((len(shape_of_axes_to_contract))*'_').join(symbolic_string)
    return symbolic_string


def to_symbol_array(symbol_string, symbol_array_shape):
    return np.array([char for char in symbol_string.replace('_', '')]).astype(int).reshape(symbol_array_shape)


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
    corrected_pair_of_orbits = _correct_aspect_ratios(pair_of_orbits_array, gluing_axis=0)
    # Bundle all of the parameters at once, instead of "stripwise"
    zipped_dimensions = tuple(zip(*(o.dimensions for o in pair_of_orbits_array.ravel())))
    glued_parameters = class_constructor.glue_parameters(zipped_dimensions, glue_shape=glue_shape)

    # arrange the orbit states into an array of the same shape as the symbol array.
    orbit_field_list = [o.convert(to='field').state for o in pair_of_orbits_array.ravel()]
    glued_orbit_state = np.array(orbit_field_list).reshape(*pair_of_orbits_array.shape, *tiling_shape)
    # iterate through and combine all of the axes.
    while len(glued_orbit_state.shape) > len(tiling_shape):
        glued_orbit_state = np.concatenate(glued_orbit_state, axis=gluing_axis)

    glued_orbit = class_constructor(state=glued_orbit_state, basis='field',
                                    parameters=glued_parameters)


def tile_dictionary_ks(padded=False, comoving=False):
    """ Template tiles for Kuramoto-Sivashinsky equation.


    Parameters
    ----------
    padded : bool
    Whether to use the zero-padded versions of the tiles
    comoving : bool
    Whether to use the merger tile in the physical or comoving frame.

    Returns
    -------
    tile_dict : dict
    Dictionary which contains merger, streak, wiggle tiles for use in tiling and gluing.

    Notes
    -----
    The dictionary is setup as follows : {0: streak, 1: merger, 2: wiggle}
    """
    if padded:
        directory = os.path.abspath(os.path.join(__file__, '../../data/tiles/padded_space_unpadded_time/'))
    else:
        directory = os.path.abspath(os.path.join(__file__, '../../data/tiles/'))

    if comoving:
        # padded merger Orbit in comoving frame
        merger = read_h5('./OrbitKS_merger.h5', directory=directory, basis='field')
    else:
        # padded merger orbit in physical frame.
        merger = read_h5('./OrbitKS_merger_fdomain.h5', directory=directory, basis='field')

    # padded streak orbit
    streak = read_h5('./OrbitKS_streak.h5',directory=directory, basis='field')

    # padded wiggle orbit
    wiggle = read_h5('./OrbitKS_wiggle.h5', directory=directory, basis='field')

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
                array_of_orbit_instances_corrected = _correct_aspect_ratios(strip_of_orbits, axis=gluing_axis)
                # Concatenate the states with corrected proportions.
                glued_strip_state = np.concatenate(tuple(x.state for x in array_of_orbit_instances_corrected),
                                                   axis=gluing_axis)
                # Put the glued strip's state back into a class instance.
                glued_strip_orbit = class_constructor(state=glued_strip_state, basis='field',
                                                      parameters=glued_parameters)
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

        glued_orbit = class_constructor(state=glued_orbit_state, basis='field',
                                        parameters=glued_parameters)

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




