from .optimize import converge
import numpy as np

__all__ = ['parameter_continuation', 'discretization_continuation']


def _extent_equals_target(orbit_, target_extent, axis=0):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(list(orbit_.parameters)[axis], 13) == np.round(target_extent, 13)


def _increment_dimension(orbit_, target_extent, increment, axis=0):
    """

    Parameters
    ----------
    orbit_
    target_extent
    increment
    axis

    Returns
    -------

    """
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    current_extent = orbit_.parameters[axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_extent - (current_extent + increment)) != np.sign(increment):
        next_extent = target_extent
    else:
        next_extent = current_extent + increment
    parameters = tuple(next_extent if i == axis else orbit_.parameters[i]
                       for i in range(len(orbit_.parameters)))
    return orbit_.__class__(state=orbit_.state, basis=orbit_.basis,
                            parameters=parameters, constraints=orbit_.constraints)


def _increment_discretization(orbit_, target_size, increment, axis=0):
    """

    Parameters
    ----------
    orbit_
    target_size
    increment
    axis

    Returns
    -------

    """
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    current_size = orbit_.shapes()[0][axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_size - (current_size + increment)) != np.sign(increment):
        next_size = target_size
    else:
        next_size = current_size + increment
    incremented_shape = tuple(d if i != axis else next_size for i, d in enumerate(orbit_.shapes()[0]))
    return orbit_.resize(*incremented_shape)


def parameter_continuation(orbit_, new_size, axis=0, step_size=0.01, **kwargs):
    """

    Parameters
    ----------
    orbit_
    target_dimensions
    kwargs

    Returns
    -------


    """
    # As long as we keep converging to solutions, we keep stepping towards target value.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    # Use list to get the correct count, then convert to tuple as expected.
    # Check that the orbit_ is converged prior to any constraints

    converge_result = converge(orbit_.constrain(axis=axis), **kwargs)
    # check that the orbit_ instance is converged when having constraints, otherwise performance takes a big hit.
    # The constraints are applied but orbit_ can also be passed with the correct constrains.

    # Ensure that we are stepping in correct direction.
    step_size = (np.sign(new_size - converge_result.orbit.parameters[axis]) * np.abs(step_size))
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    while converge_result.status == -1 and not _extent_equals_target(converge_result.orbit, new_size, axis=axis):
        incremented_orbit = _increment_dimension(converge_result.orbit, new_size, step_size, axis=axis)
        converge_result = converge(incremented_orbit, **kwargs)

    return converge_result


def discretization_continuation(orbit_, target_shape, cycle=False, **kwargs):
    """ Incrementally change discretization while maintaining convergence

    Parameters
    ----------
    orbit_
    target_shape
    kwargs

    Returns
    -------

    """
    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit_, **kwargs)
    axes_order = kwargs.get('axes_order', np.argsort(target_shape)[::-1])
    # Use list to get the correct count, then convert to tuple as expected.
    step_sizes = kwargs.get('step_sizes', tuple(len(axes_order) * [2]))
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    if cycle:
        # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
        # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
        # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
        cycle_index = 0 
        while converge_result.status == -1 and converge_result.orbit.shapes()[0] != target_shape:

            # Ensure that we are stepping in correct direction.
            step_size = (np.sign(target_shape[axes_order[cycle_index]] 
                                 - converge_result.orbit.shapes()[0][axes_order[cycle_index]])
                         * np.abs(step_sizes[axes_order[cycle_index]]))
            incremented_orbit = _increment_discretization(converge_result.orbit, target_shape[axes_order[cycle_index]],
                                                          step_size, axis=axes_order[cycle_index])
            converge_result = converge(incremented_orbit, **kwargs)
            cycle_index = np.mod(cycle_index+1, len(axes_order))
    else:
        # As long as we keep converging to solutions, we keep stepping towards target value.
        for axis in axes_order:
            # Ensure that we are stepping in correct direction.
            step_size = (np.sign(target_shape[axis] - converge_result.orbit.shapes()[0][axis]) * np.abs(step_sizes[axis]))
            # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
            # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
            # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
            while converge_result.status == -1 and (not converge_result.orbit.shapes()[0][axis] == target_shape[axis]
                                                    and
                                                    not converge_result.orbit.shapes()[0][axis] == 1):
                incremented_orbit = _increment_discretization(converge_result.orbit, target_shape[axis],
                                                              step_size, axis=axis)
                converge_result = converge(incremented_orbit, **kwargs)

    if converge_result.status == -1:
        # At the very end, we are either at the correct shape, such that the next line does nothing OR we have
        # a discretization of an equilibrium solution that is brought to the final target shape by rediscretization.
        # In other words, the following rediscretization does not destroy the convergence of the orbit, if it
        # has indeed converged.
        converge_result.orbit = converge_result.orbit.resize(*target_shape)
    return converge_result
