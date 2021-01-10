from .optimize import converge
import numpy as np

__all__ = ['continuation', 'discretization_continuation']


def _equals_target(orbit_, target_extent, index=0):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(list(orbit_.parameters)[index], 13) == np.round(target_extent, 13)


def _increment_orbit_parameter(orbit_, target_extent, increment, index=0):
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
    current_extent = orbit_.parameters[index]
    # If the next step would overshoot, then the target value is the next step.
    if np.sign(target_extent - (current_extent + increment)) != np.sign(increment):
        next_extent = target_extent
    else:
        next_extent = current_extent + increment
    parameters = tuple(next_extent if i == index else orbit_.parameters[i]
                       for i in range(len(orbit_.parameters)))
    return orbit_.__class__(state=orbit_.state, basis=orbit_.basis,
                            parameters=parameters, constraints=orbit_.constraints)


def continuation(orbit_, target_value, constraint_label, *extra_constraints,  step_size=0.01, **kwargs):
    """

    Parameters
    ----------
    orbit_
    target_value
    kwargs

    Returns
    -------


    """
    # As long as we keep converging to solutions, we keep stepping towards target value.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    # Use list to get the correct count, then convert to tuple as expected.
    # Check that the orbit_ is converged prior to any constraints

    converge_result = converge(orbit_.constrain((constraint_label, *extra_constraints), **kwargs))
    # check that the orbit_ instance is converged when having constraints, otherwise performance takes a big hit.
    # The constraints are applied but orbit_ can also be passed with the correct constrains.

    index = orbit_.parameter_labels().index(constraint_label)
    # Ensure that we are stepping in correct direction.
    step_size = (np.sign(target_value - converge_result.orbit.parameters[index]) * np.abs(step_size))
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    while converge_result.status == -1 and not _equals_target(converge_result.orbit, target_value,
                                                              index=index):
        incremented_orbit = _increment_orbit_parameter(converge_result.orbit, target_value, step_size,
                                                       index=index)
        converge_result = converge(incremented_orbit, **kwargs)
    return converge_result


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


def discretization_continuation(orbit_, target_discretization, cycle=False, **kwargs):
    """ Incrementally change discretization while maintaining convergence

    Parameters
    ----------
    orbit_
    target_discretization
    kwargs

    Returns
    -------

    """
    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit_, **kwargs)
    axes_order = kwargs.get('axes_order', np.argsort(target_discretization)[::-1])
    # Use list to get the correct count, then convert to tuple as expected.
    step_sizes = kwargs.get('step_sizes', tuple(len(axes_order) * [2]))
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    if cycle:
        # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
        # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
        # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
        cycle_index = 0 
        while converge_result.status == -1 and converge_result.orbit.shapes()[0] != target_discretization:

            # Ensure that we are stepping in correct direction.
            step_size = (np.sign(target_discretization[axes_order[cycle_index]] 
                                 - converge_result.orbit.shapes()[0][axes_order[cycle_index]])
                         * np.abs(step_sizes[axes_order[cycle_index]]))
            incremented_orbit = _increment_discretization(converge_result.orbit,
                                                          target_discretization[axes_order[cycle_index]],
                                                          step_size, axis=axes_order[cycle_index])
            converge_result = converge(incremented_orbit, **kwargs)
            cycle_index = np.mod(cycle_index+1, len(axes_order))
    else:
        # As long as we keep converging to solutions, we keep stepping towards target value.
        for axis in axes_order:
            # Ensure that we are stepping in correct direction.
            step_size = np.abs(step_sizes[axis]) * np.sign(target_discretization[axis]
                                                           - converge_result.orbit.shapes()[0][axis])

            # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
            # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
            # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
            while (converge_result.status == -1
                   and not converge_result.orbit.shapes()[0][axis] == target_discretization[axis]):
                incremented_orbit = _increment_discretization(converge_result.orbit, target_discretization[axis],
                                                              step_size, axis=axis)
                converge_result = converge(incremented_orbit, **kwargs)

    if converge_result.status == -1:
        # At the very end, we are either at the correct shape, such that the next line does nothing OR we have
        # a discretization of an equilibrium solution that is brought to the final target shape by rediscretization.
        # In other words, the following rediscretization does not destroy the convergence of the orbit, if it
        # has indeed converged.
        converge_result.orbit = converge_result.orbit.resize(*target_discretization)

    return converge_result
