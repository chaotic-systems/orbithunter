from .optimize import converge
from .discretization import rediscretize
import numpy as np

__all__ = ['dimension_continuation', 'discretization_continuation']


def _extent_equals_target(orbit_, target_extent, axis=0):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(list(orbit_.orbit_parameters)[axis], 13) == np.round(target_extent, 13)


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
    current_extent = orbit_.orbit_parameters[axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_extent - current_extent) != np.sign(increment):
        next_extent = target_extent
    else:
        next_extent = current_extent + increment
    orbit_parameters = tuple(next_extent if i == axis else orbit_.orbit_parameters[i]
                             for i in range(len(orbit_.orbit_parameters)))
    return orbit_.__class__(state=orbit_.state, state_type=orbit_.state_type,
                            orbit_parameters=orbit_parameters, constraints=orbit_.constraints)


def dimension_continuation(orbit_, target_dimensions, **kwargs):
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

    order_of_axes_to_increment = kwargs.get('axis_order', np.argsort(tuple(np.abs(target_dimensions[i]
                                                                                  - orbit_.orbit_parameters[i])
                                                                           for i in range(len(target_dimensions)))))
    # Use list to get the correct count, then convert to tuple as expected.
    step_sizes = kwargs.get('step_sizes', tuple(len(order_of_axes_to_increment) * [0.01]))

    # Check that the orbit_ is converged prior to any constraints
    converge_result = converge(orbit_, **kwargs)

    for axis in order_of_axes_to_increment:
        # check that the orbit_ instance is converged when having constraints, otherwise performance takes a big hit.
        # The constraints are applied but orbit_ can also be passed with the correct constrains.
        # This choice is described in the Notes section.
        orbit_.constrain(axis=axis)
        # Ensure that we are stepping in correct direction.
        step_size = (np.sign(target_dimensions[axis] - converge_result.orbit.orbit_parameters[axis])
                     * np.abs(step_sizes[axis]))
        # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
        while converge_result.exit_code == 1 and not _extent_equals_target(converge_result.orbit,
                                                                           target_dimensions[axis],
                                                                           axis=axis):

            incremented_orbit = _increment_dimension(converge_result.orbit, target_dimensions[axis],
                                                     step_size, axis=axis)
            converge_result = converge(incremented_orbit, **kwargs)
            # If we want to save all states in the family else save the returned orbit from converge_result
            # outside the function
            if kwargs.get('save', False):
                converge_result.orbit.to_h5(**kwargs)
                converge_result.orbit.plot(show=kwargs.pop('show', False), **kwargs)

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
    current_size = orbit_.field_shape[axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_size - current_size) != np.sign(increment):
        next_size = target_size
    else:
        next_size = current_size + increment
    incremented_shape = tuple(d if i != axis else next_size for i, d in enumerate(orbit_.field_shape))
    return rediscretize(orbit_, new_shape=incremented_shape)


def discretization_continuation(orbit_, target_shape, **kwargs):
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
    converge_result = converge(orbit_)
    order_of_axes_to_increment = np.argsort(target_shape)
    # Use list to get the correct count, then convert to tuple as expected.
    step_sizes = kwargs.get('step_sizes', tuple(len(order_of_axes_to_increment) * [2]))
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.

    # As long as we keep converging to solutions, we keep stepping towards target value.
    for axis in order_of_axes_to_increment:
        # Ensure that we are stepping in correct direction.
        step_size = (np.sign(target_shape[axis] - converge_result.orbit.field_shape[axis]) * np.abs(step_sizes[axis]))
        # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
        # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
        # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
        while converge_result.exit_code == 1 and (not converge_result.orbit.field_shape[axis] == target_shape[axis] and
                                                  not converge_result.orbit.field_shape[axis] == 1):
            incremented_orbit = _increment_discretization(converge_result.orbit, target_shape[axis],
                                                          step_size, axis=axis)
            converge_result = converge(incremented_orbit, **kwargs)
            if kwargs.get('save', False):
                converge_result.orbit.to_h5(**kwargs)
                converge_result.orbit.plot(show=kwargs.pop('show', False), **kwargs)
    # At the very end, we are either at the correct shape, such that the next line does nothing OR we have
    # a discretization of an equilibrium solution that is brought to the final target shape by rediscretization.
    # In other words, the following rediscretization does not destroy the convergence of the orbit.
    converge_result.orbit = rediscretize(converge_result.orbit, new_shape=target_shape)
    return converge_result
