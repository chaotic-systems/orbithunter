from .optimize import minimize
import numpy as np

__all__ = ['continuation', 'discretization_continuation']


def _equals_target(orbit_, target_extent, parameter_label):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(getattr(orbit_, parameter_label), 13) == np.round(target_extent, 13)


def _increment_orbit_parameter(orbit_, target_extent, increment, parameter_label):
    """
    Parameters
    ----------
    orbit_
    target_extent
    increment
    label

    Returns
    -------

    """
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    current_extent = getattr(orbit_, parameter_label)
    # If the next step would overshoot, then the target value is the next step.
    if np.sign(target_extent - (current_extent + increment)) != np.sign(increment):
        next_extent = target_extent
    else:
        next_extent = current_extent + increment
    parameters = tuple(next_extent if lab == parameter_label else getattr(orbit_, lab)
                       for lab in orbit_.parameter_labels())
    return orbit_.__class__(state=orbit_.state, basis=orbit_.basis,
                            parameters=parameters, constraints=orbit_.constraints)


def continuation(orbit_, target_value, constraint_label, *extra_constraints,  step_size=0.01, **kwargs):
    """

    Parameters
    ----------
    orbit_
    target_value
    constraint_label
    step_size :
    kwargs :

    Returns
    -------


    """
    # As long as we keep converging to solutions, we keep stepping towards target value.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    # Use list to get the correct count, then convert to tuple as expected.
    # Check that the orbit_ is converged prior to any constraints

    minimize_result = minimize(orbit_.constrain((constraint_label, *extra_constraints), **kwargs))
    # check that the orbit_ instance is converged when having constraints, otherwise performance takes a big hit.
    # The constraints are applied but orbit_ can also be passed with the correct constrains.

    parameter_label = orbit_.parameter_labels().index(constraint_label)
    # Ensure that we are stepping in correct direction.
    step_size = (np.sign(target_value - getattr(minimize_result.orbit, parameter_label)) * np.abs(step_size))
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    while minimize_result.status == -1 and not _equals_target(minimize_result.orbit, target_value, parameter_label):
        incremented_orbit = _increment_orbit_parameter(minimize_result.orbit, target_value, step_size, parameter_label)
        minimize_result = minimize(incremented_orbit, **kwargs)
    return minimize_result


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
    orbit_ : Orbit or Orbit child
        The instance whose discretization is to be changed.
    target_discretization :
        The shape that will be incremented towards; failure to converge will terminate this process.
    cycle : bool
        Whether or not to applying the cycling strategy. See Notes for details.
    kwargs :
        any keyword arguments relevant for orbithunter.minimize


    Returns
    -------

    Notes
    -----
    The cycling strategy alternates between the axes of the smallest discretization size, as to allow for small changes
    in each dimension as opposed to incrementing all in one dimension at once.

    """
    # check that we are starting from converged solution, first of all.
    minimize_result = minimize(orbit_, **kwargs)
    axes_order = kwargs.get('axes_order', np.argsort(target_discretization)[::-1])
    # Use list to get the correct count, then convert to tuple as expected.
    step_sizes = kwargs.get('step_sizes', tuple(1 if min_dim % 2 else 2 for min_dim
                                                in np.array(orbit_.minimal_shape())[axes_order]))
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    if cycle:
        # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
        # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
        # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
        cycle_index = 0 
        while minimize_result.status == -1 and minimize_result.orbit.shapes()[0] != target_discretization:

            # Ensure that we are stepping in correct direction.
            step_size = (np.sign(target_discretization[axes_order[cycle_index]] 
                                 - minimize_result.orbit.shapes()[0][axes_order[cycle_index]])
                         * np.abs(step_sizes[axes_order[cycle_index]]))
            incremented_orbit = _increment_discretization(minimize_result.orbit,
                                                          target_discretization[axes_order[cycle_index]],
                                                          step_size, axis=axes_order[cycle_index])
            minimize_result = minimize(incremented_orbit, **kwargs)
            cycle_index = np.mod(cycle_index+1, len(axes_order))
    else:
        # As long as we keep converging to solutions, we keep stepping towards target value.
        for axis in axes_order:
            # Ensure that we are stepping in correct direction.
            step_size = np.abs(step_sizes[axis]) * np.sign(target_discretization[axis]
                                                           - minimize_result.orbit.shapes()[0][axis])

            # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
            # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
            # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
            while (minimize_result.status == -1
                   and not minimize_result.orbit.shapes()[0][axis] == target_discretization[axis]):
                incremented_orbit = _increment_discretization(minimize_result.orbit, target_discretization[axis],
                                                              step_size, axis=axis)
                minimize_result = minimize(incremented_orbit, **kwargs)

    return minimize_result
