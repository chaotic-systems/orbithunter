from .optimize import converge
from .discretization import rediscretize
import numpy as np

__all__ = ['continuation', 'discretization_continuation']


def extent_equals_target(orbit_, target_extent, axis=1):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(list(orbit_.parameters.values())[axis], 13) == np.round(target_extent, 13)


def increment_dimension(orbit_, target_extent, increment, axis=0):
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    params = orbit_.parameters
    param_key, param_val = list(params.items())[axis]

    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_extent - param_val) != np.sign(increment):
        params[param_key] = target_extent
    else:
        params[param_key] += increment

    return orbit_.__class__(state=orbit_.state, state_type=orbit_.state_type,
                            parameters=params, constraints=orbit_.constraints)


def continuation(orbit_, target_extent, axis=1, step_size=0.1, **kwargs):

    # Check to see that the continuation dimension is constrained. Other axes should not be constrained for practical
    # purposes but it is allowed. I'm using an assertion here to ensure that this is a very deliberate function.
    # I'm not sure if I really want to keep this, as it is sometimes possible to slowly 'nudge' an orbit
    # in the right direction
    assert list(orbit_.constraints.values())[axis]

    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    assert np.sign(target_extent - list(orbit_.parameters.values())[axis]) == np.sign(step_size)

    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit_)
    # As long as we keep converging to solutions, we keep stepping towards target value.
    while converge_result.exit_code == 1 and not extent_equals_target(converge_result.orbit, target_extent,
                                                                      axis=axis):
        incremented_orbit = increment_dimension(converge_result.orbit, target_extent, step_size, axis=axis)
        converge_result = converge(incremented_orbit, **kwargs)
        if kwargs.get('save', False):
            converge_result.orbit.to_h5(**kwargs)
            converge_result.orbit.plot(show=kwargs.pop('show', False), **kwargs)

    return converge_result


def increment_discretization(orbit_, increment, axis=0):
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    incremented_shape = tuple(d if i != axis else d + increment for i, d in enumerate(orbit_.parameters['field_shape']))
    return rediscretize(orbit_, new_shape=incremented_shape)


def discretization_continuation(orbit_, target_shape, step_size=2,  **kwargs):

    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit_)
    order_of_axes_to_increment = np.argsort(target_shape)
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.

    # As long as we keep converging to solutions, we keep stepping towards target value.
    for axis in order_of_axes_to_increment:
        step_size = np.sign(target_shape[axis] - orbit_.parameters['field_shape'][axis]) * step_size
        while converge_result.exit_code == 1 and (not converge_result.orbit.parameters['field_shape'][axis]
                                                  == target_shape[axis]):
            incremented_orbit = increment_discretization(converge_result.orbit, step_size, axis=axis)
            converge_result = converge(incremented_orbit, **kwargs)
            if kwargs.get('save', False):
                converge_result.orbit.to_h5(**kwargs)
                converge_result.orbit.plot(show=kwargs.pop('show', False), **kwargs)

    return converge_result