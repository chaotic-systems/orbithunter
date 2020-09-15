from .optimize import converge
import numpy as np


def extent_equals_target(orbit_, target_extent, axis=1):
    # For the sake of floating point error, round to 13 decimals.
    return np.round(list(orbit_.parameters.values())[axis], 13) == np.round(target_extent, 13)


def increment_continuation_dimension(orbit_, target_extent, increment, axis=0):
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    params = orbit_.parameters()
    param_key, param_val = list(params.items())[axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_extent - param_val) != np.sign(increment):
        params[param_key] = target_extent
    else:
        params[param_key] += increment

    return orbit_.__class__(state=orbit_.state, state_type=orbit_.state_type,
                            parameters=params, constraints=orbit_.constraints)


def continuation(orbit_, target_extent, axis=1, step_size=0.1, save_directory='local', **kwargs):

    # Check to see that the continuation dimension is constrained. Other axes should not be constrained for practical
    # purposes but it is allowed. I'm using an assertion here to ensure that this is a very deliberate function.
    assert list(orbit_.constraints.values())[axis]

    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit_)
    # As long as we keep converging to solutions, we keep stepping towards target value.
    while converge_result.exit_code == 1 and not extent_equals_target(converge_result.orbit, target_extent,
                                                                      axis=axis):
        incremented_orbit = increment_continuation_dimension(converge_result.orbit, target_extent, step_size, axis=axis)
        converge_result = converge(incremented_orbit)
        if kwargs.get('save_all', False):
            converge_result.orbit.to_h5(directory='../data/continuations/local')
            converge_result.orbit.plot(show=False, directory='../figs/continuations/local')

    if converge_result.exit_code == 1 and extent_equals_target(converge_result.orbit, target_extent,
                                                               axis=axis):
        converge_result.orbit.to_h5(directory='../data/continuations/local')
        converge_result.orbit.plot(show=False, directory='../figs/continuations/local')
    return None

