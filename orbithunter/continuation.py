from orbithunter.optimize import converge
import numpy as np


def extent_equals_target(orbit, target_extent, dimension='space'):
    # For the sake of floating point error, round to 13 decimals.
    if dimension == 'space':
        return np.round(orbit.L, 13) == np.round(target_extent, 13)
    else:
        return np.round(orbit.T, 13) == np.round(target_extent, 13)


def increment_continuation_dimension(orbit, target_extent, increment, dimension='space'):
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    if dimension == 'space':
        if np.sign(target_extent - orbit.L) != np.sign(increment):
            orbit.L = target_extent
        else:
            orbit.L = orbit.L + increment
    else:
        if np.sign(target_extent - orbit.T) != np.sign(increment):
            orbit.T = target_extent
        else:
            orbit.T = orbit.T + increment


def continuation_constraints(dimension='space'):
    if dimension == 'space':
        return False, True, False
    else:
        return True, False, False


def continuation(orbit, target_extent, dimension='space', step_size=0.1, save_directory='local', **kwargs):
    # check that we are starting from converged solution, first of all.
    converge_result = converge(orbit)
    # As long as we keep converging to solutions, we keep stepping towards target value.
    while converge_result.exit_code == 1 and not extent_equals_target(converge_result.orbit, target_extent,
                                                                      dimension=dimension):
        increment_continuation_dimension(converge_result.orbit, target_extent, step_size, dimension=dimension)
        converge_result = converge(converge_result.orbit,
                                   parameter_constraints=continuation_constraints(dimension=dimension))
        if kwargs.get('save_all', False):
            converge_result.orbit.to_h5(directory=save_directory)
            converge_result.orbit.plot(show=False, directory=save_directory)

    if converge_result.exit_code == 1 and extent_equals_target(converge_result.orbit, target_extent,
                                                               dimension=dimension):
        converge_result.orbit.to_h5(directory=save_directory)
        converge_result.orbit.plot(show=False, directory=save_directory)
    return None
