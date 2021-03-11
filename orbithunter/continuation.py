from .optimize import hunt
from collections import deque
from itertools import islice
import numpy as np
import warnings

__all__ = ['continuation', 'discretization_continuation', 'span_family']


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
    # This overwrites current parameters with updated parameters, while also keeping all necessary attributes
    return orbit_.__class__(**{**vars(orbit_), 'parameters': parameters})


def continuation(orbit_, constraint_item, *extra_constraints,  step_size=0.01, **kwargs):
    """

    Parameters
    ----------
    orbit_
    constraint_item
    step_size :
    kwargs :

    Returns
    -------


    """
    # check that the orbit_ instance is converged when having constraints
    if isinstance(constraint_item, type({}.items())):
        constraint_label, target_value = tuple(*constraint_item)
    elif isinstance(constraint_item, dict):
        constraint_label, target_value = tuple(*constraint_item.items())
    elif isinstance(constraint_item, tuple):
        constraint_label, target_value = constraint_item
    else:
        raise TypeError('constraint_item is expected to be dict, dict_item, tuple containing a single key, value pair.')

    orbit_.constrain((constraint_label, *extra_constraints))
    minimize_result = hunt(orbit_, **kwargs)

    # Derive step size with correct sign.
    step_size = (np.sign(target_value - getattr(minimize_result.orbit, constraint_label)) * np.abs(step_size))

    while minimize_result.status == -1 and not _equals_target(minimize_result.orbit, target_value, constraint_label):
        # Having to specify both seems strange and so the options are: provide save=True and then use default
        # filename, or provide filename.
        
        if kwargs.get('save', False):
            # When generating an orbits' continuous family, it is useful to save the intermediate states
            # so that they may be referenced in future calculations
            valstr = str(np.round(getattr(minimize_result.orbit, constraint_label),
                                  int(abs(np.log10(abs(step_size))))+1)).replace('.', 'p')
            dname = ''.join([constraint_label, valstr])
            fname = kwargs.get('filename', None) or ''.join(['continuation_', orbit_.filename()])
            gname = kwargs.get('groupname', '')
            # pass keywords like this to avoid passing multiple values to same keyword.
            minimize_result.orbit.to_h5(**{**kwargs, 'filename': fname, 'dataname': dname, 'groupname': gname})
            
        incremented_orbit = _increment_orbit_parameter(minimize_result.orbit, target_value, step_size, constraint_label)
        minimize_result = hunt(incremented_orbit, **kwargs)
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
    minimize_result = hunt(orbit_, **kwargs)
    axes_order = kwargs.get('axes_order', np.argsort(target_discretization)[::-1])
    # The minimum step size is inferred from the minimal shapes if not provided; the idea here is that if
    # the minimum shape is odd then
    step_sizes = kwargs.get('step_sizes', np.array(orbit_.minimal_shape_increments())[axes_order])
    # To be efficient, always do the smallest target axes first.
    # We need to be incrementing in the correct direction. i.e. to get smaller we need to have a negative increment.
    if cycle:
        # While maintaining convergence proceed with continuation. If the shape equals the target, stop.
        # If the shape along the axis is 1, and the corresponding dimension is 0, then this means we have
        # an equilibrium solution along said axis; this can be handled by simply rediscretizing the field.
        cycle_index = 0 
        while minimize_result.status == -1 and minimize_result.orbit.shapes()[0] != target_discretization:
            # Having to specify both seems strange and so the options are: provide save=True and then use default
            # filename, or provide filename.
            if kwargs.get('save', False):
                # When generating an orbits' continuous family, it is useful to save the intermediate states
                # so that they may be referenced in future calculations
                fname = kwargs.get('filename', None) or ''.join(['discretization_continuation_', orbit_.filename()])
                gname = kwargs.get('groupname', '')
                # pass keywords like this to avoid passing multiple values to same keyword.
                minimize_result.orbit.to_h5(**{**kwargs, 'filename': fname, 'groupname': gname})

            # Ensure that we are stepping in correct direction.
            step_size = (np.sign(target_discretization[axes_order[cycle_index]] 
                                 - minimize_result.orbit.shapes()[0][axes_order[cycle_index]])
                         * np.abs(step_sizes[axes_order[cycle_index]]))
            incremented_orbit = _increment_discretization(minimize_result.orbit,
                                                          target_discretization[axes_order[cycle_index]],
                                                          step_size, axis=axes_order[cycle_index])
            minimize_result = hunt(incremented_orbit, **kwargs)
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
                # When generating an orbits' continuous family, it is useful to save the intermediate states
                # so that they may be referenced in future calculations
                if kwargs.get('save', False):
                    fname = kwargs.get('filename', None) or ''.join(['discretization_continuation_', orbit_.filename()])
                    gname = kwargs.get('groupname', '')
                    # pass keywords like this to avoid passing multiple values to same keyword.
                    minimize_result.orbit.to_h5(**{**kwargs, 'filename': fname, 'groupname': gname})

                incremented_orbit = _increment_discretization(minimize_result.orbit, target_discretization[axis],
                                                              step_size, axis=axis)
                minimize_result = hunt(incremented_orbit, **kwargs)

    return minimize_result


def span_family(orbit_, **kwargs):
    """ Explore and span an orbit's family (continuation and group orbit)

    Parameters
    ----------
    orbit_ : Orbit
        The orbit whose family is to be spanned.
    kwargs :
        Keyword arguments accepted by to_h5 and continuation methods (and hunt function, by proxy),
        and Orbit.group_orbit

    Returns
    -------
    orbit_family :
        A list of double ended queues, each queue being a branch of orbit states (NOT the group orbits, for memory
        considerations).

    -------

    Notes
    -----
    The ability to continue all continuations and hence span the family geometrically has been removed. It simply
    is too much to allow in a single function call. To span the *entire* family, run this function on 
    various members of the family, possibly those populated by a previous function call. In that instance, using
    same filename between runs is beneficial. 

    This function saves everything by default. To disable all saving, **BOTH** save=None AND filename=None are required
    in keyword arguments. 
    
    How naming conventions work: family name is the filename. Each branch is a group or subgroup, depending on 
    root only. If root_only=False then this behaves recursively and can get incredibly large. Use at your own risk.
    """
    # Check and make sure the root orbit is actually a converged orbit.
    root_orbit_result = hunt(orbit_, **kwargs)
    if root_orbit_result.status != -1:
        warn_str = '\nunconverged root orbit in family spanning. Change tol or orbit to avoid this message.'
        warnings.warn(warn_str, RuntimeWarning)

    # In order to be able to account for different behaviors when constraining different dimensions, allow
    # iterable of step_sizes. Keys should be dimension labels, vals should be step sizes for that dimension.
    step_sizes = kwargs.get('step_sizes', {})

    # Step the bounds of the continuation per dimension, provided as dict much like step sizes.
    bounds = kwargs.get('bounds', {k: (0, np.inf) for k in orbit_.dimension_labels()})
    kwargs.setdefault('filename', ''.join(['family_', orbit_.filename()]))

    # This is to avoid default producing excessively large families.
    kwargs.setdefault('strides', (n_points//4 for n_points in orbit_.discretization))

    # Only save the root orbit once. Save as deque, even though single element, simply for consistency.
    family = [deque([orbit_])]
    for dim in bounds:
        # If a dimension is 0 then its continuation is not meaningful.
        if getattr(root_orbit_result.orbit, dim) == 0.:
            continue
        branch = deque()
        # Regardless of what the user says, do the saving here and not inside continuation function
        step_size = step_sizes.get(dim, 0.01)
        kwargs = {'step_size': step_size, 'save': False, **kwargs}

        branch_orbit_result = root_orbit_result
        # While converged and in bounds, step in the positive direction, starting from the root orbit
        while branch_orbit_result.status == -1 and (getattr(branch_orbit_result.orbit, dim)
                                                    < bounds.get(dim)[1]):
            # Do not want to redundantly add root node
            if getattr(branch_orbit_result.orbit, dim) != getattr(orbit_, dim):
                branch.append(branch_orbit_result.orbit)
            constraint_item = (dim, getattr(branch_orbit_result.orbit, dim) + step_size)
            branch_orbit_result = continuation(branch_orbit_result.orbit, constraint_item, **kwargs)
        # Span the dimension in the negative direction, starting from the root orbit.
        branch_orbit_result = root_orbit_result
        # bounds.get using 'interval' tuple as default only then to slice it is consistently expect bounds
        # to be given as interval.

        # While converged and in bounds, step in the negative direction, starting from the root orbit
        while branch_orbit_result.status == -1 and (getattr(branch_orbit_result.orbit, dim)
                                                    > bounds.get(dim)[0]):
            # Do not want to redundantly add root node
            if getattr(branch_orbit_result.orbit, dim) != getattr(orbit_, dim):
                branch.appendleft(branch_orbit_result.orbit)
            constraint_item = (dim, getattr(branch_orbit_result.orbit, dim) - step_size)
            branch_orbit_result = continuation(branch_orbit_result.orbit, constraint_item, **kwargs)
        # After the family branch is populated, iterate over each branch members' group orbit. This can
        # be a LOT of orbits if you are not careful with sampling/keyword arguments.
        for leaf in islice(branch, 0, len(branch), kwargs.get('sampling_rate', 1)):
            # the ordering provided by the deques is invalidated if the orbits are allowed to be named
            # regarding their parameters.
            if kwargs.get('leafnames', False):
                dataname = leaf.filename(cls_name=False, extension='').lstrip('_')
            else:
                dataname = None
            leaf.to_h5(dataname=dataname, **kwargs)
        family.append(branch)
    return family
