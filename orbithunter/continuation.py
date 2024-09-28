from .optimize import hunt, OrbitResult
from collections import deque
from itertools import islice
import numpy as np
import warnings

__all__ = ["continuation", "discretization_continuation", "span_family"]


def _equals_target(orbit_instance, target_extent, parameter_label):
    """
    Helper function that checks if the target has been reached, approximately.

    """
    # For the sake of floating point error, round to 13 decimals.
    return np.round(getattr(orbit_instance, parameter_label), 13) == np.round(
        target_extent, 13
    )


def _increment_parameter(orbit_instance, target_extent, increment, parameter_label):
    """
    Increment an Orbit's constrained parameter by a fixed amount, bounded by target value.

    Parameters
    ----------
    orbit_instance : Orbit
        The Orbit whose parameter is to be incremented
    target_extent : float, int, complex
        The upper or lower bound of the continuation process.
    increment : float, int, complex
        The step size with which to increment by
    parameter_label : str
        The label which specifies the parameter attribute to be incremented.

    Returns
    -------
    Orbit :
        Instance with identical attributes as orbit_instance other than the incremented parameter.

    """
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    current_extent = getattr(orbit_instance, parameter_label)
    # If the next step would overshoot, then the target value is the next step.
    if np.sign(target_extent - (current_extent + increment)) != np.sign(increment):
        next_extent = target_extent
    else:
        next_extent = current_extent + increment
    parameters = tuple(
        next_extent if lab == parameter_label else getattr(orbit_instance, lab)
        for lab in orbit_instance.parameter_labels()
    )
    # This overwrites current parameters with updated parameters, while also keeping all necessary attributes
    return orbit_instance.__class__(
        **{**vars(orbit_instance), "parameters": parameters}
    )


def continuation(
    orbit_instance, constraint_item, *extra_constraints, step_size=0.01, **kwargs
):
    """
    Numerical continuation parameterized by a single parameter but supporting any number of constraints.

    Parameters
    ----------
    orbit_instance : Orbit
        Instance whose state's parameters are to be continued.
    constraint_item : dict, tuple, dict_items
        A key value pair indicating the parameter being continued and its target value.
    extra_constraints : dict
        When constraining for continuation, it may be important to constrain other parameters which are not directly
        changed or incremented.
    step_size : float
        The value to use as a continuation increment. E.g. if step_size = 0.1, the continuation will try to converge
        Orbits at p + 0.1, p + 0.2, ... (if target < p then these would be substractions). For most field equations
        the continuation represents continuous deformations and so this should be reflected in this step size; not all
        dimensions are equal; for example, the KSE is more lenient to changes in time 't' rather than space 'x' because
        it is a first order equation in 't' and fourth order in 'x'.

    Returns
    -------
    OrbitResult :
        Optimization result with orbit resulting from continuation; if continuation failed (solution did not converge)
        then the parameter value may be different from the target; this failure or success will be indicated
        in the 'status' attribute of the result.

    """
    if isinstance(constraint_item, type({}.items())):
        constraint_label, target_value = tuple(*constraint_item)
    elif isinstance(constraint_item, dict):
        constraint_label, target_value = tuple(*constraint_item.items())
    elif isinstance(constraint_item, tuple):
        constraint_label, target_value = constraint_item
    else:
        raise TypeError(
            "constraint_item is expected to be dict, dict_item, tuple containing a single key, value pair."
        )

    orbit_instance.constrain((constraint_label, *extra_constraints))
    minimize_result = OrbitResult(orbit=orbit_instance, status=1)

    # Derive step size with correct sign.
    step_size = np.sign(
        target_value - getattr(minimize_result.orbit, constraint_label)
    ) * np.abs(step_size)

    while minimize_result.status == 1 and not _equals_target(
        minimize_result.orbit, target_value, constraint_label
    ):
        if kwargs.get("save", False):
            # When generating an orbits' continuous family, it can be useful to save the intermediate states
            # so that they may be referenced in future calculations
            valstr = str(
                np.round(
                    getattr(minimize_result.orbit, constraint_label),
                    int(abs(np.log10(abs(step_size)))) + 1,
                )
            ).replace(".", "p")
            dname = "".join([constraint_label, valstr])
            fname = kwargs.get("filename", None) or "".join(
                ["continuation_", orbit_instance.filename()]
            )
            gname = kwargs.get("groupname", "")
            # pass keywords like this to avoid passing multiple values to same keyword.
            minimize_result.orbit.to_h5(
                **{**kwargs, "filename": fname, "dataname": dname, "groupname": gname}
            )

        incremented_orbit = _increment_parameter(
            minimize_result.orbit, target_value, step_size, constraint_label
        )
        minimize_result = hunt(incremented_orbit, **kwargs)
    return minimize_result


def _increment_discretization(orbit_instance, target_size, increment, axis=0):
    """Increment the discretization of an Orbit for discretization continuation.

    Parameters
    ----------
    orbit_instance : Orbit
        Instance whose state's shape is being changed.
    target_size : int
        The final target of the discretization
    increment : int
        The amount to change the change discretization by
    axis : int
        Orbit state array axis to change discretization of

    Returns
    -------
    Orbit :
        Orbit at new size.

    """
    # increments the target dimension but checks to see if incrementing places us out of bounds.
    current_size = orbit_instance.shapes()[0][axis]
    # The affirmative occurs when overshooting the target value of the param.
    if np.sign(target_size - (current_size + increment)) != np.sign(increment):
        next_size = target_size
    else:
        next_size = current_size + increment
    incremented_shape = tuple(
        d if i != axis else next_size for i, d in enumerate(orbit_instance.shapes()[0])
    )
    return orbit_instance.resize(*incremented_shape)


def discretization_continuation(
    orbit_instance, target_discretization, cycle=False, **kwargs
):
    """
    Incrementally change discretization while maintaining convergence

    Parameters
    ----------
    orbit_instance : Orbit or Orbit child
        The instance whose discretization is to be changed.
    target_discretization :
        The shape that will be incremented towards; failure to converge will terminate this process.
    cycle : bool
        Whether or not to applying the cycling strategy. See Notes for details.
    kwargs :
        any keyword arguments relevant for orbithunter.minimize


    Returns
    -------
    minimize_result : OrbitResult
        Orbit result from `hunt` function resulting from continuation; if continuation failed
        (solution did not converge) then the contained orbit's discretization may be different from the target.

    Notes
    -----
    The cycling strategy alternates between the axes of the smallest discretization size, as to allow for small changes
    in each dimension as opposed to incrementing all in one dimension at once.

    """
    # check that we are starting from converged solution, first of all.
    minimize_result = OrbitResult(orbit=orbit_instance, status=1)
    axes_order = np.array(
        kwargs.get("axes_order", np.argsort(target_discretization)[::-1])
    )
    # discretization increment
    step_sizes = kwargs.get(
        "step_sizes", np.array(orbit_instance.minimal_shape_increments())[axes_order]
    )

    if cycle:
        # cycling alternates between the incrementing axes.
        cycle_index = 0
        while (
            minimize_result.status == 1
            and minimize_result.orbit.shapes()[0] != target_discretization
        ):
            # Having to specify both seems strange and so the options are: provide save=True and then use default
            # filename, or provide filename.
            if kwargs.get("save", False):
                # When generating an orbits' continuous family, it is useful to save the intermediate states
                # so that they may be referenced in future calculations
                fname = kwargs.get("filename", None) or "".join(
                    ["discretization_continuation_", orbit_instance.filename()]
                )
                gname = kwargs.get("groupname", "")
                # pass keywords like this to avoid passing multiple values to same keyword.
                minimize_result.orbit.to_h5(
                    **{**kwargs, "filename": fname, "groupname": gname}
                )

            # Ensure that we are stepping in correct direction.
            step_size = np.sign(
                target_discretization[axes_order[cycle_index]]
                - minimize_result.orbit.shapes()[0][axes_order[cycle_index]]
            ) * np.abs(step_sizes[axes_order[cycle_index]])
            incremented_orbit = _increment_discretization(
                minimize_result.orbit,
                target_discretization[axes_order[cycle_index]],
                step_size,
                axis=axes_order[cycle_index],
            )
            minimize_result = hunt(incremented_orbit, **kwargs)
            cycle_index = np.mod(cycle_index + 1, len(axes_order))
    else:
        # As long as we keep converging to solutions, we keep stepping towards target value.
        for axis in axes_order:
            # Ensure that we are stepping in correct direction.
            step_size = np.abs(step_sizes[axis]) * np.sign(
                target_discretization[axis] - minimize_result.orbit.shapes()[0][axis]
            )

            # While maintaining convergence proceed with continuation.
            while (
                minimize_result.status == 1
                and not minimize_result.orbit.shapes()[0][axis]
                == target_discretization[axis]
            ):
                # When generating an orbits' continuous family, it can be useful to save the intermediate states
                # so that they may be referenced in future calculations
                if kwargs.get("save", False):
                    fname = kwargs.get("filename", None) or "".join(
                        ["discretization_continuation_", orbit_instance.filename()]
                    )
                    gname = kwargs.get("groupname", "")
                    minimize_result.orbit.to_h5(
                        **{**kwargs, "filename": fname, "groupname": gname}
                    )

                incremented_orbit = _increment_discretization(
                    minimize_result.orbit,
                    target_discretization[axis],
                    step_size,
                    axis=axis,
                )
                minimize_result = hunt(incremented_orbit, **kwargs)

    return minimize_result


def span_family(orbit_instance, **kwargs):
    """
    Explore and span an orbit's family (continuation and group orbit)

    Parameters
    ----------
    orbit_instance : Orbit
        The orbit whose family is to be spanned.

    kwargs :
        Keyword arguments accepted by to_h5 and continuation methods (and hunt function, by proxy),
        and Orbit.group_orbit

    Returns
    -------
    orbit_family : list of list
        A list of lists where each list is a branch of orbit states generated by continuation.

    Notes
    -----
    The ability to continue all continuations and hence span the family geometrically has been removed. It simply
    is too much to allow in a single function call. To span the entire family, run this function on
    various members of the family, possibly those populated by a previous function call. In that instance, using
    same filename between runs is beneficial.

    How naming conventions work: family name is the filename. Each branch is a group or subgroup, depending on
    root only. If root_only=False then this behaves recursively and can get incredibly large. Use at your own risk.

    """
    # Check and make sure the root orbit is actually a converged orbit.
    minimize_result = OrbitResult(orbit=orbit_instance, status=1)
    step_sizes = kwargs.get("step_sizes", {})

    # Step the bounds of the continuation per dimension, provided as dict much like step sizes.
    bounds = kwargs.get(
        "bounds", {k: (0, np.inf) for k in orbit_instance.dimension_labels()}
    )
    kwargs.setdefault("filename", "".join(["family_", orbit_instance.filename()]))

    # This is to avoid default producing excessively large families.
    kwargs.setdefault(
        "strides", (n_points // 4 for n_points in orbit_instance.discretization)
    )

    # Only save the root orbit once. Save as deque, even though single element, simply for consistency.
    family = [deque([orbit_instance])]
    for dim in bounds:
        # If a dimension is 0 then its continuation is not meaningful.
        if getattr(minimize_result.orbit, dim) == 0.0:
            continue
        branch = deque()
        # Regardless of what the user says, do the saving here and not inside continuation function
        step_size = step_sizes.get(dim, 0.01)
        kwargs = {"step_size": step_size, "save": False, **kwargs}

        branch_orbit_instanceresult = minimize_result
        # While converged and in bounds, step in the positive direction, starting from the root orbit
        while branch_orbit_instanceresult.status == 1 and (
            getattr(branch_orbit_instanceresult.orbit, dim) < bounds.get(dim)[1]
        ):
            # Do not want to redundantly add root node
            if getattr(branch_orbit_instanceresult.orbit, dim) != getattr(
                orbit_instance, dim
            ):
                branch.append(branch_orbit_instanceresult.orbit)
            constraint_item = (
                dim,
                getattr(branch_orbit_instanceresult.orbit, dim) + step_size,
            )
            branch_orbit_instanceresult = continuation(
                branch_orbit_instanceresult.orbit, constraint_item, **kwargs
            )
        # Span the dimension in the negative direction, starting from the root orbit.
        branch_orbit_instanceresult = minimize_result
        # bounds.get using 'interval' tuple as default only then to slice it is consistently expect bounds
        # to be given as interval.

        # While converged and in bounds, step in the negative direction, starting from the root orbit
        while branch_orbit_instanceresult.status == 1 and (
            getattr(branch_orbit_instanceresult.orbit, dim) > bounds.get(dim)[0]
        ):
            # Do not want to redundantly add root node
            if getattr(branch_orbit_instanceresult.orbit, dim) != getattr(
                orbit_instance, dim
            ):
                branch.appendleft(branch_orbit_instanceresult.orbit)
            constraint_item = (
                dim,
                getattr(branch_orbit_instanceresult.orbit, dim) - step_size,
            )
            branch_orbit_instanceresult = continuation(
                branch_orbit_instanceresult.orbit, constraint_item, **kwargs
            )
        # After the family branch is populated, iterate over each branch members' group orbit. This can
        # be a LOT of orbits if you are not careful with sampling/keyword arguments.
        for leaf in islice(branch, 0, len(branch), kwargs.get("sampling_rate", 1)):
            # the ordering provided by the deques is invalidated if the orbits are allowed to be named
            # regarding their parameters.
            if kwargs.get("leafnames", False):
                dataname = leaf.filename(cls_name=False, extension="").lstrip("_")
            else:
                dataname = None
            leaf.to_h5(dataname=dataname, **kwargs)
        family.append(list(branch))
    return family
