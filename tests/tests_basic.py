from .context import orbithunter as oh
import pytest
import numpy as np
import os
import h5py


@pytest.fixture()
def fixed_orbit_data():
    """ Fixed test data to be used to initialize orbits"""
    # Originally populated from
    # np.random.seed(1)
    # np.random.randn(2,2,2,2).round(7)
    state = np.array(
        [
            [
                [[1.6243454, -0.6117564], [-0.5281718, -1.0729686]],
                [[0.8654076, -2.3015387], [1.7448118, -0.7612069]],
            ],
            [
                [[0.3190391, -0.2493704], [1.4621079, -2.0601407]],
                [[-0.3224172, -0.3840544], [1.1337694, -1.0998913]],
            ],
        ]
    )
    return state


@pytest.fixture()
def fixed_kwarg_dict():
    """ Passed to class methods to see if they breakdown upon unrecognized keyword arguments """
    return {
        "aint": "nothin",
        "gonna": "break",
        "my": "stride",
        "nobodys": "gonna",
        "slow": "me",
        "down,": "ohno",
        "I": "got",
        "to": "keep",
        "on": "movin",
    }


def test_create_orbit(fixed_orbit_data):
    """ Test initialization of an Orbit instance """
    return oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )


def test_Orbit_attributes(fixed_orbit_data):
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    assert orbit_.parameters[0] == orbit_.t
    assert orbit_.parameters[1] == orbit_.x
    assert orbit_.parameters[2] == orbit_.y
    assert orbit_.parameters[3] == orbit_.z
    assert orbit_.discretization[0] == orbit_.n
    assert orbit_.discretization[1] == orbit_.i
    assert orbit_.discretization[2] == orbit_.j
    assert orbit_.discretization[3] == orbit_.k
    assert oh.Orbit().state.shape == tuple(len(orbit_.default_shape()) * [0])
    assert oh.Orbit().basis is None
    with pytest.raises(AttributeError):
        _ = oh.Orbit().fakeattr


def test_binary_operations(fixed_orbit_data):
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    assert (orbit_ + orbit_).state.sum() == 2 * orbit_.state.sum()
    assert (orbit_ - orbit_).state.sum() == 0
    assert np.array_equal((orbit_ * orbit_).state, (orbit_ ** 2).state)
    assert (orbit_ / 2).state.sum() == (orbit_.state / 2).sum()
    assert (orbit_ // 5).state.sum() == -10.0


def test_assignment_operators(fixed_orbit_data):
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    orbit_ += orbit_
    orbit_ -= orbit_
    orbit_ *= orbit_
    orbit_ **= orbit_
    orbit_ /= orbit_
    orbit_ //= orbit_


def test_assignment_operators_no_state():
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit()
    orbit_ += orbit_
    orbit_ -= orbit_
    orbit_ *= orbit_
    orbit_ **= orbit_
    orbit_ /= orbit_
    orbit_ //= orbit_


def test_binary_operations_no_state():
    orbit_ = oh.Orbit()
    test_sum = orbit_ + orbit_
    test_sub = orbit_ - orbit_
    test_mul = orbit_ * orbit_
    test_pow = orbit_ ** 2
    test_div = orbit_ / 2
    test_floor_div = orbit_ // 2

def test_getitem(fixed_orbit_data):
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    test = orbit_[:1, :1, :1, :1]
    with pytest.raises(IndexError):
            empty_orbit = oh.Orbit()
            raise_index_error_because_empty_array = empty_orbit[:2]
def test_populate():
    """ Initialization in cases where generated information is desired. Occurs in-place"""
    z = oh.Orbit()
    x = z.populate(attr="parameters")
    for p, op in zip(x.parameters, z.parameters):
        assert p is not None
        assert (p >= 0) & (p <= 1)
        assert p == op

    y = z.populate(attr="state")
    assert y.size > 0
    assert y.shape != (0, 0, 0, 0)
    assert y.parameters == x.parameters
    # operations occur in place, should affect orbit_ and hence x as well because no copy was made.
    assert z.size > 0 and x.size > 0


def test_populate_seeding():
    x = oh.Orbit()
    x.populate(seed=0)
    y = oh.Orbit()
    y.populate(seed=0)
    assert np.isclose(x.state, y.state).all()
    assert x.parameters == y.parameters


def test_parameter_population():
    """ Test the parsing and population of parameters with or without user specified parameter ranges."""
    choice_list_possible_confused_with_interval = ["am I an", "interval?"]
    choice_list = ["this", "is", "the", "way", "to", "select"]
    parameter_ranges = {
        "t": (2, 2),
        "x": (100, 200),
        "y": choice_list_possible_confused_with_interval,
        "z": choice_list,
    }
    x = oh.Orbit().populate(attr="parameters", parameter_ranges=parameter_ranges)
    assert x.t == 2
    assert (x.x >= 100) and (x.x <= 200)
    assert x.y in choice_list_possible_confused_with_interval
    assert x.z in choice_list
    # bad parameter ranges
    should_raise_error = tuple(choice_list_possible_confused_with_interval)
    bad_parameter_ranges = {
        "t": (2,),
        "x": (100, 200),
        "y": should_raise_error,
        "z": choice_list,
    }
    with pytest.raises((TypeError, ValueError)):
        oh.Orbit().populate(attr="parameters", parameter_ranges=bad_parameter_ranges)

    pranges_missing_keys = {"t": np.ones([2, 2, 2, 2]), "z": choice_list}
    oh.Orbit().populate(attr="parameters", parameter_ranges=pranges_missing_keys)

    pranges_missing_keys_bundled_array = {
        "t": (np.ones([2, 2, 2, 2]),),
        "z": choice_list,
    }
    oh.Orbit().populate(
        attr="parameters", parameter_ranges=pranges_missing_keys_bundled_array
    )

    # partially populated
    oh.Orbit(parameters=(1, None, 1)).populate(
        attr="parameters", parameter_ranges=parameter_ranges
    )

    # partially populated with missing keys
    oh.Orbit(parameters=(0, 0, None)).populate(
        attr="parameters", parameter_ranges=pranges_missing_keys
    )


def test_overwriting():
    x = oh.Orbit().populate("state")
    with pytest.raises(ValueError):
        x.populate("state")
    x.populate("state", overwrite=True)
    assert x.parameters is None
    x.populate(overwrite=True)
    assert x.parameters is not None
    old_parameters = x.parameters
    y = x.populate(overwrite=True)
    assert y is x and old_parameters != y.parameters


def test_matrix_methods(fixed_orbit_data):
    """ Numerical methods required for matrix-constructing optimization"""
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    j = orbit_.jacobian()
    assert j.shape == (orbit_.size, orbit_.size + len(orbit_.parameters))


def test_constrain():
    x = oh.Orbit()
    x.constrain(oh.Orbit.parameter_labels())
    assert x.constraints == {"t": True, "x": True, "y": True, "z": True}
    # constraints are "refreshed" upon each call in order to allow for changing without checking values.
    x.constrain("I do not exist")
    assert x.constraints == {"t": False, "x": False, "y": False, "z": False}

    with pytest.raises(TypeError):
        oh.Orbit().constrain(["I am in a list"])


def test_from_numpy(fixed_orbit_data):
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    v = orbit_.orbit_vector()
    assert pytest.approx(v.sum(), (orbit_.state.sum() + sum(orbit_.parameters)))

    orbit_from_numpy = orbit_.from_numpy_array(v)
    assert (orbit_from_numpy.state == orbit_.state).all()

    orbit_from_numpy_passed_param = orbit_.from_numpy_array(v, parameters=(5, 5, 5, 5))
    assert orbit_from_numpy_passed_param.parameters == (5, 5, 5, 5)


def test_matrix_free_methods(fixed_orbit_data, fixed_kwarg_dict):
    """ Numerical methods required for matrix-free optimization"""
    orbit_ = oh.Orbit(
        state=fixed_orbit_data, basis="physical", parameters=(10, 10, 10, 10)
    )
    f = orbit_.eqn(fixed_kwarg_dict)
    grad = orbit_.cost_function_gradient(fixed_kwarg_dict)
    res = orbit_.residual(fixed_kwarg_dict)
    assert (f.state == np.zeros((2, 2, 2, 2))).all()
    assert (grad.state == np.zeros((2, 2, 2, 2))).all()
    assert res == 0.0

    matvec_ = orbit_.matvec(f, kwargs=fixed_kwarg_dict)
    assert matvec_.norm() == 0
    assert matvec_.parameters == orbit_.parameters

    rmatvec_ = orbit_.rmatvec(f, kwargs=fixed_kwarg_dict)
    assert rmatvec_.norm() == 0.0
    assert rmatvec_.parameters == (0, 0, 0, 0)


def test_properties(fixed_orbit_data):
    """ Call all properties to check if they are defined. """
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis="physical")
    _ = orbit_.shape
    _ = orbit_.size
    _ = orbit_.bases()
    _ = orbit_.parameter_labels()
    _ = orbit_.discretization_labels()
    _ = orbit_.dimension_labels()
    _ = orbit_.default_shape()
    _ = orbit_.minimal_shape()
    _ = orbit_.default_parameter_ranges()


def test_rediscretization(fixed_orbit_data):
    """ Check the reversibility of padding and truncation"""
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis="physical")
    enlarged = orbit_.resize(16, 16, 16, 16)
    shrank = enlarged.resize(orbit_.discretization)
    assert (shrank.state == orbit_.state).all()


def test_glue_dimensions(fixed_orbit_data):
    """ Test the manner by which new parameter values are generated for gluings"""
    x = oh.Orbit(state=fixed_orbit_data, basis="physical", parameters=(2, 2, 3, 4))
    y = oh.Orbit(state=fixed_orbit_data, basis="physical", parameters=(10, 0, 0, -4))
    dimension_tuples = tuple(zip(x.parameters, y.parameters))
    glue_shape = (2, 1, 1, 1)
    assert oh.Orbit.glue_dimensions(
        dimension_tuples, glue_shape, exclude_nonpositive=True
    ) == (12.0, 2.0, 3, 4.0)
    assert oh.Orbit.glue_dimensions(
        dimension_tuples, glue_shape, exclude_nonpositive=False
    ) == (12.0, 1.0, 1.5, 0.0)
    with pytest.raises((IndexError, ValueError)):
        glue_shape = (2, 1)
        oh.Orbit.glue_dimensions(
            dimension_tuples, glue_shape, exclude_nonpositive=False
        )


def test_symmetry(fixed_orbit_data):
    """ Test symmetry operations such as discrete rotations and reflections"""
    x = oh.Orbit(state=fixed_orbit_data, basis="physical", parameters=(2, 2, 3, 4))
    y = oh.Orbit(state=fixed_orbit_data, basis="physical", parameters=(10, 0, 0, -4))
    z = x.roll(1, axis=(0, 1, 2, 3))
    w = y.roll(1, axis=0).roll(1, axis=1).roll(1, axis=2).roll(1, axis=3)
    assert (z.state == w.state).all()
    v = x.cell_shift((1, 2, 2, 2), axis=(0, 1, 2, 3))
    u = y.roll((2, 1, 1, 1), axis=(0, 1, 2, 3))
    assert (u.state == v.state).all()
    assert (x[::-1, ::-1, ::-1, ::-1] + x.reflection(signed=True)).state.sum() == 0.0


@pytest.fixture()
def fixed_OrbitKS_data():
    # Generated from
    # np.random.seed(0)
    # np.random.randn(6, 6).round(8); values chosen because of size, truncation error
    state = np.array(
        [
            [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799, -0.97727788],
            [0.95008842, -0.15135721, -0.10321885, 0.4105985, 0.14404357, 1.45427351],
            [0.76103773, 0.12167502, 0.44386323, 0.33367433, 1.49407907, -0.20515826],
            [0.3130677, -0.85409574, -2.55298982, 0.6536186, 0.8644362, -0.74216502],
            [2.26975462, -1.45436567, 0.04575852, -0.18718385, 1.53277921, 1.46935877],
            [0.15494743, 0.37816252, -0.88778575, -1.98079647, -0.34791215, 0.15634897],
        ]
    )
    return state


def test_orbit_data(datafiles):
    with h5py.File("./tests/test_data.h5", "r") as file:

        def h5_helper(name, cls):
            nonlocal file
            attrs = dict(file["/".join([name, "0"])].attrs.items())
            state = file["/".join([name, "0"])][...]
            return cls(
                state=state,
                **{
                    **attrs,
                    "parameters": tuple(attrs["parameters"]),
                    "discretization": tuple(attrs["discretization"]),
                }
            )

        rpo = h5_helper("rpo", oh.RelativeOrbitKS)
        defect = h5_helper("defect", oh.RelativeOrbitKS)
        large_defect = h5_helper("large_defect", oh.RelativeOrbitKS)
        drifter = h5_helper("drifter", oh.RelativeEquilibriumOrbitKS)
        wiggle = h5_helper("wiggle", oh.AntisymmetricOrbitKS)
        streak = h5_helper("streak", oh.EquilibriumOrbitKS)
        double_streak = h5_helper("double_streak", oh.EquilibriumOrbitKS)
        manual = [rpo, defect, large_defect, drifter, wiggle, streak, double_streak]

    # Read in the same orbits as above using the native orbithunter io
    # keys included so that the import order matches `static_orbits`
    keys = (
        "rpo",
        "defect",
        "large_defect",
        "drifter",
        "wiggle",
        "streak",
        "double_streak",
    )
    automatic = oh.read_h5("./tests/test_data.h5", keys)

    for static, read in zip(manual, automatic):
        assert static.residual() < 1e-7
        assert static.residual() == read.residual()
        assert np.isclose(static.state, read.state).all()
        assert static.parameters == read.parameters


@pytest.fixture()
def fixed_data_transform_norms_dict():
    orbitks_norms = [
        6.783718210674288,
        5.2245770410096055,
        5.050390311315469,
        5.050390311315469,
        5.050390311315469,
    ]
    rel_norms = [
        6.783718210674288,
        5.2245770410096055,
        5.050390311315469,
        5.050390311315469,
        5.050390311315469,
    ]
    sr_norms = [
        6.783718210674288,
        5.2245770410096055,
        3.635581670379944,
        3.6355816703799446,
        3.6355816703799437,
    ]
    anti_norms = [
        6.783718210674288,
        5.2245770410096055,
        3.821906904088631,
        3.821906904088631,
        3.821906904088631,
    ]
    eqv_norms = [
        6.783718210674288,
        5.2245770410096055,
        1.159165311191572,
        2.8393635399538266,
        2.8393635399538266,
    ]
    reqv_norms = [
        6.783718210674288,
        5.2245770410096055,
        2.384651938236806,
        5.841180462819081,
        5.8411804628190795,
    ]
    norms = [orbitks_norms, rel_norms, sr_norms, anti_norms, eqv_norms, reqv_norms]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))


@pytest.fixture()
def fixed_eqn_norm_dict():
    norms = [
        1.258363954936136,
        1.258363954936136,
        0.8876335300535965,
        0.9747965276558949,
        0.033697585549483405,
        0.37894868060091114,
    ]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))


@pytest.fixture()
def fixed_spt_projection_space_derivative_norms():
    """ Norms for first four spatial derivatives and the first temporal derivative for each class

    Returns
    -------
    norms : np.ndarray
        The norms of derivatives u_x, u_xx, u_xxx, u_xxxx and u_t (columns) for classes
        OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, AntisymmetricOrbitKS, EquilibriumOrbitKS
        RelativeEquilibriumOrbitKS.
    """
    norms = np.array(
        [
            [1.112068, 0.289253, 0.080459, 0.022823, 1.071426],
            [1.112068, 0.289253, 0.080459, 0.022823, 1.071426],
            [0.782449, 0.20109, 0.055719, 0.015789, 0.731273],
            [0.800333, 0.202515, 0.055823, 0.015796, 0.750396],
            [0.797298, 0.226725, 0.064682, 0.018468, 0.0],
            [1.474815, 0.406224, 0.114923, 0.032743, 0.0],
        ]
    )
    return norms


@pytest.fixture()
def fixed_space_projection_space_derivative_norms():
    """ Norms for first four spatial derivatives and the first temporal derivative for each class

    Returns
    -------
    norms : np.ndarray
        The norms of derivatives u_x, u_xx, u_xxx, u_xxxx and u_t (columns) for classes
        OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, AntisymmetricOrbitKS, EquilibriumOrbitKS
        RelativeEquilibriumOrbitKS.
    """
    norms = np.array(
        [
            [1.138022, 0.294354, 0.081732, 0.023173, 1.071426],
            [1.138022, 0.294354, 0.081732, 0.023173, 1.071426],
            [1.138022, 0.294354, 0.081732, 0.023173, 0.731273],
            [1.138022, 0.294354, 0.081732, 0.023173, 0.750396],
            [1.138022, 0.294354, 0.081732, 0.023173, 0.0],
            [1.138022, 0.294354, 0.081732, 0.023173, 0.0],
        ]
    )
    return norms


@pytest.fixture()
def fixed_spt_projection_spt_derivative_norms():
    """ Norms for first four spatial derivatives and the first temporal derivative for each class

    Returns
    -------
    norms : np.ndarray
        The norms of derivatives u_x, u_xx, u_xxx, u_xxxx and u_t (columns) for classes
        OrbitKS, RelativeOrbitKS, ShiftReflectionOrbitKS, AntisymmetricOrbitKS, EquilibriumOrbitKS
        RelativeEquilibriumOrbitKS.
    """
    norms = np.array(
        [
            [1.112068, 0.289253, 0.080459, 0.022823, 1.071426],
            [1.112068, 0.289253, 0.080459, 0.022823, 1.071426],
            [0.0, 0.20109, 0.0, 0.015789, 0.731273],
            [0.0, 0.202515, 0.0, 0.015796, 0.750396],
            [0.0, 0.09256, 0.0, 0.00754, 0.0],
            [0.602091, 0.16584, 0.046917, 0.013367, 0.0],
        ]
    )
    return norms


@pytest.fixture()
def fixed_ks_parameters():
    # Parameter tuples and incorrect input scalar
    return (44, 44, 5), (44, 44), (44,), 44, None


def test_OrbitKS_attributes(fixed_OrbitKS_data, fixed_ks_parameters, kse_classes):
    for name, cls in kse_classes.items():
        orbit_ = cls(
            state=fixed_OrbitKS_data, basis="field", parameters=fixed_ks_parameters[0]
        )
        assert orbit_.parameters[0] == orbit_.t
        assert orbit_.parameters[1] == orbit_.x
        assert orbit_.parameters[2] == orbit_.s
        assert orbit_.discretization[0] == orbit_.n
        assert orbit_.discretization[1] == orbit_.m
        assert cls().state.shape == tuple(len(orbit_.default_shape()) * [0])
        assert cls().basis is None
        with pytest.raises(AttributeError):
            _ = oh.Orbit().fakeattr


@pytest.fixture()
def kse_classes():
    return dict(
        [(name, cls) for name, cls in oh.ks.__dict__.items() if isinstance(cls, type)]
    )


@pytest.fixture()
def jacobian_abssums():
    return np.array(
        [11.62823786, 11.87087419, 3.8862443, 3.98355746, 0.11814049, 1.10636676]
    )


def instance_generator(fixed_OrbitKS_data, kse_classes, fixed_ks_parameters):
    for (name, cls) in kse_classes.items():
        yield cls(
            state=fixed_OrbitKS_data, basis="field", parameters=fixed_ks_parameters
        )


def test_spt_projection_space_derivative(
    fixed_OrbitKS_data, fixed_spt_projection_space_derivative_norms, kse_classes
):
    all_norms = []
    for name, cls in kse_classes.items():
        class_norms = []
        o = cls(
            state=fixed_OrbitKS_data, parameters=(44, 44, 5), basis="field"
        ).transform(to="modes")
        for order in range(1, 5):
            class_norms.append(o.dx(order=order, return_basis="modes").norm())
        class_norms.append(o.dt().norm())
        all_norms.append(class_norms)
    pytest.approx(
        np.array(all_norms).round(6), fixed_spt_projection_space_derivative_norms
    )


def test_space_projection_space_derivative(
    fixed_OrbitKS_data, fixed_space_projection_space_derivative_norms, kse_classes
):
    all_norms = []
    for name, cls in kse_classes.items():
        class_norms = []
        o = cls(
            state=fixed_OrbitKS_data, parameters=(44, 44, 5), basis="field"
        ).transform(to="modes")
        for order in range(1, 5):
            class_norms.append(o.dx(order=order, return_basis="modes").norm())
        class_norms.append(o.dt().norm())
        all_norms.append(class_norms)
    pytest.approx(
        np.array(all_norms).round(6), fixed_space_projection_space_derivative_norms
    )


def test_spt_projection_spt_derivative(
    fixed_OrbitKS_data, fixed_spt_projection_spt_derivative_norms, kse_classes
):
    all_norms = []

    for name, cls in kse_classes.items():
        class_norms = []
        o = cls(
            state=fixed_OrbitKS_data, parameters=(44, 44, 5), basis="field"
        ).transform(to="modes")
        for order in range(1, 5):
            class_norms.append(o.dx(order=order, return_basis="modes").norm())
        class_norms.append(o.dt().norm())
        all_norms.append(class_norms)
    pytest.approx(
        np.array(all_norms).round(6), fixed_spt_projection_spt_derivative_norms
    )


def test_rmatvec(fixed_OrbitKS_data, fixed_ks_parameters):
    with h5py.File("./tests/test_data.h5", "r") as file:
        attrs = dict(file["rpo/0"].attrs.items())
        relorbit_ = oh.RelativeOrbitKS(
            state=file["rpo/0"][...],
            **{
                **attrs,
                "parameters": tuple(attrs["parameters"]),
                "discretization": tuple(attrs["discretization"]),
            }
        ).transform(to="modes")

    assert pytest.approx(relorbit_.rmatvec(relorbit_).norm(), 60.18805016)
    assert pytest.approx(relorbit_.cost_function_gradient(relorbit_).norm(), 0.0)

    orbit_ = oh.OrbitKS(
        state=fixed_OrbitKS_data, parameters=fixed_ks_parameters[0], basis="field"
    ).transform(to="modes")
    assert pytest.approx(orbit_.rmatvec(orbit_).norm(), 1.295386)
    assert pytest.approx(orbit_.cost_function_gradient(orbit_.eqn()).norm(), 1.0501956)


def test_matvec(fixed_OrbitKS_data, fixed_ks_parameters):
    relorbit_ = oh.read_h5("./tests/test_data.h5", "rpo")
    orbit_ = oh.OrbitKS(
        state=fixed_OrbitKS_data, parameters=fixed_ks_parameters[0], basis="field"
    ).transform(to="modes")

    assert pytest.approx(relorbit_.matvec(relorbit_).norm(), 0.3641827310989932)
    assert pytest.approx(orbit_.matvec(orbit_).norm(), 155.86156902189782)


def test_transforms(
    fixed_OrbitKS_data,
    kse_classes,
    fixed_data_transform_norms_dict,
    fixed_ks_parameters,
):
    """ Testing the transforms by comparing norms

    Returns
    -------

    Notes
    -----
    The transforms are orthogonal transformations but only after projection into the various symmetry invariant
    subspaces; these projections are built-in to the transforms themselves such that the chain of transformations
    (labeled by basis) field --> spatial_modes --> modes --> spatial_modes --> field
    (can norm change?)       --> yes --> yes --> no --> no

    """
    for (name, cls) in kse_classes.items():
        orbit_in_field_basis = cls(state=fixed_OrbitKS_data, basis="field")
        orbit_in_spatial_mode_basis = orbit_in_field_basis.transform(to="spatial_modes")
        orbit_in_mode_basis = orbit_in_spatial_mode_basis.transform(to="modes")

        # Using the fixed data, all initial norms and spatial norms should be the same, agnostic of type
        assert (
            pytest.approx(orbit_in_field_basis.norm(), rel=1e-6)
            == fixed_data_transform_norms_dict[name][0]
        )
        assert (
            pytest.approx(orbit_in_spatial_mode_basis.norm(), rel=1e-6)
            == fixed_data_transform_norms_dict[name][1]
        )
        assert (
            pytest.approx(orbit_in_mode_basis.norm(), rel=1e-6)
            == fixed_data_transform_norms_dict[name][2]
        )

        # Apply inverse transforms
        spatial_modes_from_inverse = orbit_in_mode_basis.transform(to="spatial_modes")
        field_from_inverse = spatial_modes_from_inverse.transform(to="field")
        assert (
            pytest.approx(spatial_modes_from_inverse.norm(), rel=1e-6)
            == fixed_data_transform_norms_dict[name][3]
        )
        assert (
            pytest.approx(field_from_inverse.norm(), rel=1e-6)
            == fixed_data_transform_norms_dict[name][4]
        )


def test_instantiation(kse_classes):
    """

    Returns
    -------

    """
    for (name, cls) in kse_classes.items():
        with pytest.raises(ValueError):
            _ = oh.Orbit(state=np.ones(cls.minimal_shape()))
        _ = cls(parameters=(100, 100, 0))
        _ = cls(state=np.ones(cls.minimal_shape()), basis="field")
        _ = cls(
            state=np.ones(cls.minimal_shape()), basis="field", parameters=(100, 100, 0)
        )
    return None


def test_jacobian(
    fixed_OrbitKS_data, fixed_ks_parameters, jacobian_abssums, kse_classes
):
    """
    Returns
    -------
    """
    for jacsum, (name, cls) in zip(jacobian_abssums, kse_classes.items()):
        orbit_ = cls(
            state=fixed_OrbitKS_data, basis="field", parameters=fixed_ks_parameters[0],
        ).transform(to="modes")
        # The jacobians have all other matrices within them; just use this as a proxy to test.
        jac_ = orbit_.jacobian()
        pytest.approx(np.abs(jac_).sum(), jacsum)
