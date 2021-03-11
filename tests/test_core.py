import pytest
import numpy as np
import orbithunter as oh

@pytest.fixture()
def fixed_orbit_data():
    """ Fixed test data to be used to initialize orbits"""
    # Originally populated from
    # np.random.seed(1)
    # np.random.randn(2,2,2,2).round(7)
    state = np.array([[[[ 1.6243454, -0.6117564],
                         [-0.5281718, -1.0729686]],
                        [[ 0.8654076, -2.3015387],
                         [ 1.7448118, -0.7612069]]],
                       [[[ 0.3190391, -0.2493704],
                         [ 1.4621079, -2.0601407]],
                        [[-0.3224172, -0.3840544],
                         [ 1.1337694, -1.0998913]]]])
    return state

@pytest.fixture()
def fixed_kwarg_dict():
    """ Passed to class methods to see if they breakdown upon unrecognized keyword arguments """
    return {'aint': 'nothin', 'gonna': 'break', 'my': 'stride', 'nobodys': 'gonna', 'slow': 'me', 'down,': 'ohno',
            'I': 'got', 'to': 'keep', 'on': 'movin'}


def test_create_orbit(fixed_orbit_data):
    """ Test initialization of an Orbit instance """
    return oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))


def test_attributes(fixed_orbit_data):
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
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
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    test_sum = orbit_ + orbit_
    test_sub = orbit_ - orbit_
    test_mul = orbit_ * orbit_
    test_pow = orbit_ ** 2
    test_div = orbit_ / 2
    test_floor_div = orbit_ // 2


def test_assignment_operators(fixed_orbit_data):
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
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

def test_populate():
    """ Initialization in cases where generated information is desired. Occurs in-place"""
    z = oh.Orbit()
    x = z.populate(attr='parameters')
    for p, op in zip(x.parameters, z.parameters):
        assert p is not None
        assert (p >= 0) & (p <= 1)
        assert p == op

    y = z.populate(attr='state')
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
    choice_list_possible_confused_with_interval = ['am I an', 'interval?']
    choice_list = ['this', 'is', 'the', 'way', 'to', 'select']
    parameter_ranges = {'t': (2, 2), 'x': (100, 200),
                        'y': choice_list_possible_confused_with_interval,
                        'z': choice_list}
    x = oh.Orbit().populate(attr='parameters', parameter_ranges=parameter_ranges)
    assert x.t == 2
    assert (x.x >= 100) and (x.x <= 200)
    assert x.y in choice_list_possible_confused_with_interval
    assert x.z in choice_list
    # bad parameter ranges
    should_raise_error = tuple(choice_list_possible_confused_with_interval)
    bad_parameter_ranges = {'t': (2,), 'x': (100, 200), 'y': should_raise_error, 'z': choice_list}
    with pytest.raises((TypeError, ValueError)):
        oh.Orbit().populate(attr='parameters', parameter_ranges=bad_parameter_ranges)

    pranges_missing_keys = {'t': np.ones([2, 2, 2, 2]), 'z': choice_list}
    oh.Orbit().populate(attr='parameters', parameter_ranges=pranges_missing_keys).parameters

    pranges_missing_keys_bundled_array = {'t': (np.ones([2, 2, 2, 2]),), 'z': choice_list}
     oh.Orbit().populate(attr='parameters', parameter_ranges=pranges_missing_keys)

    # partially populated
    oh.Orbit(parameters=(1, None, 1)).populate(attr='parameters', parameter_ranges=parameter_ranges)

    # partially populated with missing keys
    oh.Orbit(parameters=(0, 0, None)).populate(attr='parameters', parameter_ranges=pranges_missing_keys)


def test_overwriting():
    x = oh.Orbit().populate('state')
    with pytest.raises(ValueError):
        x.populate('state')
    x.populate('state', overwrite=True)
    assert x.parameters is None
    x.populate(overwrite=True)
    assert x.parameters is not None
    old_parameters = x.parameters
    y = x.populate(overwrite=True)
    assert y is x and old_parameters != y.parameters


def test_matrix_methods(fixed_orbit_data):
    """ Numerical methods required for matrix-constructing optimization"""
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    j = orbit_.jacobian()
    assert j.shape == (orbit_.size, orbit_.size + len(orbit_.parameters))

def test_constrain():
    x = oh.Orbit()
    x.constrain(oh.Orbit.parameter_labels())
    assert x.constraints == {'t': True, 'x': True, 'y': True, 'z': True}
    # constraints are "refreshed" upon each call in order to allow for changing without checking values.
    x.constrain('I do not exist')
    assert x.constraints == {'t': False, 'x': False, 'y': False, 'z': False}

    with pytest.raises(TypeError):
        oh.Orbit().constrain(['I am in a list'])


def test_from_numpy(fixed_orbit_data):
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    v = orbit_.orbit_vector()
    assert v.sum() == (orbit_.state.sum() + sum(orbit_.parameters))

    orbit_from_numpy = orbit_.from_numpy_array(v)
    assert (orbit_from_numpy.state == orbit_.state).all()

    orbit_from_numpy_passed_param = orbit_.from_numpy_array(v, parameters=(5, 5, 5, 5))
    assert orbit_from_numpy_passed_param.parameters == (5, 5, 5, 5)


def test_matrix_free_methods(fixed_orbit_data, fixed_kwarg_dict):
    """ Numerical methods required for matrix-free optimization"""
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    f = orbit_.eqn(fixed_kwarg_dict)
    grad = orbit_.cost_function_gradient(fixed_kwarg_dict)
    res = orbit_.residual(fixed_kwarg_dict)
    assert (f.state == np.zeros((2, 2, 2, 2))).all()
    assert (grad.state == np.zeros((2, 2, 2, 2))).all()
    assert res == 0.

    matvec_ = orbit_.matvec(f, kwargs=fixed_kwarg_dict)
    assert matvec_.norm() == 0
    assert matvec_.parameters == orbit_.parameters

    rmatvec_ = orbit_.rmatvec(f, kwargs=fixed_kwarg_dict)
    assert rmatvec_.norm() == 0.
    assert rmatvec_.parameters == (0, 0, 0, 0)

def test_properties(fixed_orbit_data):
    """ Call all properties to check if they are defined. """
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical')
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
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical')
    enlarged = orbit_.resize(16, 16, 16, 16)
    shrank = enlarged.resize(orbit_.discretization)
    assert (shrank.state == orbit_.state).all()


def test_glue_dimensions(fixed_orbit_data):
    """ Test the manner by which new parameter values are generated for gluings"""
    x = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(2, 2, 3, 4))
    y = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 0, 0, -4))
    dimension_tuples = tuple(zip(x.parameters, y.parameters))
    glue_shape = (2, 1, 1, 1)
    assert oh.Orbit.glue_dimensions(dimension_tuples, glue_shape, exclude_nonpositive=True) == (12., 2., 3, 4.)
    assert oh.Orbit.glue_dimensions(dimension_tuples, glue_shape, exclude_nonpositive=False) == (12., 1., 1.5, 0.)
    with pytest.raises((IndexError, ValueError)):
        glue_shape = (2, 1)
        oh.Orbit.glue_dimensions(dimension_tuples, glue_shape, exclude_nonpositive=False)


def test_symmetry(fixed_orbit_data):
    """ Test symmetry operations such as discrete rotations and reflections"""
    x = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(2, 2, 3, 4))
    y = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 0, 0, -4))
    z = x.roll(1, axis=(0, 1, 2, 3))
    w = y.roll(1, axis=0).roll(1, axis=1).roll(1, axis=2).roll(1, axis=3)
    assert (z.state == w.state).all()
    v = x.cell_shift((1, 2, 2, 2), axis=(0, 1, 2, 3))
    u = y.roll((2,1,1,1), axis=(0, 1, 2, 3))
    assert (u.state == v.state).all()
    assert (x[::-1, ::-1, ::-1, ::-1] + x.reflection(signed=True)).state.sum() == 0.
