import pytest
import numpy as np
import orbithunter as oh

@pytest.fixture()
def fixed_orbit_data():
    """ Fixed test data to be used to initialize orbits"""
    # Originally generated from
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

def test_generate():
    orbit_ = oh.Orbit()
    x = orbit_.generate(attr='parameters')
    for p in x.parameters:
        assert p is not None
        assert (p >= 0) & (p <= 1)
    y = orbit_.generate(attr='state')
    assert y.size > 0
    assert y.shape != (0, 0, 0, 0)


def test_matrix_methods(fixed_orbit_data):
    """ Numerical methods required for matrix-constructing optimization"""
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    j = orbit_.jacobian()
    assert v.size == orbit_.size + len(orbit_.parameters)



def test_from_numpy(fixed_orbit_data):
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    v = orbit_.orbit_vector()
    assert v.sum() == (orbit_.state.sum() + sum(orbit_.parameters))

    orbit_from_numpy = orbit_.from_numpy_array(v)
    assert (orbit_from_numpy.state==orbit_.state).all()

    orbit_from_numpy_passed_param = orbit_.from_numpy_array(v, parameters=(5,5,5,5))
    assert orbit_from_numpy_passed_param.parameters == (5, 5, 5, 5)


def test_matrix_free_methods(fixed_orbit_data, fixed_kwarg_dict):
    """ Numerical methods required for matrix-free optimization"""
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    f = orbit_.eqn(fixed_kwarg_dict)
    grad = orbit_.cost_function_gradient(fixed_kwarg_dict)
    res = orbit_.residual(fixed_kwarg_dict)
    assert (f.state == np.zeros((2,2,2,2))).all()
    assert (grad.state == np.zeros((2,2,2,2))).all()
    assert res == 0.

    matvec_ = orbit_.matvec(f, kwargs=fixed_kwarg_dict)
    assert matvec_.norm() == 0
    assert matvec_.parameters == orbit_.parameters

    rmatvec_ = orbit_.rmatvec(f, kwargs=fixed_kwarg_dict)
    assert rmatvec_.norm() == 0.
    assert rmatvec_.parameters == (0, 0, 0, 0)

def test_equation_methods(fixed_orbit_data, fixed_kwarg_dict):
    orbit_ = oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))
    f = orbit_.eqn(fixed_kwarg_dict)

    grad = orbit_.cost_function_gradient(fixed_kwarg_dict)
    res = orbit_.residual(fixed_kwarg_dict)
    matvec_ = orbit_.matvec(f, kwargs=fixed_kwarg_dict)
    rmatvec_ = orbit_.rmatvec(grad, kwargs=fixed_kwarg_dict)
    orbit_vector_ = orbit_.orbit_vector()
    # equivalent to addition of state and parameters.
    _ = ((2 * orbit_) - orbit_.increment(orbit_)).norm()
    orbit_from_vector = orbit_.from_numpy_array(orbit_vector_)


def test_properties(fixed_orbit_data):
    orbit_ = test_orbit(fixed_orbit_data)
    _ = orbit_.shape
    _ = orbit_.size
    _ = orbit_.bases()
    _ = orbit_.parameter_labels()
    _ = orbit_.discretization_labels()
    _ = orbit_.dimension_labels()
    _ = orbit_.default_shape()
    _ = orbit_.minimal_shape()
    _ = orbit_.default_parameter_ranges()


def test_discretization_methods(fixed_orbit_data):
    orbit_ = test_orbit(fixed_orbit_data)
    enlarged = orbit_.resize(16, 16, 16, 16)
    shrank = enlarged.resize(orbit_.discretization)
    assert (shrank - orbit_).norm() == 0.
