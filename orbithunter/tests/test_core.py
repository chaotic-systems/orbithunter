import pytest
import numpy as np
import orbithunter as oh

@pytest.fixture()
def fixed_orbit_data():
    # Generated from
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
    # keyword arguments that do nothing except be passed to methods.
    return {'aint': 'nothin', 'gonna': 'break', 'my': 'stride', 'nobodys': 'gonna', 'slow': 'me', 'down,': 'ohno',
            'I': 'got', 'to': 'keep', 'on': 'movin'}


def test_orbit(fixed_orbit_data):
    # PyTest fixture of data used for testing
    return oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(10, 10, 10, 10))


def test_binary_operations(fixed_orbit_data):
    # Testing the different overloaded binary operators
    orbit_ = test_orbit(fixed_orbit_data)
    test_sum = orbit_ + orbit_
    test_sub = orbit_ - orbit_
    test_mul = orbit_ * orbit_
    test_pow = orbit_ ** 2
    test_div = orbit_ / 2
    test_floor_div = orbit_ // 2

def test_binary_operations_no_state():
    # Testing the different overloaded binary operators
    orbit_ = oh.Orbit(fixed_orbit_data)
    test_sum = orbit_ + orbit_
    test_sub = orbit_ - orbit_
    test_mul = orbit_ * orbit_
    test_pow = orbit_ ** 2
    test_div = orbit_ / 2
    test_floor_div = orbit_ // 2
    # testing binary operators for orbit instances without declaring states.

def test_dunder_methods(fixed_orbit_data):
    orbit_ = test_orbit(fixed_orbit_data)
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

def test_equation_methods(fixed_orbit_data, fixed_kwarg_dict):
    orbit_ = test_orbit(fixed_orbit_data)
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
    large = orbit_.resize(16, 16, 16, 16)
    original = orbit_.resize(orbit_.shape)
    assert (original.state - orbit_.state).norm() == 0.
# def test_numerical_manipulations():
#     # constrain
#     # rescale
#     # glue_parameters
#
# def test_generation():
#     parameters_only = oh.Orbit().generate(attr='parameters')
#     state_only = oh.Orbit().generate(attr='state')
#     all = oh.Orbit().generate()