import pytest
import numpy as np
import orbithunter as oh
from math import pi

@pytest.fixture()
def example_orbit_data():
    return np.ones([32, 32])

@pytest.fixture()
def fixed_orbit_data():
    # Generated from
    # np.random.seed(0)
    # np.random.randn(6, 6).round(8); values chosen because of size, truncation error
    state = np.array([[1.76405235,  0.40015721,  0.97873798,  2.2408932, 1.86755799, -0.97727788],
                      [0.95008842, -0.15135721, -0.10321885,  0.4105985, 0.14404357, 1.45427351],
                      [0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.20515826],
                      [0.3130677 , -0.85409574, -2.55298982,  0.6536186 ,  0.8644362, -0.74216502],
                      [2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921, 1.46935877],
                      [0.15494743,  0.37816252, -0.88778575, -1.98079647, -0.34791215, 0.15634897]])
    return state

@pytest.fixture()
def fixed_mode_norm_dict():
    norms = [5.050390311315469, 5.050390311315469, 3.635581670379944,
             3.821906904088631, 1.159165311191572, 2.384651938236806]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))

@pytest.fixture()
def fixed_eqn_norm_dict():
    norms = [1.258363954936136, 1.258363954936136, 0.8876335300535965,
             0.9747965276558949, 0.033697585549483405, 0.37894868060091114]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))

@pytest.fixture()
def fixed_orbit_parameters():
    # Parameter tuples and incorrect input scalar
    return (44, 44, 0.), (44, 44), (44,), 44, None

@pytest.fixture()
def kse_classes():
    return dict([(name, cls) for name, cls in oh.__dict__.items() if isinstance(cls, type)])

def test_orbit(fixed_orbit_data):
    return oh.Orbit(state=fixed_orbit_data, basis='physical', parameters=(36, 36))

def instance_generator(fixed_orbit_data, kse_classes, fixed_orbit_parameters):
    for (name, cls) in kse_classes.items():
        yield cls(state=fixed_orbit_data, basis='field', parameters=fixed_orbit_parameters)


def test_binary_operations(fixed_orbit_data, kse_classes, fixed_orbit_parameters):
    for orbit_instance in instance_generator(fixed_orbit_data, kse_classes, fixed_orbit_parameters):
        # orbit instance is one of the 6 OrbitKS types, all in the 'field' basis.
        save_signage = np.sign(orbit_instance.state)
        assert np.abs((((((2*orbit_instance)**2)**0.5)/2).statemul(save_signage) - orbit_instance).state).sum() == 0.


def test_transforms(fixed_orbit_data, kse_classes, fixed_mode_norm_dict, fixed_orbit_parameters):
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
        orbit_in_field_basis = cls(state=fixed_orbit_data, basis='field')
        orbit_in_spatial_mode_basis = orbit_in_field_basis.transform(to='spatial_modes')
        orbit_in_mode_basis = orbit_in_spatial_mode_basis.transform(to='modes')

        # Using the fixed data, all initial norms and spatial norms should be the same, agnostic of type
        assert pytest.approx(orbit_in_field_basis.norm(), rel=1e-6) == 6.783718210674288
        assert pytest.approx(orbit_in_spatial_mode_basis.norm(), rel=1e-6) == 5.2245770410096055
        assert pytest.approx(orbit_in_mode_basis.norm(), rel=1e-6) == fixed_mode_norm_dict[name]

        # Apply inverse transforms
        spatial_modes_from_inverse = orbit_in_mode_basis.transform(to='spatial_modes')
        field_from_inverse = spatial_modes_from_inverse.transform(to='field')
        assert pytest.approx(spatial_modes_from_inverse.norm(), rel=1e-6) == fixed_mode_norm_dict[name]
        assert pytest.approx(field_from_inverse.norm(), rel=1e-6) == fixed_mode_norm_dict[name]

def test_seeding():
    """

    Returns
    -------

    """
    return None

def test_instantiation():
    """

    Returns
    -------

    """
    for (name, cls) in kse_classes.items():
        with pytest.raises(ValueError):
            _ = oh.Orbit(state=np.ones(cls.minimal_shape()))
        _ = cls(parameters=(100, 100, 0))
        _ = cls(state=np.ones(cls.minimal_shape()), basis='field')
        _ = cls(state=np.ones(cls.minimal_shape()), basis='field', parameters=(100, 100, 0))
    return None

def test_converge():


def test_continuation():
    """

    Returns
    -------

    """
    pass

def test_glue():
    """

    Returns
    -------

    """
    pass

def test_clipping():
    """

    Returns
    -------

    """
    orbit_ = test_orbit(fixed_orbit_data)
    test_clipping = oh.clip(orbit_, )
    pass

def test_io():
    pass