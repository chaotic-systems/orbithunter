import pytest
import numpy as np
import h5py
import orbithunter as oh

@pytest.fixture()
def fixed_orbit_data():
    # Generated from
    # np.random.seed(0)
    # np.random.randn(6, 6).round(8); values chosen because of size, truncation error
    state = np.array([[1.76405235,  0.40015721,  0.97873798,  2.2408932, 1.86755799, -0.97727788],
                      [0.95008842, -0.15135721, -0.10321885,  0.4105985, 0.14404357, 1.45427351],
                      [0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.20515826],
                      [0.3130677, -0.85409574, -2.55298982,  0.6536186 ,  0.8644362, -0.74216502],
                      [2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921, 1.46935877],
                      [0.15494743,  0.37816252, -0.88778575, -1.98079647, -0.34791215, 0.15634897]])
    return state

@pytest.fixture()
def fixed_RelativeOrbitKS():
    return oh.read_h5('../../data/ks/RelativeOrbitKS.h5', 't44p304_x33p280').transform(to='modes')

@pytest.fixture()
def fixed_data_transform_norms_dict():
    orbitks_norms = [6.783718210674288, 5.2245770410096055, 5.050390311315469, 5.050390311315469, 5.050390311315469]
    rel_norms = [6.783718210674288, 5.2245770410096055, 5.050390311315469, 5.050390311315469, 5.050390311315469]
    sr_norms = [6.783718210674288, 5.2245770410096055, 3.635581670379944, 3.6355816703799446, 3.6355816703799437]
    anti_norms = [6.783718210674288, 5.2245770410096055, 3.821906904088631, 3.821906904088631, 3.821906904088631]
    eqv_norms = [6.783718210674288, 5.2245770410096055, 1.159165311191572, 2.8393635399538266, 2.8393635399538266]
    reqv_norms = [6.783718210674288, 5.2245770410096055, 2.384651938236806, 5.841180462819081, 5.8411804628190795]
    norms = [orbitks_norms, rel_norms, sr_norms, anti_norms, eqv_norms, reqv_norms]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))

@pytest.fixture()
def fixed_eqn_norm_dict():
    norms = [1.258363954936136, 1.258363954936136, 0.8876335300535965,
             0.9747965276558949, 0.033697585549483405, 0.37894868060091114]
    names = [name for name, cls in oh.__dict__.items() if isinstance(cls, type)]
    return dict(zip(names, norms))

@pytest.fixture()
def fixed_derivative_norms():
    test_orbit_norms = np.array([1.11206823, 1.07142601, 0.28925289, 0.29652498, 0.08045879,
                                   0.08399731, 0.02282276, 0.02394005, 0.40029903, 1.25836395,
                                   1.10417144])
    test_rpo_norms = np.array([31.63701272,  9.37819755, 31.63947896,  5.11209024, 36.77213848,
                                4.82539334, 51.14116224,  6.45371111, 31.44907051,  0.        ,
                               31.44907051])
    return test_orbit_norms, test_rpo_norms

@pytest.fixture()
def fixed_orbit_parameters():
    # Parameter tuples and incorrect input scalar
    return (44, 44, 0.), (44, 44), (44,), 44, None

@pytest.fixture()
def kse_classes():
    return dict([(name, cls) for name, cls in oh.ks.__dict__.items() if isinstance(cls, type)])


def instance_generator(fixed_orbit_data, kse_classes, fixed_orbit_parameters):
    for (name, cls) in kse_classes.items():
        yield cls(state=fixed_orbit_data, basis='field', parameters=fixed_orbit_parameters)


def test_ks_derivatives(fixed_orbit_data, fixed_derivative_norms, fixed_orbit_parameters):
    # parameters = (44.30438636668926, 33.28035979609304, -5.809692922307713)
    # x = oh.OrbitKS(state=fixed_orbit_data, parameters=x.parameters, basis='field').transform(to='modes')
    # y = oh.RelativeOrbitKS(state=fixed_orbit_data, parameters=y.parameters, basis='field').transform(to='modes')
    orbit_ = oh.OrbitKS(state=fixed_orbit_data,
                        parameters=fixed_orbit_parameters[0], basis='field').transform(to='modes')
    norms = []
    for order in range(1, 5):
        norms.append(orbit_.dx(order).norm())
        norms.append(orbit_.dt(order).norm())
    orbitf = orbit_.transform(to='field')
    norms.append(orbitf.nonlinear(orbitf).norm())
    norms.append(orbit_.eqn().norm())
    norms.append(orbit_._eqn_linear_component().norm())
    assert np.equal(np.array(norms).round(8), fixed_derivative_norms[0].round(8)).all()


def test_ks_derivatives_relativeorbitks(fixed_derivative_norms):
    norms = []
    relorbit_ = oh.read_h5('../../data/ks/RelativeOrbitKS.h5', 't44p304_x33p280').transform(to='modes')
    for order in range(1, 5):
        norms.append(relorbit_.dx(order).norm())
        norms.append(relorbit_.dt(order).norm())
    orbitf = relorbit_.transform(to='field')
    norms.append(orbitf.nonlinear(orbitf).norm())
    norms.append(relorbit_.eqn().norm())
    norms.append(relorbit_._eqn_linear_component().norm())
    assert np.equal(np.array(norms).round(8), fixed_derivative_norms[1].round(8)).all()


def test_rmatvec(fixed_orbit_data, fixed_orbit_parameters):
    relorbit_ = oh.read_h5('../../data/ks/RelativeOrbitKS.h5', 't44p304_x33p280').transform(to='modes')
    assert pytest.approx(relorbit_.rmatvec(relorbit_).norm(), 60.18805016)
    assert pytest.approx(relorbit_.cost_function_gradient(relorbit_).norm(), 0.)

    orbit_ = oh.OrbitKS(state=fixed_orbit_data,
                        parameters=fixed_orbit_parameters[0], basis='field').transform(to='modes')
    assert pytest.approx(orbit_.rmatvec(orbit_).norm(), 1.295386)
    assert pytest.approx(orbit_.cost_function_gradient(orbit_.eqn()).norm(), 1.0501956)


def test_matvec(fixed_orbit_data, fixed_orbit_parameters):
    relorbit_ = oh.read_h5('../../data/ks/RelativeOrbitKS.h5', 't44p304_x33p280').transform(to='field')
    orbit_ = oh.OrbitKS(state=fixed_orbit_data,
                        parameters=fixed_orbit_parameters[0], basis='field').transform(to='modes')

    assert pytest.approx(relorbit_.matvec(relorbit_).norm(), 0.3641827310989932)
    assert pytest.approx(orbit_.matvec(orbit_).norm(), 155.86156902189782)


def test_binary_operations(fixed_orbit_data, kse_classes, fixed_orbit_parameters):
    for orbit_instance in instance_generator(fixed_orbit_data, kse_classes, fixed_orbit_parameters[0]):
        # orbit instance is one of the 6 OrbitKS types, all in the 'field' basis.
        save_signage = np.sign(orbit_instance.state)
        assert np.abs((((((2*orbit_instance)**2)**0.5)/2)*save_signage - orbit_instance).state).sum() == 0.


def test_transforms(fixed_orbit_data, kse_classes, fixed_data_transform_norms_dict, fixed_orbit_parameters):
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
        assert pytest.approx(orbit_in_field_basis.norm(), rel=1e-6) == fixed_data_transform_norms_dict[name][0]
        assert pytest.approx(orbit_in_spatial_mode_basis.norm(), rel=1e-6) == fixed_data_transform_norms_dict[name][1]
        assert pytest.approx(orbit_in_mode_basis.norm(), rel=1e-6) == fixed_data_transform_norms_dict[name][2]

        # Apply inverse transforms
        spatial_modes_from_inverse = orbit_in_mode_basis.transform(to='spatial_modes')
        field_from_inverse = spatial_modes_from_inverse.transform(to='field')
        assert pytest.approx(spatial_modes_from_inverse.norm(), rel=1e-6) == fixed_data_transform_norms_dict[name][3]
        assert pytest.approx(field_from_inverse.norm(), rel=1e-6) == fixed_data_transform_norms_dict[name][4]


def test_seeding():
    """

    Returns
    -------

    """
    return None


def test_instantiation(kse_classes):
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


def test_jacobian(fixed_orbit_data, fixed_orbit_parameters, kse_classes):
    """
    Returns
    -------
    """
    for (name, cls) in kse_classes.items():
        orbit_ = cls(state=fixed_orbit_data, basis='field', parameters=fixed_orbit_parameters[0])
        # The jacobians have all other matrices within them; just use this as a proxy to test.
        jac_ = orbit_.jacobian()
    return None


def test_minimize():
    pass


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


def test_clipping(fixed_orbit_data):
    """

    Returns
    -------

    """
    # orbit_ = test_orbit(fixed_orbit_data)
    # test_clipping = oh.clip(orbit_, )


def test_io():
    pass