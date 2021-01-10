import pytest
import numpy as np
from orbithunter import *
from math import pi

@pytest.fixture()
def example_orbit_data():
    return np.ones([32, 32])

@pytest.fixture()
def example_orbit_data():
    t, x = np.meshgrid(np.linspace(0, 2*pi, 32), np.linspace(0, 6*pi, 32))
    return np.cos(t) + 2 * np.sin(x)


def test_binary_operations():
    x = OrbitKS()
    y = OrbitKS()
    return

def test_transform():
    x = OrbitKS()
    y = OrbitKS()
    return

def test_seeding():
    """

    Returns
    -------

    """
    # Seed only affects state and parameters; not the instance itself.
    orbit = OrbitKS(seed=0)
    orbit_again = OrbitKS(seed=0)

    antisymmetric = AntisymmetricOrbitKS(seed=1)
    shift_reflection = ShiftReflectionOrbitKS(seed=2)
    relative = RelativeOrbitKS(seed=3)
    relative_equilibrium = RelativeEquilibriumOrbitKS(seed=4)
    equilibrium = EquilibriumOrbitKS(seed=5)
    return None

def test_instantiation():
    """

    Returns
    -------

    """
    orbit = OrbitKS(seed=0)
    antisymmetric = AntisymmetricOrbitKS(seed=1)
    shift_reflection = ShiftReflectionOrbitKS(seed=2)
    relative = RelativeOrbitKS(seed=3)
    relative_equilibrium = RelativeEquilibriumOrbitKS(seed=4)
    equilibrium = EquilibriumOrbitKS(seed=5)
    return None

def test_converge():
    # TODO : Compare the output of converge
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

def test_clipping():
    """

    Returns
    -------

    """
    pass

def test_io():
    pass