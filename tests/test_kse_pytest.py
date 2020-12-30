import pytest
from ..orbithunter import *

@pytest.fixture()
def example_orbit_data():
    return [0, 0]

def test_instantiation():
    orbit = OrbitKS(seed=0)
    antisymmetric = AntisymmetricOrbitKS(seed=1)
    shift_reflection = ShiftReflectionOrbitKS(seed=2)
    relative = RelativeOrbitKS(seed=3)
    relative_equilibrium = RelativeEquilibriumOrbitKS(seed=4)
    equilibrium = EquilibriumOrbitKS(seed=5)
    return None

def test_converge():
    pass

def test_continuation():
    pass

def test_glue():
    pass

def test_clipping():
    pass



