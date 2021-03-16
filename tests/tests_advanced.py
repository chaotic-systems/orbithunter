import pytest
import numpy as np
from .context import orbithunter as oh
import h5py

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

def test_glue():
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

