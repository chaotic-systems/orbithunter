import numpy as np
from functools import lru_cache
__all__ = ['swap_modes', 'so2_generator', 'so2_coefficients']


def swap_modes(modes, dimension='space'):
    if dimension == 'space':
        m = modes.shape[1]//2
        # to account for RelativeEquilibriumKS and EquilibriumKS having axis dimensions [1, 2*m], need
        # to reshape or else there will not be two axes.
        t_dim = modes.shape[0]
        swapped_modes = np.concatenate((modes[:, -m:].reshape(t_dim, -1), modes[:, :-m].reshape(t_dim, -1)), axis=1)
    else:
        n = (modes.shape[0]+1)//2 - 1
        # do not need the special case as .dt() shouldn't be used for either subclass mentioned above.
        swapped_modes = np.concatenate((modes[0, :].reshape(1, -1), modes[-n:, :], modes[1:-n, :]), axis=0)
    return swapped_modes

# @lru_cache(maxsize=None)
def so2_generator(order=1):
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), np.mod(order, 4))

# @lru_cache(maxsize=None)
def so2_coefficients(order=1):
    return np.sum(so2_generator(order=order), axis=0)


