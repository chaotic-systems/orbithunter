import numpy as np
from functools import lru_cache
from math import pi

__all__ = ['swap_modes', 'so2_generator', 'so2_coefficients', 'calculate_spatial_shift']


def swap_modes(modes, axis=1):
    if axis == 1:
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


@lru_cache(maxsize=8)
def so2_generator(power=1):
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), np.mod(power, 4))


@lru_cache(maxsize=8)
def so2_coefficients(power=1):
    return np.sum(so2_generator(power=power), axis=0)


def calculate_spatial_shift(s_modes, L, **kwargs):
    """ Calculate the phase difference between the spatial modes at t=0 and t=T """
    m0 = s_modes.shape[1]//2
    modes_included = np.min([kwargs.get('n_modes', 1), m0])
    if -m0 + modes_included == 0:
        space_imag_slice_end = None
    else:
        space_imag_slice_end = -m0 + modes_included
    modes_0 = np.concatenate((s_modes[-1, :modes_included], s_modes[-1, -m0:space_imag_slice_end])).ravel()
    modes_T = np.concatenate((s_modes[0, :modes_included], s_modes[0, -m0:space_imag_slice_end])).ravel()
    m = modes_T.size//2
    from scipy.optimize import fsolve
    import warnings
    shift_guess = (L / (2 * pi))*float(np.arccos((np.dot(np.transpose(modes_T), modes_0)
                                   / (np.linalg.norm(modes_T)*np.linalg.norm(modes_0)))))

    def fun_(shift):
        thetak = -1.0 * shift * ((2 * pi) / L) * np.arange(1, m+1)
        cosinek = np.cos(thetak)
        sinek = np.sin(thetak)
        rotated_real_modes_T = np.multiply(cosinek,  modes_T[:-m]) + np.multiply(sinek,  modes_T[-m:])
        rotated_imag_modes_T = np.multiply(-sinek,  modes_T[:-m]) + np.multiply(cosinek,  modes_T[-m:])
        rotated_modes = np.concatenate((rotated_real_modes_T, rotated_imag_modes_T))
        return np.linalg.norm(modes_0 - rotated_modes)

    # suppress fsolve's warnings that occur when it stalls.
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    shift = fsolve(fun_, shift_guess)[0]
    warnings.resetwarnings()

    shift = np.sign(shift) * np.mod(np.abs(shift), L)
    return shift

