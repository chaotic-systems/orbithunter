import numpy as np
from functools import lru_cache
from math import pi
from scipy.fft import rfftfreq

__all__ = ['swap_modes', 'so2_generator', 'so2_coefficients', 'elementwise_dtn', 'elementwise_dxn',
           'calculate_spatial_shift', 'dxn_block', 'dtn_block', 'spatial_frequencies', 'temporal_frequencies']


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


@lru_cache()
def so2_generator(order=1):
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), np.mod(order, 4))


@lru_cache()
def so2_coefficients(order=1):
    return np.sum(so2_generator(order=order), axis=0)


@lru_cache()
def spatial_frequencies(L, M, order=1):
    """ Array of spatial frequencies

    Parameters
    ----------
    dx_parameters :

    Returns
    -------
    ndarray :
        Array of spatial frequencies of shape (1, m), raised to 'order' power for n-th order derivatives.
    """
    q_k = rfftfreq(M, d=L/(2*pi*M))[1:-1].reshape(1, -1)
    return q_k**order


@lru_cache()
def temporal_frequencies(T, N, order=1):
    """
    Returns
    -------
    ndarray
        Temporal frequency array of shape (n, 1)

    Notes
    -----
    Extra factor of '-1' because of how the state is ordered; see __init__ for
    more details.

    """
    # the parameter 'd' divides the values of rfftfreq s.t. the result is 2 pi N / T * rfftfreq
    w_j = rfftfreq(N, d=-T/(2*pi*N))[1:-1].reshape(-1, 1)
    return w_j**order


@lru_cache()
def elementwise_dtn(T, N, tiling_dimension, order=1):
    """ Matrix of temporal mode frequencies

    Creates and returns a matrix whose elements are the properly ordered temporal frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes is equivalent to taking a spatial derivative.

    Parameters
    ----------
    T : float

    N : int

    tiling_dimension :


    order : int
        The order of the derivative, and the according power of the spatial frequencies.

    Returns
    ----------
    dtn_multipliers : ndarray
        Array of spatial frequencies in the same shape as modes
    """

    w = temporal_frequencies(T, N, order=order)
    # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
    c1, c2 = so2_coefficients(order=order)
    # The Nyquist frequency is never included, this is how time frequency modes are ordered.
    # Elementwise product of modes with time frequencies is the spectral derivative.
    dtn_multipliers = np.tile(np.concatenate(([[0]], c1*w, c2*w), axis=0), (1, tiling_dimension))
    return dtn_multipliers


@lru_cache()
def elementwise_dxn(L, M, tiling_dimension, order=1):
    """ Matrix of temporal mode frequencies

    Parameters
    ----------
    L : float

    M : int

    tiling_dimension : int
        The dimension of the mode te

    order : int
        The order of the derivative, and the according power of the spatial frequencies.

    Notes
    -----
    Creates and returns a matrix whose elements are the properly ordered spatial frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes is equivalent to taking a spatial derivative.

    The choice of making this a classmethod with caching, is that

    Returns
    ----------
    dxn_multipliers : ndarray
        Array of spatial frequencies in the same shape as modes
    """
    q = spatial_frequencies(L, M, order=order)
    # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
    c1, c2 = so2_coefficients(order=order)
    # Create elementwise spatial frequency matrix
    dxn_multipliers = np.tile(np.concatenate((c1*q, c2*q), axis=1), (tiling_dimension, 1))
    return dxn_multipliers


@lru_cache()
def dxn_block(L, M, order=1):
    """

    Parameters
    ----------
    L
    M
    tiling_dimension
    order

    Returns
    -------

    """
    return np.kron(so2_generator(order=order), np.diag(spatial_frequencies(L, M, order=order).ravel()))


@lru_cache()
def dtn_block(T, N, order=1):
    """

    Parameters
    ----------
    L
    M
    tiling_dimension
    order

    Returns
    -------

    """
    return np.kron(so2_generator(order=order), np.diag(temporal_frequencies(T, N, order=order).ravel()))


def calculate_spatial_shift(s_modes, L, **kwargs):
    """ Calculate the phase difference between the spatial modes at t=0 and t=T """
    m0 = s_modes.shape[1]//2
    modes_included = np.min([kwargs.get('n_modes', m0), m0])
    if -m0 + modes_included == 0:
        space_imag_slice_end = None
    else:
        space_imag_slice_end = -m0 + modes_included
    modes_0 = np.concatenate((s_modes[-1, :modes_included], s_modes[-1, -m0:space_imag_slice_end])).ravel()
    modes_T = np.concatenate((s_modes[0, :modes_included], s_modes[0, -m0:space_imag_slice_end])).ravel()
    m = modes_T.size//2
    from scipy.optimize import fsolve
    import warnings
    if np.linalg.norm(modes_0-modes_T) <= 10**-6:
        shift = L / s_modes.shape[1]
    else:
        shift_guess = (L / (2 * pi))*float(np.arccos((np.dot(np.transpose(modes_T), modes_0)
                                           / (np.linalg.norm(modes_T)*np.linalg.norm(modes_0)))))
        def fun_(shift):
            thetak = shift * ((2 * pi) / L) * np.arange(1, m+1)
            cosinek = np.cos(thetak)
            sinek = np.sin(thetak)
            rotated_real_modes_T = np.multiply(cosinek,  modes_T[:-m]) + np.multiply(sinek,  modes_T[-m:])
            rotated_imag_modes_T = np.multiply(-sinek,  modes_T[:-m]) + np.multiply(cosinek,  modes_T[-m:])
            rotated_modes = np.concatenate((rotated_real_modes_T, rotated_imag_modes_T))
            return np.linalg.norm(modes_0 - rotated_modes)

        # suppress fsolve's warnings that occur when it stalls; not expecting an exact answer anyway.
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        shift = fsolve(fun_, shift_guess)[0]
        warnings.resetwarnings()
        shift = np.sign(shift) * np.mod(np.abs(shift), L)
    return shift



