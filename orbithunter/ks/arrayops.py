import numpy as np
from functools import lru_cache
from math import pi
from scipy.fft import rfftfreq

__all__ = ['swap_modes', 'so2_generator', 'so2_coefficients', 'elementwise_dtn', 'elementwise_dxn',
           'calculate_spatial_shift', 'dxn_block', 'dtn_block', 'spatial_frequencies', 'temporal_frequencies']


def swap_modes(modes, axis=1):
    """ Function which swaps halves of arrays for SO(2) differentiation purposes"""
    if axis == 1:
        m = modes.shape[1]//2
        t_dim = modes.shape[0]
        swapped_modes = np.concatenate((modes[:, -m:].reshape(t_dim, -1), modes[:, :-m].reshape(t_dim, -1)), axis=1)
    else:
        n = (modes.shape[0]+1)//2 - 1
        # do not need the special case as .dt() shouldn't be used for either subclass mentioned above.
        swapped_modes = np.concatenate((modes[0, :].reshape(1, -1), modes[-n:, :], modes[1:-n, :]), axis=0)
    return swapped_modes


@lru_cache()
def so2_generator(order=1):
    """ Generator of the SO(2) Lie algebra """
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), np.mod(order, 4))


@lru_cache()
def so2_coefficients(order=1):
    """ Non-zero elements of the Lie algebra generator to the order-th power"""
    return np.sum(so2_generator(order=order), axis=0)


@lru_cache()
def spatial_frequencies(L, M, order=1):
    """ Array of spatial frequencies that the corresponding Fourier modes represent

    Parameters
    ----------
    L : float
        Spatial dimension
    M : int
        Spatial discretization size
    order : int
        Order of the derivative

    Returns
    -------
    ndarray :
        Array of spatial frequencies of shape (1, m), raised to 'order' power for n-th order derivatives.
    """
    q_k = rfftfreq(M, d=L/(2*pi*M))[1:-1].reshape(1, -1)
    return q_k**order


@lru_cache()
def temporal_frequencies(T, N, order=1):
    """ Array of time frequencies that the corresponding Fourier modes represent

    Parameters
    ----------
    T : float
        Temporal dimension
    N : int
        Temporal discretization size
    order : int
        Order of the derivative

    Returns
    -------
    ndarray
        Temporal frequency array of shape (n, 1)

    Notes
    -----
    Extra factor of '-1' because of how the state is ordered, 0th row corresponds to t=T, last row t=0.
    """
    # the parameter 'd' divides the values of rfftfreq s.t. the result is 2 pi N / T * rfftfreq
    w_j = rfftfreq(N, d=-T/(2*pi*N))[1:-1].reshape(-1, 1)
    return w_j**order


@lru_cache()
def elementwise_dtn(T, N, tiling_dimension, order=1):
    """ Matrix/rank 2 tensor of temporal mode frequencies

    Parameters
    ----------
    T : float
        Temporal period
    N : int
        Temporal discretization size
    tiling_dimension : int
        Number of "copies" of the frequencies so that frequency tensor is the same dimension as mode tensor.
    order : int
        The order of the derivative/power of the frequencies desired.

    Returns
    ----------
    dtn_multipliers : ndarray
        Array of spatial frequencies in the same shape as modes

    Notes
    -----
    Creates and returns a matrix whose elements are the properly ordered temporal frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes (and an addition "mode swap" if the derivative is of odd order)
    is the temporal derivative.

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
    """ Matrix/rank 2 tensor of spatial mode frequencies

    Parameters
    ----------
    L : float
        Spatial period
    M : int
        Spatial discretization size
    tiling_dimension : int
        Number of "copies" of the frequencies so that frequency tensor is the same dimension as mode tensor.
    order : int
        The order of the derivative/power of the frequencies desired.

    Returns
    ----------
    dxn_multipliers : ndarray
        Array of spatial frequencies in the same shape as modes

    Notes
    -----
    Creates and returns a matrix whose elements are the properly ordered spatial frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes (and an addition "mode swap" if the derivative is of odd order)
    is the spatial derivative.

    """
    q = spatial_frequencies(L, M, order=order)
    # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
    c1, c2 = so2_coefficients(order=order)
    # Create elementwise spatial frequency matrix
    dxn_multipliers = np.tile(np.concatenate((c1*q, c2*q), axis=1), (tiling_dimension, 1))
    return dxn_multipliers


@lru_cache()
def dxn_block(L, M, order=1):
    """ Block diagonal matrix of spatial frequencies

    Parameters
    ----------
    L : float
        spatial period
    M : int
        spatial discretization size.
    order : int
        Order of the desired derivative.

    Returns
    -------
    np.ndarray :
        Two dimensional block diagonal array
    Notes
    -----
    This is the SO(2) generator for multiple Fourier modes. Only used in explicit construction of matrices.
    """
    return np.kron(so2_generator(order=order), np.diag(spatial_frequencies(L, M, order=order).ravel()))


@lru_cache()
def dtn_block(T, N, order=1):
    """ Block diagonal matrix of temporal frequencies

    Parameters
    ----------
    T : float
        Temporal period
    N : int
        Temporal discretization size.
    order : int
        Order of the desired derivative.

    Returns
    -------
    np.ndarray :
        Two dimensional block diagonal array
    Notes
    -----
    This is the SO(2) generator for multiple Fourier modes. Only used in explicit construction of matrices.
    """
    return np.kron(so2_generator(order=order), np.diag(temporal_frequencies(T, N, order=order).ravel()))


def calculate_spatial_shift(s_modes, L, **kwargs):
    """ Calculate the phase difference between the spatial modes at t=0 and t=T

    Parameters
    ----------
    s_modes : np.ndarray
        The array of spatial Fourier modes
    L : float
        Spatial period in "physical units" (i.e. not plotting units)
    kwargs :
        n_modes : int
            Number of spatial modes to use in the phase calculation.

    Returns
    -------
    shift : float
        The best approximation for physical->comoving shift for relative periodic solutions.
    """
    m0 = s_modes.shape[1]//2
    modes_included = np.min([kwargs.get('n_modes', m0), m0])
    if -m0 + modes_included == 0:
        space_imag_slice_end = None
    else:
        space_imag_slice_end = -m0 + modes_included
    # slice the spatial modes at t=0 and t=T
    modes_0 = np.concatenate((s_modes[-1, :modes_included], s_modes[-1, -m0:space_imag_slice_end])).ravel()
    modes_T = np.concatenate((s_modes[0, :modes_included], s_modes[0, -m0:space_imag_slice_end])).ravel()
    m = modes_T.size//2
    # This function is used very sparingly, extra imports kept in this scope only.
    # Warnings come from fsolve not converging; only want approximate guess as exact solution won't generally exist
    from scipy.optimize import fsolve
    import warnings
    # If they are close enough to the same point, then shift equals 0
    if np.linalg.norm(modes_0-modes_T) <= 10**-6:
        shift = L / s_modes.shape[1]
    else:
        # Get guess shift from the angle between the vectors
        shift_guess = (L / (2 * pi))*float(np.arccos((np.dot(np.transpose(modes_T), modes_0)
                                           / (np.linalg.norm(modes_T)*np.linalg.norm(modes_0)))))
        # find shift which minimizes the differences at the boundaries.
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
        shift = fsolve(fun_, np.array(shift_guess))[0]
        warnings.resetwarnings()
        # because periodic boundary conditions take modulo; "overstretching" doesn't occur from physical limits.
        shift = np.sign(shift) * np.mod(np.abs(shift), L)
    return shift

