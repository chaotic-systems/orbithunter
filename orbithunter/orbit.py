from math import pi
from orbithunter.arrayops import swap_modes, so2_generator, so2_coefficients
from orbithunter.discretization import rediscretize
from scipy.fft import rfft, irfft
from scipy.linalg import block_diag
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
import copy
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()

__all__ = ['OrbitKS', 'RelativeOrbitKS', 'ShiftReflectionOrbitKS', 'AntisymmetricOrbitKS', 'EquilibriumOrbitKS',
           'RelativeEquilibriumOrbitKS']


class OrbitKS:
    """ Object that represents invariant 2-torus solution of the Kuramoto-Sivashinsky equation.

    Parameters
    ----------

    state : ndarray(dtype=float, ndim=2)
        Array which contains one of the following: velocity field,
        spatial Fourier modes, or spatiotemporal Fourier modes.
        If None then a randomly generated set of spatiotemporal modes
        will be produced.
    state_type : str
        Which basis the array 'state' is currently in. Takes values
        'field', 's_modes', 'modes'. Needs to reflect the current basis.
    T : float
        The temporal period.
    L : float
        The spatial period.
    S : float
        The spatial shift for doubly-periodic orbits with continuous translation symmetry.
        S represents rotation/translation such that S = S mod L
    **kwargs :
    N : int
        The lattice size in the temporal dimension.
    M : int
        The lattice size in the spatial dimension.

    Raises
    ----------
    ValueError :
        If 'state' is not a NumPy array

    See Also
    --------


    Notes
    -----
    The 'state' is ordered such that when in the physical basis, the last row corresponds to 't=0'. This
    results in an extra negative sign when computing time derivatives. This convention was chosen
    because it is conventional to display positive time as 'up'. This convention prevents errors
    due to flipping fields up and down. The spatial shift parameter only applies to RelativeOrbitKS.
    Its inclusion in the base class is again a convention
    for exporting and importing data. If no state is None then a randomly generated state will be
    provided. It's dimensions will provide on the spatial and temporal periods unless provided
    as keyword arguments {N, M}.


    Examples
    --------
    """

    def __init__(self, state=None, state_type='modes', T=0., L=0., S=0., **kwargs):
        try:
            if state is not None:
                shp = state.shape
                self.state = state
                self.state_type = state_type
                if state_type == 'modes':
                    # This separate behavior is for Antisymmetric and ShiftReflection Tori;
                    # This avoids having to define subclass method with repeated code.
                    self.N, self.M = shp[0] + 1, shp[1] + 2
                elif state_type == 'field':
                    self.N, self.M = shp
                elif state_type == 's_modes':
                    self.N, self.M = shp[0], shp[1] + 2

                self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
                self.T, self.L = T, L
            else:
                self.state_type = 'modes'
                self.random_initial_condition(T, L, **kwargs)
                self.convert(to='field', inplace=True)
                self.state = (self // (1.0/4.0)).state
                self.convert(to='modes', inplace=True)
        except ValueError:
            print('Incompatible type provided for field or modes: 2-D NumPy arrays only')
        self.mode_shape = (self.N-1, self.M-2)
        # For uniform save format
        self.S = S

    def __add__(self, other):
        return self.__class__(state=(self.state + other.state),
                              T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __radd__(self, other):
        return self.__class__(state=(self.state + other.state),
                              T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __sub__(self, other):
        return self.__class__(state=(self.state-other.state),  T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __rsub__(self, other):
        return self.__class__(state=(other.state - self.state), T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __mul__(self, num):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        Notes
        -----
        This is not state-wise multiplication because that it more complicated and depends on symmetry type.
        Current implementation makes it so no checks of the type of num need to be made.
        """
        return self.__class__(state=num*self.state,  T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __rmul__(self, num):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        Notes
        -----
        This is not state-wise multiplication because that it more complicated and depends on symmetry type.
        Current implementation makes it so no checks of the type of num need to be made.
        """
        return self.__class__(state=num*self.state,  T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __truediv__(self, num):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to division by.

        Notes
        -----
        State-wise division is ill-defined because of division by 0.
        """
        return self.__class__(state=self.state / num, T=self.T, L=self.L, S=self.S, state_type=self.state_type)

    def __floordiv__(self, num):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to renormalize by.

        Notes
        -----
        This renormalizes the physical field such that the absolute value of the max/min takes on a new value
        of (1.0 / num).

        Examples
        --------
        >>> renormalized_orbit = self // (1.0/2.0)
        >>> print(np.max(np.abs(renormalized_orbit.state.ravel())))
        2.0
        """
        field = self.convert(to='field').state
        state = field / (num * np.max(np.abs(field.ravel())))
        return self.__class__(state=state, state_type='field', T=self.T, L=self.L, S=self.S)

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __getattr__(self, attr):
        # Only called if self.attr is not found.
        try:
            if str(attr) in ['T', 'L', 'S']:
                return 0.0
            elif str(attr) == 'state':
                return None
            else:
                error_message = ' '.join([self.__class__.__name__, 'has no attribute called \' {} \''.format(attr)])
                raise AttributeError(error_message)
        except ValueError:
            print('Attribute is not of readable type')

    def state_vector(self):
        """ Vector which completely specifies the orbit """
        return np.concatenate((self.state.reshape(-1, 1),
                               np.array([[float(self.T)]]),
                               np.array([[float(self.L)]])), axis=0)

    def status(self):
        """ Check whether the orbit converged to an equilibrium or close-to-zero solution """
        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        zero_check = np.linalg.norm(self.convert(to='field').state.ravel())
        # Calculate the time derivative
        equilibrium_modes = self.dt().state
        # Equilibrium have non-zero zeroth modes in time, exclude these from the norm.
        equilibrium_check = np.linalg.norm(equilibrium_modes[1:, :])

        # See if the L_2 norm is beneath a threshold value, if so, replace with zeros.
        if zero_check < self.N*self.M*10**-10:
            code = 4
            return self.__class__(state=np.zeros([self.N, self.M], state_type='field'),
                                  state_type='field', T=self.T, L=self.L, S=self.S), code
        # Equilibrium is defined by having no temporal variation, i.e. time derivative is a uniformly zero.
        elif equilibrium_check < self.N*self.M*10**-10:
            # If there is sufficient evidence that solution is an equilibrium, change its class
            code = 3
            return self.__class__(state=equilibrium_modes, T=self.T, L=self.L, S=self.S), code

        else:
            return self, 1

    def convert(self, inplace=False, to='modes'):
        """ Convert current state to a different basis.

        This instance method is just a wrapper for different
        Fourier transforms. It's purpose is to remove the
        need for the user to keep track of the state_type by hand.
        This should be used as opposed to Fourier transforms.

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the conversion "in place" or not.
        to : str
            One of the following: 'field', 's_modes', 'modes'. Specifies
            the basis which the orbit will be converted to. \

        Raises
        ----------

        ValueError
            Raised if the provided basis is unrecognizable.

        Returns
        ----------

        converted_orbit : orbit or orbit subclass instance
            The class instance in the new basis.
        """
        if to == 'field':
            if self.state_type == 's_modes':
                converted_orbit = self.inv_space_transform()
            elif self.state_type == 'modes':
                converted_orbit = self.inv_spacetime_transform()
            else:
                converted_orbit = self
        elif to == 's_modes':
            if self.state_type == 'field':
                converted_orbit = self.space_transform()
            elif self.state_type == 'modes':
                converted_orbit = self.inv_time_transform()
            else:
                converted_orbit = self
        elif to == 'modes':
            if self.state_type == 's_modes':
                converted_orbit = self.time_transform()
            elif self.state_type == 'field':
                converted_orbit= self.spacetime_transform()
            else:
                converted_orbit = self
        else:
            raise ValueError('Trying to convert to unrecognizable state type.')

        if inplace:
            self.state = converted_orbit.state
            self.state_type = to
            return self
        else:
            return converted_orbit

    def copy(self):
        return self.__class__(state=self.state, state_type=self.state_type, T=self.T, L=self.L, S=self.S)

    def dot(self, other):
        """ Return the L_2 inner product of two 2-tori

        Returns
        -------
        float :
            The value of self * other via L_2 inner product.
        """
        return float(np.dot(self.state.ravel(), other.state.ravel()))

    def dt_matrix(self, power=1):
        """ The time derivative matrix operator for the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        wk_matrix : matrix
            The operator whose matrix-vector product with spatiotemporal
            Fourier modes is equal to the time derivative. Used in
            the construction of the Jacobian operator.

        Notes
        -----
        Before the kronecker product, the matrix dt_n_matrix is the operator which would correctly take the
        time derivative of a single set of N-1 temporal modes. Because we have space as an extra dimension,
        we need a number of copies of dt_n_matrix equal to the number of spatial frequencies.
        """
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        dt_n_matrix = np.kron(so2_generator(power=power), np.diag(self.frequency_vector(self.parameters,
                                                                                        power=power).ravel()))
        # Zeroth frequency was not included in frequency vector.
        dt_n_matrix = block_diag([[0]], dt_n_matrix)
        # Take kronecker product to account for the number of spatial modes.
        spacetime_dtn = np.kron(dt_n_matrix, np.eye(self.mode_shape[1]))
        return spacetime_dtn

    def dt(self, power=1, return_modes=False):
        """ A time derivative of the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        orbit_dtn : OrbitKS or subclass instance
            The class instance whose state is the time derivative in
            the spatiotemporal mode basis.
        """
        modes = self.convert(to='modes').state

        # Elementwise multiplication of modes with frequencies, this is the derivative.
        dtn_modes = np.multiply(self.elementwise_dtn(self.parameters, power=power), modes)

        # If the order of the derivative is odd, then imaginary component and real components switch.
        if np.mod(power, 2):
            dtn_modes = swap_modes(dtn_modes, dimension='time')

        if return_modes:
            return dtn_modes
        else:
            orbit_dtn = self.__class__(state=dtn_modes, state_type='modes', T=self.T, L=self.L, S=self.S)
            return orbit_dtn.convert(to=self.state_type)

    def dx(self, power=1):
        """ A spatial derivative of the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        orbit_dxn : OrbitKS or subclass instance
            Class instance whose spatiotemporal state represents the spatial derivative in the
            the basis of the original state.

        Notes
        -----
        Dimensions provided to np.tile in defining elementwise_dxn were originally (self.N-1, 1).
        """
        modes = self.convert(to='modes').state

        # Elementwise multiplication of modes with frequencies, this is the derivative.
        dxn_modes = np.multiply(self.elementwise_dxn(self.parameters, power=power), modes)

        # If the order of the differentiation is odd, need to swap imaginary and real components.
        if np.mod(power, 2):
            dxn_modes = swap_modes(dxn_modes, dimension='space')

        orbit_dxn = self.__class__(state=dxn_modes, state_type='modes', T=self.T, L=self.L, S=self.S)
        return orbit_dxn.convert(to=self.state_type)

    @classmethod
    @lru_cache(maxsize=16)
    def wave_vector(cls, parameters, power=1):
        """ Spatial frequency vector for the current state

        Returns
        -------
        ndarray :
            Array of spatial frequencies of shape (m, 1)
        """
        L, M, m = parameters[1], parameters[4], parameters[6]
        q_m = ((2 * pi * M / L) * np.fft.fftfreq(M)[1:m+1]).reshape(1, -1)
        return q_m**power

    @classmethod
    @lru_cache(maxsize=16)
    def frequency_vector(cls, parameters, power=1):
        """
        Returns
        -------
        ndarray
            Temporal frequency array of shape {n, 1}

        Notes
        -----
        Extra factor of '-1' because of how the state is ordered; see __init__ for
        more details.

        """
        T, N, n = parameters[0], parameters[3], parameters[5]
        w_n = (-1.0 * (2 * pi * N / T) * np.fft.fftfreq(N)[1:n+1]).reshape(-1, 1)
        return w_n**power

    def dx_matrix(self, power=1, **kwargs):
        """ The space derivative matrix operator for the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.
        **kwargs :
            state_type: str
            The basis the current state is in, can be 'modes', 's_modes'
            or 'field'. (spatiotemporal modes, spatial_modes, or velocity field, respectively).
        Returns
        ----------
        spacetime_dxn : matrix
            The operator whose matrix-vector product with spatiotemporal
            Fourier modes is equal to the time derivative. Used in
            the construction of the Jacobian operator.

        Notes
        -----
        Before the kronecker product, the matrix space_dxn is the operator which would correctly take the
        time derivative of a set of M-2 spatial modes (technically M/2-1 real + M/2-1 imaginary components).
        Because we have time as an extra dimension, we need a number of copies of
        space_dxn equal to the number of temporal frequencies. If in the spatial mode basis, this is the
        number of time points instead.
        """

        state_type = kwargs.get('state_type', self.state_type)
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        space_dxn = np.kron(so2_generator(power=power), np.diag(self.wave_vector(self.parameters,
                                                                                 power=power).ravel()))
        if state_type == 'modes':
            spacetime_dxn = np.kron(np.eye(self.mode_shape[0]), space_dxn)
        else:
            spacetime_dxn = np.kron(np.eye(self.N), space_dxn)

        return spacetime_dxn

    @classmethod
    @lru_cache(maxsize=16)
    def elementwise_dxn(cls, parameters, power=1):
        """ Matrix of temporal mode frequencies

        Creates and returns a matrix whose elements
        are the properly ordered spatial frequencies,
        which is the same shape as the spatiotemporal
        Fourier mode state. The elementwise product
        with a set of spatiotemporal Fourier modes
        is equivalent to taking a spatial derivative.

        Returns
        ----------
        matrix
            Matrix of spatial frequencies
        """
        q = cls.wave_vector(parameters, power=power)
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        c1, c2 = so2_coefficients(power=power)
        # Create elementwise spatial frequency matrix
        N = parameters[3]
        dxn_multipliers = np.tile(np.concatenate((c1*q, c2*q), axis=1), (N - 1, 1))
        return dxn_multipliers

    @classmethod
    @lru_cache(maxsize=16)
    def elementwise_dtn(cls, parameters, power=1):
        """ Matrix of temporal mode frequencies

        Creates and returns a matrix whose elements
        are the properly ordered temporal frequencies,
        which is the same shape as the spatiotemporal
        Fourier mode state. The elementwise product
        with a set of spatiotemporal Fourier modes
        is equivalent to taking a time derivative.


        Returns
        ----------
        matrix
            Matrix of temporal frequencies
        """

        w = cls.frequency_vector(parameters, power=power)
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        c1, c2 = so2_coefficients(power=power)
        # The Nyquist frequency is never included, this is how time frequency modes are ordered.
        # Elementwise product of modes with time frequencies is the spectral derivative.
        M = parameters[4]
        dtn_multipliers = np.tile(np.concatenate(([[0]], c1*w, c2*w), axis=0), (1, M - 2))
        return dtn_multipliers

    def from_fundamental_domain(self):
        """ This is a placeholder for the subclasses """
        return self

    def increment(self, other, stepsize=1):
        """ Add optimization correction  to current state

        Parameters
        ----------
        other : OrbitKS
            Represents the values to increment by.
        stepsize : float
            Multiplicative factor which decides the step length of the correction.

        Returns
        -------
        OrbitKS
            New instance which results from adding an optimization correction to self.
        """
        return self.__class__(state=self.state+stepsize*other.state, state_type=self.state_type,
                              T=self.T+stepsize*other.T, L=self.L+stepsize*other.L, S=self.S+stepsize*other.S)

    def jacobian(self, parameter_constraints=(False, False)):
        """ Jacobian matrix evaluated at the current state.
        Parameters
        ----------
        parameter_constraints : tuple of bools
            Determines whether to include period and spatial period
            as variables.
        Returns
        -------
        jac_ : matrix ((N-1)*(M-2), (N-1)*(M-2) + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(parameter_constraints)
        """
        # The Jacobian components for the spatiotemporal Fourier modes
        jac_ = self.jac_lin() + self.jac_nonlin()

        # If period is not fixed, need to include dF/dT for changes to period.
        if not parameter_constraints[0]:
            # Derivative with respect to T of the time derivative term, equal to -1/T u_t
            dt = swap_modes(np.multiply(self.elementwise_dtn(self.parameters), self.state), dimension='time')
            dfdt = (-1.0 / self.T)*dt.reshape(1, -1)
            jac_ = np.concatenate((jac_, dfdt.reshape(-1, 1)), axis=1)

        # If period is not fixed, need to include dF/dL for changes to period.
        if not parameter_constraints[1]:
            orbit_field = self.convert(to='field')
            qk_matrix = self.elementwise_dxn(self.parameters)

            # The derivative with respect to L of the linear component. Equal to -2/L u_xx - 4/L u_xxxx
            dx2 = np.multiply(-1.0 * qk_matrix**2, self.state)
            dx4 = np.multiply(qk_matrix**4, self.state)
            dfdl_linear = (-2.0/self.L) * dx2 + (-4.0/self.L) * dx4

            # The derivative with respect to L of the nonlinear component. Equal to -1/L (0.5 (u^2)_x)
            dfdl_nonlinear = (- 1.0 / self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear

            # Augment the Jacobian with the partial derivative
            jac_ = np.concatenate((jac_, dfdl.reshape(-1, 1)), axis=1)

        return jac_

    def jac_lin(self):
        """ The linear component of the Jacobian matrix of the Kuramoto-Sivashinsky equation"""
        return self.dt_matrix() + self.dx_matrix(power=2) + self.dx_matrix(power=4)

    def jac_nonlin(self):
        """ The nonlinear component of the Jacobian matrix of the Kuramoto-Sivashinsky equation

        Returns
        -------
        nonlinear_dx : matrix
            Matrix which represents the nonlinear component of the Jacobian. The derivative of
            the nonlinear term, which is
            (D/DU) 1/2 d_x (u .* u) = (D/DU) 1/2 d_x F (diag(F^-1 u)^2)  = d_x F( diag(F^-1 u) F^-1).
            See
            Chu, K.T. A direct matrix method for computing analytical Jacobians of discretized nonlinear
            integro-differential equations. J. Comp. Phys. 2009
            for details.
        """
        nonlinear = np.dot(np.diag(self.inv_spacetime_transform().state.ravel()), self.inv_spacetime_transform_matrix())
        nonlinear_dx = np.dot(self.time_transform_matrix(),
                              np.dot(self.dx_matrix(state_type='s_modes'),
                                     np.dot(self.space_transform_matrix(), nonlinear)))
        return nonlinear_dx

    def norm(self, order=None):
        """ Norm of spatiotemporal state via numpy.linalg.norm

        Example
        -------
        L_2 distance between two states
        >>> (self - other).norm()
        """
        return np.linalg.norm(self.state.ravel(), ord=order)

    def matvec(self, other, parameter_constraints=(False, False), preconditioning=True):
        """ Matrix-vector product of a vector with the Jacobian of the current state.

        Parameters
        ----------
        other : OrbitKS
            OrbitKS instance whose state represents the vector in the matrix-vector multiplication.
        parameter_constraints : tuple of bool
            Determines whether to include period and spatial period
            as variables.
        preconditioning : bool
            Whether or not to apply (left) preconditioning P (Ax)

        Returns
        -------
        OrbitKS :
            OrbitKS whose state and other parameters result from the matrix-vector product.

        Notes
        -----
        Equivalent to computation of v_t + v_xx + v_xxxx + d_x (u .* v)

        """
        assert (self.state_type == 'modes') and (other.state_type == 'modes')
        modes = self.state
        other_modes = other.state
        elementwise_dt = self.elementwise_dtn(self.parameters)
        elementwise_dx2 = self.elementwise_dxn(self.parameters, power=2)
        elementwise_dx4 = self.elementwise_dxn(self.parameters, power=4)

        dt = swap_modes(np.multiply(elementwise_dt, other_modes), dimension='time')
        dx2 = np.multiply(elementwise_dx2, other_modes)
        dx4 = np.multiply(elementwise_dx4, other_modes)
        linear = dt + dx2 + dx4

        # Compute nonlinear term
        field = self.convert(to='field')
        other_field = other.convert(to='field')
        nonlinear = swap_modes(np.multiply(self.elementwise_dxn(self.parameters),
                                field.statemul(other_field).convert(to='modes').state), dimension='space')

        matvec_modes = linear + nonlinear
        if not parameter_constraints[0]:
            # Compute the product of the partial derivative with respect to T with the vector's value of T.
            # This is typically an incremental value dT.
            matvec_modes += other.T * (-1.0 / self.T) * dt

        if not parameter_constraints[1]:
            # Compute the product of the partial derivative with respect to L with the vector's value of L.
            # This is typically an incremental value dL.
            dx2_self = np.multiply(elementwise_dx2, modes)
            dx4_self = np.multiply(elementwise_dx4, modes)
            dfdl_linear = (-2.0/self.L)*dx2_self + (-4.0/self.L)*dx4_self
            dfdl_nonlinear = (-1.0/self.L) * 0.5 * swap_modes(np.multiply(self.elementwise_dxn(self.parameters),
                                field.statemul(field).convert(to='modes').state), dimension='space')
            dfdl = dfdl_linear + dfdl_nonlinear
            matvec_modes += other.L * dfdl

        # This is equivalent to LEFT preconditioning.
        if preconditioning:
            p_matrix = 1.0 / (np.abs(elementwise_dt) + np.abs(elementwise_dx2) + elementwise_dx4)
            matvec_modes = np.multiply(matvec_modes, p_matrix)

        return self.__class__(state=matvec_modes, state_type='modes', T=self.T, L=self.L)

    def mode_padding(self, size, dimension='space'):
        """ Increase the size of the discretization via zero-padding

        Parameters
        ----------
        size : int
            The new size of the discretization, must be an even integer
            larger than the current size of the discretization.
        dimension : str
            Takes values 'space' or 'time'. The dimension that will be padded.

        Returns
        -------
        OrbitKS :
            OrbitKS instance with larger discretization.
        """

        if dimension == 'time':
            # Split into real and imaginary components, pad separately.
            first_half = self.state[:-self.n, :]
            second_half = self.state[-self.n:, :]
            padding_number = int((size-self.N) // 2)
            padding = np.zeros([padding_number, self.state.shape[1]])
            padded_modes = np.concatenate((first_half, padding, second_half, padding), axis=0)
        else:
            # Split into real and imaginary components, pad separately.
            first_half = self.state[:, :-self.m]
            second_half = self.state[:, -self.m:]
            padding_number = int((size-self.M) // 2)
            padding = np.zeros([self.state.shape[0], padding_number])
            padded_modes = np.concatenate((first_half, padding, second_half, padding), axis=1)
        return self.__class__(state=padded_modes, state_type=self.state_type, T=self.T, L=self.L, S=self.S)

    def mode_truncation(self, size, dimension='space'):
        """ Decrease the size of the discretization via truncation

        Parameters
        -----------
        size : int
            The new size of the discretization, must be an even integer
            smaller than the current size of the discretization.
        dimension : str
            Takes values 'space' or 'time'. The dimension that will be truncated.

        Returns
        -------
        OrbitKS
            OrbitKS instance with larger discretization.
        """
        if dimension == 'time':
            truncate_number = int(size // 2) - 1
            # Split into real and imaginary components, truncate separately.
            first_half = self.state[:truncate_number+1, :]
            second_half = self.state[-self.n:-self.n+truncate_number, :]
            truncated_modes = np.concatenate((first_half, second_half), axis=0)
        else:
            truncate_number = int(size // 2) - 1
            # Split into real and imaginary components, truncate separately.
            first_half = self.state[:, :truncate_number]
            second_half = self.state[:, -self.m:-self.m + truncate_number]
            truncated_modes = np.concatenate((first_half, second_half), axis=1)
        return self.__class__(state=truncated_modes, state_type=self.state_type, T=self.T, L=self.L, S=self.S)

    @property
    def parameters(self):
        return (self.T, self.L, self.S, self.N, self.M, self.n, self.m)

    def parameter_dependent_filename(self, extension='.h5', decimals=3):

        Lsplit = str(self.L).split('.')
        Lint = str(Lsplit[0])
        Ldec = str(Lsplit[1])
        Lname = ''.join([Lint, 'p', Ldec[:decimals]])

        Tsplit = str(self.T).split('.')
        Tint = str(int(Tsplit[0]))
        Tdec = str(int(Tsplit[1]))
        Tname = ''.join([Tint, 'p', Tdec[:decimals]])

        save_filename = ''.join([self.__class__.__name__, '_L', Lname, '_T', Tname, extension])
        return save_filename

    def plot(self, show=True, save=False, padding=True, fundamental_domain=True, **kwargs):
        """ Plot the velocity field as a 2-d density plot using matplotlib's imshow

        Parameters
        ----------
        show : bool
            Whether or not to display the figure
        save : bool
            Whether to save the figure
        padding : bool
            Whether to interpolate with zero padding before plotting. (Increases the effective resolution).
        fundamental_domain : bool
            Whether to plot only the fundamental domain or not.
        **kwargs :
            newN : int
                Even integer for the new discretization size in time
            newM : int
                Even integer for the new discretization size in space.
            filename : str
                The save name of the figure, if save==True
            directory : str
                The location to save to, if save==True
        Notes
        -----
        newN and newM are accessed via .get() because this is the only manner in which to incorporate
        the current N and M values as defaults.

        """
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams.update({'font.size': 16})
        verbose = kwargs.get('verbose', False)

        if padding:
            pad_n, pad_m = kwargs.get('newN', 16*self.N), kwargs.get('newM', 16*self.M)
            plot_orbit_tmp = rediscretize(self, newN=pad_n, newM=pad_m)
        else:
            plot_orbit_tmp = self

        # The following creates custom tick labels and accounts for some pathological cases
        # where the period is too small (only a single label) or too large (many labels, overlapping due
        # to font size) Default label tick size is 10 for time and the fundamental frequency, 2 pi sqrt(2) for space.
        if fundamental_domain:
            orbit_to_plot = plot_orbit_tmp.to_fundamental_domain().convert(to='field')
        else:
            orbit_to_plot = plot_orbit_tmp.convert(to='field')

        if orbit_to_plot.T != 0:
            timetick_step = np.min([10, 10 * (2**(int(np.log10(orbit_to_plot.T))))])
            yticks = np.arange(timetick_step, orbit_to_plot.T, timetick_step)
            ylabels = np.array([str(int(y)) for y in yticks])
        else:
            orbit_to_plot.T = 1
            yticks = np.array([0, orbit_to_plot.T])
            ylabels = np.array(['0', 'inf'])

        if orbit_to_plot.L > 2*pi:
            xticks = np.arange(0, orbit_to_plot.L, 2*pi)
            xlabels = [str(int(x/(2*pi))) for x in xticks]
        elif orbit_to_plot.L > pi:
            xticks = np.arange(0, orbit_to_plot.L, pi)
            xlabels = [str(int(x/pi)) for x in xticks]
        else:
            orbit_to_plot.L = 1
            xticks = np.array([0, orbit_to_plot.L])
            xlabels = np.array(['0', 'inf'])

        # Modify the size so that relative sizes between different figures is approximately representative
        # of the different sizes; helps with side-by-side comparison.
        default_figsize = (max([2, 2**np.log10(orbit_to_plot.L)]), max([2, 2**np.log10(orbit_to_plot.T)]))
        figsize = kwargs.get('figsize', default_figsize)

        fig, ax = plt.subplots(figsize=figsize)
        # plot the field
        image = ax.imshow(orbit_to_plot.state, extent=[0, orbit_to_plot.L, 0, orbit_to_plot.T], cmap='jet')
        # Include custom ticks and tick labels
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels, fontsize=12)
        ax.set_yticklabels(ylabels, fontsize=12)
        ax.grid(True, linestyle=':', color='k', alpha=0.8)
        fig.subplots_adjust(right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.1, pad=0.02)

        # Custom colorbar values
        maxu = np.max(orbit_to_plot.state.ravel())
        minu = np.min(orbit_to_plot.state.ravel())
        plt.colorbar(image, cax=cax, ticks=[round(minu, 1) + 0.1, round(maxu, 1)-0.1])
        plt.tight_layout()

        if save:
            filename = kwargs.get('filename', None)
            directory = kwargs.get('directory', '')

            # Create save name if one doesn't exist.
            if filename is None:
                filename = self.parameter_dependent_filename(extension='.png')
            elif filename.endswith('.h5'):
                filename = filename.split('.h5')[0] + '.png'

            # Create save directory if one doesn't exist.
            if directory is None:
                pass
            else:
                if directory == 'default':
                    directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../figs/')), '')
                elif directory == '':
                    pass
                elif not os.path.isdir(directory):
                    warnings.warn('Trying to save figure to a directory that does not exist:' + directory, Warning)
                    sys.stdout.flush()
                    proceed = input('Would you like to create this directory? [y]/[n]')
                    if proceed == 'y':
                        os.mkdir(directory)
                    else:
                        directory = ''

                filename = os.path.join(directory, filename)
            if verbose:
                print('Saving figure to {}'.format(filename))
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

        if show:
            plt.show()
        else:
            plt.close()

        return None

    def precondition(self, target, parameter_constraints=(False, False)):
        """ Precondition a vector with the inverse (aboslute value) of linear spatial terms

        Parameters
        ----------

        target : OrbitKS
            OrbitKS to precondition
        parameter_constraints : (bool, bool)
            Whether or not period T or spatial period L are fixed.

        Returns
        -------
        target : OrbitKS
            Return the OrbitKS instance, modified by preconditioning.
        """
        qk_matrix = self.elementwise_dxn(self.parameters)
        p_matrix = np.abs(self.elementwise_dtn(self.parameters)) + qk_matrix**2 + qk_matrix**4
        target.state = np.divide(target.state, p_matrix)

        # Precondition the change in T and L so that they do not dominate
        if not parameter_constraints[0]:
            target.T = target.T / self.T
        if not parameter_constraints[1]:
            target.L = target.L / (self.L**4)

        return target

    def preconditioner(self, parameter_constraints=(False, False), side='left'):
        """ Preconditioning matrix

        Parameters
        ----------
        parameter_constraints : (bool, bool)
            Whether or not period T or spatial period L are fixed.
        side : str
            Takes values 'left' or 'right'. This is an accomodation for
            the typically rectangular Jacobian matrix.

        Returns
        -------
        matrix :
            Preconditioning matrix

        """
        # Preconditioner is the inverse of the aboslute value of the linear spatial derivative operators.
        qk_matrix = self.elementwise_dxn(self.parameters)
        ptmp = 1 / (np.abs(self.elementwise_dtn(self.parameters)) + qk_matrix**2 + qk_matrix**4)
        p = ptmp.ravel()
        parameters = []
        # If including parameters, need an extra diagonal matrix to account for this (right-side preconditioning)
        if side == 'right':
            if not parameter_constraints[0]:
                parameters.append(1 / self.T)
            if not parameter_constraints[1]:
                parameters.append(1 / (self.L**4))
            parameters = np.array(parameters).reshape(1, -1)
            p_row = np.concatenate((p.reshape(1, -1), parameters.reshape(1, -1)), axis=1)
            return np.tile(p_row, (p.size, 1))
        else:
            return np.tile(p.reshape(-1, 1), (1, p.size+(2-sum(parameter_constraints))))

    def pseudospectral(self, other):
        """ nonlinear computation of the nonlinear term of the Kuramoto-Sivashinsky equation

        Parameters
        ----------

        other : OrbitKS
            The second component of the nonlinear product see Notes for details
        qk_matrix : matrix
            The matrix with the correctly ordered spatial frequencies.

        Notes
        -----
        The nonlinear product is the name given to the elementwise product equivalent to the
        convolution of spatiotemporal Fourier modes. It's faster and more accurate hence why it is used.
        The matrix vector product takes the form d_x (u * v), but the "normal" usage is d_x (u * u); in the latter
        case other is the same as self. The spatial frequency matrix is passed to avoid redundant function calls,
        improving speed.

        """
        # Elementwise product, both self and other should be in physical field basis.
        assert (self.state_type == 'field') and (other.state_type == 'field')
        pseudospectral = self.statemul(other)
        # Return Spatial derivative with 1/2 factor. The conversion to modes does not do anything unless original
        # state has discrete symmetry
        return 0.5 * pseudospectral.dx().convert(to='modes')

    def rpseudospectral(self, other):
        """ nonlinear computation of the nonlinear term of the adjoint Kuramoto-Sivashinsky equation

        Parameters
        ----------

        other : OrbitKS
            The second component of the nonlinear product see Notes for details
        qk_matrix : matrix
            The matrix with the correctly ordered spatial frequencies.

        Notes
        -----
        The nonlinear product is the name given to the elementwise product equivalent to the
        convolution of spatiotemporal Fourier modes. It's faster and more accurate hence why it is used.
        The matrix vector product takes the form -1 * u * d_x v. The spatial frequency matrix is passed to avoid
        redundant function calls, improving speed.

        """
        # Take spatial derivative
        return -1.0 * self.convert(to='field').statemul(other.dx().convert(to='field')).convert(to='modes')

    def random_initial_condition(self, T, L, *args, **kwargs):
        """ Initial a set of random spatiotemporal Fourier modes

        Parameters
        ----------
        T : float
            Time period
        L : float
            Space period

        **kwargs
            time_scale : int
                The number of temporal frequencies to keep after truncation.
            space_scale : int
                The number of spatial frequencies to get after truncation.
        Returns
        -------
        self :
            OrbitKS whose state has been modified to be a set of random Fourier modes.

        Notes
        -----
        Anecdotal evidence suggests that "worse" initial conditions converge more often to solutions of the
        predetermined symmetry group. In other words it's better to start far away from the chaotic attractor
        because then it is less likely to start near equilibria. Spatial scale currently unused, still testing
        for the best random fields.

        """
        if T == 0.:
            self.T = 20 + 100*np.random.rand(1)
        else:
            self.T = T
        if L == 0.:
            self.L = 22 + 44*np.random.rand(1)
        else:
            self.L = L

        spectrum_type = kwargs.get('spectrum', 'random')
        self.N = kwargs.get('N', np.max([32, 2**(int(np.log2(self.T)-1))]))
        self.M = kwargs.get('M', np.max([2**(int(np.log2(self.L))), 32]))
        self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1

        if spectrum_type == 'gaussian':
            time_scale = np.min([kwargs.get('time_scale', self.n), self.n])
            space_scale = np.min([kwargs.get('space_scale', self.m), self.m])
            # Account for different sized spectra
            rmodes = np.random.randn(self.N-1, self.M-2)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0 ** mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier[time_scale:, :] = 0
            mollifier = np.concatenate((mollifier, mollifier), axis=1)
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, self.M-2]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)
        else:
            time_scale = np.min([kwargs.get('time_scale', self.n), self.n])
            space_scale = np.min([kwargs.get('space_scale', self.m), self.m])
            # Account for different sized spectra
            rmodes = np.random.randn(self.N-1, self.M-2)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0 ** mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier[time_scale:, :] = 0
            mollifier = np.concatenate((mollifier, mollifier), axis=1)
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, self.M-2]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)
        self.mode_shape = rmodes.shape
        return self

    def reflection(self):
        """ Reflect the velocity field about the spatial midpoint

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is the reflected velocity field -u(-x,t).
        """
        # Different points in space represented by columns of the state array
        reflected_field = -1.0*np.roll(np.fliplr(self.convert(to='field').state), 1, axis=1)
        return self.__class__(state=reflected_field, state_type='field', T=self.T, L=self.L, S=-1.0*self.S)

    def residual(self, apply_mapping=True):
        """ The value of the cost function

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 the squared L_2 norm of the spatiotemporal mapping,
            R = 1/2 ||F||^2.
        """
        if apply_mapping:
            v = self.convert(to='modes').spatiotemporal_mapping().state.ravel()
            return 0.5 * v.dot(v)
        else:
            u = self.state.ravel()
            return 0.5 * u.dot(u)

    def rmatvec(self, other, parameter_constraints=(False, False), preconditioning=True):
        """ Matrix-vector product with the adjoint of the Jacobian

        Parameters
        ----------
        other : OrbitKS
            OrbitKS whose state represents the vector in the matrix-vector product.
        parameter_constraints : (bool, bool)
            Whether or not period T or spatial period L are fixed.
        preconditioning : bool
            Whether or not to apply (left) preconditioning to the adjoint matrix vector product.

        Returns
        -------
        orbit_rmatvec :
            OrbitKS with values representative of the adjoint-vector product A^H * x. Equivalent to
            evaluation of -v_t + v_xx + v_xxxx  - (u .* v_x)

        """
        assert (self.state_type == 'modes') and (other.state_type == 'modes')
        modes = self.state
        other_modes = other.state
        elementwise_dt = self.elementwise_dtn(self.parameters)
        elementwise_dx = self.elementwise_dxn(self.parameters)
        elementwise_dx2 = self.elementwise_dxn(self.parameters, power=2)
        elementwise_dx4 = self.elementwise_dxn(self.parameters, power=4)

        dt = swap_modes(np.multiply(elementwise_dt, other_modes), dimension='time')
        dx2 = np.multiply(elementwise_dx2, other_modes)
        dx4 = np.multiply(elementwise_dx4, other_modes)
        linear = -1.0 * dt + dx2 + dx4

        # Nonlinear component, equal to -(u .* v_x)
        field = self.convert(to='field')
        other_modes_dx = swap_modes(np.multiply(elementwise_dx,
                                                other_modes), dimension='space')
        other_field_dx = other.__class__(state=other_modes_dx, state_type='modes').convert(to='field')

        nonlinear = -1.0 * field.statemul(other_field_dx).convert(to='modes').state
        rmatvec_modes = linear + nonlinear

        if not parameter_constraints[0]:
            # Derivative with respect to T term equal to DF/DT * v
            dfdt = (-1.0 / self.T) * swap_modes(np.multiply(elementwise_dt, modes), dimension='time')
            rmatvec_T = dfdt.ravel().dot(other_modes.ravel())
            if preconditioning:
                rmatvec_T = rmatvec_T / self.T


        if not parameter_constraints[1]:
            # Derivative with respect to L equal to DF/DL * v
            dfdl_linear = ((-2.0/self.L)*np.multiply(elementwise_dx2, modes)
                           + (-4.0/self.L)*np.multiply(elementwise_dx4, modes))
            nonlinear = 0.5 * swap_modes(np.multiply(self.elementwise_dxn(self.parameters),
                                         field.statemul(field).convert(to='modes').state), dimension='space')
            dfdl_nonlinear = (-1.0 / self.L) * nonlinear
            dfdl = dfdl_linear + dfdl_nonlinear
            rmatvec_L = dfdl.ravel().dot(other_modes.ravel())
            if preconditioning:
                rmatvec_L = rmatvec_L / (self.L**4)

        if preconditioning:
            # Apply left preconditioning
            p_matrix = 1.0 / (np.abs(elementwise_dt)
                              + np.abs(elementwise_dx2)
                              + elementwise_dx4)
            rmatvec_modes = np.multiply(p_matrix, rmatvec_modes)

        return self.__class__(state=rmatvec_modes, state_type='modes', T=rmatvec_T, L=rmatvec_L)

    def rotate(self, distance=0, direction='space', inplace=False):
        """ Rotate the velocity field in either space or time.

        Parameters
        ----------
        distance : float
            The rotation / translation amount, in dimensionless units of time or space.
        direction : str
            Takes values 'space' or 'time'. Direction which rotation will be performed in.
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose field has been rotated.

        Notes
        -----
        Due to periodic boundary conditions, translation is equivalent to rotation on a fundemantal level here.
        Hence the use of 'distance' instead of 'angle'. This can be negative. Also due to the periodic boundary
        conditions, a distance equaling the entire domain length is equivalent to no rotation. I.e.
        the rotation is always modulo L or modulo T.

        The orbit only remains a converged solution if rotations coincide with collocation
        points.  i.e. multiples of L / M and T / N. The reason for this is because arbitrary rotations require
        interpolation of the field.
        """
        if direction == 'space':
            thetak = distance*self.wave_vector(self.parameters)
        else:
            thetak = distance*self.frequency_vector(self.parameters)

        cosinek = np.cos(thetak)
        sinek = np.sin(thetak)
        original_state_type = copy.copy(self.state_type)

        if direction == 'space':
            self.convert(to='s_modes', inplace=True)
            # Rotation breaks discrete symmetry and destroys the solution.
            if self.__class__.__name__ in ['ShiftReflectionOrbitKS', 'AntisymmetricOrbitKS']:
                warnings.warn('Performing a spatial rotation on a orbit with discrete symmetry is greatly discouraged.')

            # Refer to rotation matrix in 2-D for reference.
            cosine_block = np.tile(cosinek.reshape(1, -1), (self.N, 1))
            sine_block = np.tile(sinek.reshape(1, -1), (self.N, 1))

            # Rotation performed on spatial modes because otherwise rotation is ill-defined for Antisymmetric and
            # Shift-reflection symmetric tori.
            spatial_modes_real = self.state[:, :-self.m]
            spatial_modes_imaginary = self.state[:, -self.m:]
            rotated_real = (np.multiply(cosine_block, spatial_modes_real)
                            + np.multiply(sine_block, spatial_modes_imaginary))
            rotated_imag = (-np.multiply(sine_block, spatial_modes_real)
                            + np.multiply(cosine_block, spatial_modes_imaginary))
            rotated_s_modes = np.concatenate((rotated_real, rotated_imag), axis=1)
            if inplace:
                self.state = rotated_s_modes
                self.state_type = 's_modes'
                self.convert(to=original_state_type)
                return self
            else:
                return self.__class__(state=rotated_s_modes, state_type='s_modes',
                                      T=self.T, L=self.L, S=self.S).convert(to=original_state_type)
        else:
            self.convert(to='modes', inplace=True)
            # Refer to rotation matrix in 2-D for reference.
            cosine_block = np.tile(cosinek.reshape(-1, 1), (1, self.state.shape[1]))
            sine_block = np.tile(sinek.reshape(-1, 1), (1, self.state.shape[1]))

            modes_timereal = self.state[1:-self.n, :]
            modes_timeimaginary = self.state[-self.n:, :]
            # Elementwise product to account for matrix product with "2-D" rotation matrix
            rotated_real = (np.multiply(cosine_block, modes_timereal)
                            + np.multiply(sine_block, modes_timeimaginary))
            rotated_imag = (-np.multiply(sine_block, modes_timereal)
                            + np.multiply(cosine_block, modes_timeimaginary))
            time_rotated_modes = np.concatenate((self.state[0, :].reshape(1, -1), rotated_real, rotated_imag), axis=0)
            if inplace:
                self.state = time_rotated_modes
                self.state_type = 'modes'
                self.convert(to=original_state_type, inplace=True)
                return self
            else:
                self.convert(to=original_state_type)
                return self.__class__(state=time_rotated_modes,
                                      T=self.T, L=self.L, S=self.S).convert(to=original_state_type)

    def shift_reflection(self):
        """ Return a OrbitKS with shift-reflected velocity field

        Returns
        -------
        OrbitKS :
            OrbitKS with shift-reflected velocity field

        Notes
        -----
            Shift reflection in this case is a composition of spatial reflection and temporal translation by
            half of the period. Because these are in different dimensions these operations commute.
        """
        shift_reflected_field = np.roll(-1.0*np.roll(np.fliplr(self.state), 1, axis=1), self.n, axis=0)
        return self.__class__(state=shift_reflected_field, state_type='field', T=self.T, L=self.L, S=self.S)

    def space_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        space_modes = rfft(self.state, norm='ortho', axis=1)[:, 1:-1]
        spatial_modes = np.concatenate((space_modes.real, space_modes.imag), axis=1)
        if inplace:
            self.state_type = 's_modes'
            self.state = spatial_modes
            return self
        else:
            return self.__class__(state=spatial_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def inv_space_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Make the modes complex valued again.
        complex_modes = self.state[:, :-self.m] + 1j * self.state[:, -self.m:]
        # Re-add the zeroth and Nyquist spatial frequency modes (zeros) and then transform back
        z = np.zeros([self.N, 1])
        field = irfft(np.concatenate((z, complex_modes, z), axis=1), norm='ortho', axis=1)
        if inplace:
            self.state_type = 'field'
            self.state = field
            return self
        else:
            return self.__class__(state=field, state_type='field', T=self.T, L=self.L, S=self.S)

    def inv_space_transform_matrix(self):
        """ Inverse spatial Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial Fourier modes into a physical field u(x,t).

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the inverse Fourier transform.
        """

        idft_mat_real = irfft(np.eye(self.M//2 + 1), norm='ortho', axis=0)[:, 1:-1]
        idft_mat_imag = irfft(1j*np.eye(self.M//2 + 1), norm='ortho', axis=0)[:, 1:-1]
        space_idft_mat = np.concatenate((idft_mat_real, idft_mat_imag), axis=1)
        return np.kron(np.eye(self.N), space_idft_mat)

    def space_transform_matrix(self):
        """ Spatial Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a physical field u(x,t) into a set of spatial Fourier modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        The matrix is formatted such that is u(x,t), the entire spatiotemporal discretization of the
        orbit is vector resulting from flattening the 2-d array wherein increasing column index is increasing space
        variable x and increasing rows is *decreasing* time. This is because of the Physics convention for increasing
        time to always be "up". By taking the real and imaginary components of the DFT matrix and concatenating them
        we convert the output from a vector with elements form u_k(t) = a_k + 1j*b_k(t) to a vector of the form
        [a_0, a_1, a_2, ..., b_1, b_2, b_3, ...]. The kronecker product enables us to act on the entire orbit
        at once instead of a single instant in time.

        Discard zeroth mode because of the constraint on Galilean velocity (mean flow). Discard Nyquist frequency
        because of real input of even dimension; just makes matrix operators awkward as well.
        """

        #
        dft_mat = rfft(np.eye(self.M), norm='ortho', axis=0)[1:-1, :]
        space_dft_mat = np.concatenate((dft_mat.real, dft_mat.imag), axis=0)
        return np.kron(np.eye(self.N), space_dft_mat)

    def inv_spacetime_transform(self, inplace=False):
        """ Inverse space-time Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS instance in the physical field basis.
        """
        if inplace:
            self.inv_time_transform(inplace=True).inv_space_transform(inplace=True)
            self.state_type = 'field'
            return self
        else:
            return self.inv_time_transform().inv_space_transform()

    def spacetime_transform(self, inplace=False):
        """ Space-time Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS instance in the spatiotemporal mode basis.
        """
        if inplace:
            self.space_transform(inplace=True).time_transform(inplace=True)
            self.state_type = 'modes'
            return self
        else:
            # Return transform of field
            return self.space_transform().time_transform()

    def inv_spacetime_transform_matrix(self):
        """ Inverse Space-time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a physical field u(x,t)

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        return np.dot(self.inv_space_transform_matrix(), self.inv_time_transform_matrix())

    def spacetime_transform_matrix(self):
        """ Space-time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a physical field u(x,t) into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        return np.dot(self.time_transform_matrix(), self.space_transform_matrix())

    def spatiotemporal_mapping(self):
        """ The Kuramoto-Sivashinsky equation evaluated at the current state.

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is representative of the equation u_t + u_xx + u_xxxx + 1/2 (u^2)_x
        :return:
        """
        # For specific computation of the linear component instead
        # of arbitrary derivatives we can optimize the calculation by being specific.
        modes = self.convert(to='modes').state
        elementwise_dt = self.elementwise_dtn(self.parameters)
        elementwise_dx2 = self.elementwise_dxn(self.parameters, power=2)
        elementwise_dx4 = self.elementwise_dxn(self.parameters, power=4)

        dt = swap_modes(np.multiply(elementwise_dt, modes), dimension='time')
        dx2 = np.multiply(elementwise_dx2, modes)
        dx4 = np.multiply(elementwise_dx4, modes)
        linear = dt + dx2 + dx4

        # Return Spatial derivative with 1/2 factor. The conversion to modes does not do anything unless original
        # state has discrete symmetry
        orbit_field = self.convert(to='field')

        nonlinear = 0.5 * swap_modes(np.multiply(self.elementwise_dxn(self.parameters),
                                orbit_field.statemul(orbit_field).convert(to='modes').state), dimension='space')

        mapping_modes = linear + nonlinear
        return self.__class__(state=mapping_modes, state_type='modes', T=self.T, L=self.L)

    def statemul(self, other):
        """ Elementwise multiplication of two Tori states

        Returns
        -------
        OrbitKS :
            The OrbitKS representing the product.

        Notes
        -----
        Only really makes sense when taking an elementwise product between Tori defined on spatiotemporal
        domains of the same size.
        """
        return self.__class__(state=np.multiply(self.state, other.state),
                              state_type=self.state_type, T=self.T, L=self.L, S=self.S)

    def time_transform(self, inplace=False):
        """ Temporal Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = rfft(self.state, norm='ortho', axis=0)
        modes_real = modes.real[:-1, :]
        modes_imag = modes.imag[1:-1, :]
        spacetime_modes = np.concatenate((modes_real, modes_imag), axis=0)

        if inplace:
            self.state_type = 'modes'
            self.state = spacetime_modes
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', T=self.T, L=self.L, S=self.S)

    def inv_time_transform(self, inplace=False):
        """ Temporal Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the temporal Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.

        modes = self.state
        time_real = modes[:-self.n, :]
        time_imaginary = np.concatenate((np.zeros([1, self.mode_shape[1]]), modes[-self.n:, :]), axis=0)
        complex_modes = np.concatenate((time_real + 1j * time_imaginary, np.zeros([1, self.mode_shape[1]])), axis=0)
        space_modes = irfft(complex_modes, norm='ortho', axis=0)

        if inplace:
            self.state_type = 's_modes'
            self.state = space_modes
            return self
        else:
            return self.__class__(state=space_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def time_transform_matrix(self):
        """ Inverse Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a set of spatial modes

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        dft_mat = rfft(np.eye(self.N), norm='ortho', axis=0)
        time_idft_mat = np.concatenate((dft_mat[:-1, :].real,
                                        dft_mat[1:-1, :].imag), axis=0)
        return np.kron(time_idft_mat, np.eye(self.M-2))

    def inv_time_transform_matrix(self):
        """ Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        idft_mat_real = irfft(np.eye(self.N//2 + 1), norm='ortho', axis=0)
        idft_mat_imag = irfft(1j * np.eye(self.N//2 + 1), norm='ortho', axis=0)
        time_idft_mat = np.concatenate((idft_mat_real[:, :-1],
                                        idft_mat_imag[:, 1:-1]), axis=1)
        return np.kron(time_idft_mat, np.eye(self.M-2))

    def to_fundamental_domain(self, **kwargs):
        """ Placeholder for subclassees, included for compatibility"""
        return self

    def to_h5(self, filename=None, directory='', verbose=False):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str
            Name for the save file
        directory :
            Location to save at
        verbose : If true, prints save messages to std out
        """
        if filename is None:
            filename = self.parameter_dependent_filename()
        elif filename == 'initial':
            filename = 'initial_' + self.parameter_dependent_filename()

        if directory == 'default':
            directory = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../data/')), '')
        elif directory == '':
            pass
        elif not os.path.isdir(directory):
            warnings.warn('Trying to save figure to a directory that does not exist:' + directory, Warning)
            sys.stdout.flush()
            proceed = input('Would you like to create this directory? If no, '
                            'then figure will save where current script is located [y]/[n]')
            if proceed == 'y':
                os.mkdir(directory)

        filename = os.path.join(directory, filename)
        if verbose:
            print('Saving data to {}'.format(filename))
        with h5py.File(filename, 'w') as f:
            f.create_dataset("field", data=self.convert(to='field').state)
            f.create_dataset("speriod", data=self.L)
            f.create_dataset("period", data=self.T)
            f.create_dataset("space_discretization", data=self.M)
            f.create_dataset("time_discretization", data=self.N)
            f.create_dataset("spatial_shift", data=self.S)
            f.create_dataset("residual", data=float(self.residual()))
        return None


class AntisymmetricOrbitKS(OrbitKS):

    def __init__(self, state=None, state_type='field', T=0., L=0., **kwargs):

        try:
            if state is not None:
                shp = state.shape
                self.state = state
                self.state_type = state_type
                if state_type == 'modes':
                    self.N, self.M = shp[0] + 1, 2*shp[1] + 2
                elif state_type == 'field':
                    self.N, self.M = shp
                elif state_type == 's_modes':
                    self.N, self.M = shp[0], shp[1]+2

                self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
                self.T, self.L = T, L
            else:
                self.random_initial_condition(T, L, **kwargs)
            self.mode_shape = (self.N, self.m)
            # For uniform save format
            self.S = 0.
        except ValueError:
            print('Incompatible type provided for field or modes: 2-D NumPy arrays only')

    def dx(self, power=1):
        """ Overwrite of parent method """
        qkn = self.wave_vector(self.parameters, power=power)
        if np.mod(power, 2):
            c1, c2 = so2_coefficients(power=power)
            elementwise_dxn = np.tile(np.concatenate((c1*qkn, c2*qkn), axis=1), (self.N, 1))
            dxn_s_modes = np.multiply(elementwise_dxn, self.convert(to='s_modes').state)
            dxn_s_modes = swap_modes(dxn_s_modes, dimension='space')
            return self.__class__(state=dxn_s_modes, state_type='s_modes', T=self.T, L=self.L)
        else:
            c, _ = so2_coefficients(power=power)
            elementwise_dxn = np.tile(c*qkn, (self.N-1, 1))
            dxn_modes = np.multiply(self.convert(to='modes').state, elementwise_dxn)
            return self.__class__(state=dxn_modes, state_type='modes', T=self.T, L=self.L).convert(to=self.state_type)

    def elementwise_dx(self):
        """ Overwrite of parent method """
        qk = self.wave_vector(self.parameters)
        return np.tile(qk, (self.N-1, 1))

    def dx_matrix(self, power=1, **kwargs):
        """ Overwrite of parent method """
        state_type = kwargs.get('state_type', self.state_type)
        # Define spatial wavenumber vector
        if state_type == 'modes':
            _, c = so2_coefficients(power=power)
            dx_n_matrix = c * np.diag(self.wave_vector(self.parameters, power=power).ravel())
            dx_matrix_complete = np.kron(np.eye(self.N-1), dx_n_matrix)
        else:
            dx_n_matrix = np.kron(so2_generator(power=power), np.diag(self.wave_vector(self.parameters,
                                                                                       power=power).ravel()))
            dx_matrix_complete = np.kron(np.eye(self.N), dx_n_matrix)
        return dx_matrix_complete

    def from_fundamental_domain(self, inplace=False, **kwargs):
        """ Overwrite of parent method """
        half = kwargs.get('half', 'left')
        if half == 'left':
            full_field = np.concatenate((self.state, self.reflection().state), axis=1)
        else:
            full_field = np.concatenate((self.reflection().state, self.state), axis=1)
        return self.__class__(state=full_field, state_type='field', T=self.T, L=2.0*self.L)

    def mode_padding(self, size, inplace=False, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            first_half = self.state[:-self.n, :]
            second_half = self.state[-self.n:, :]
            padding_number = int((size-self.N) // 2)
            padding = np.zeros([padding_number, self.state.shape[1]])
            padded_modes = np.concatenate((first_half, padding, second_half, padding), axis=0)
        else:
            padding_number = int((size-self.M) // 2)
            padding = np.zeros([self.state.shape[0], padding_number])
            padded_modes = np.concatenate((self.state, padding), axis=1)

        return self.__class__(state=padded_modes, state_type=self.state_type, T=self.T, L=self.L)

    def mode_truncation(self, size, inplace=False, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            truncate_number = int(size // 2) - 1
            first_half = self.state[:truncate_number+1, :]
            second_half = self.state[-self.n:-self.n+truncate_number, :]
            truncated_modes = np.concatenate((first_half, second_half), axis=0)
        else:
            truncate_number = int(size // 2) - 1
            truncated_modes = self.state[:, :truncate_number]
        return self.__class__(state=truncated_modes, state_type=self.state_type, T=self.T, L=self.L)

    def random_initial_condition(self, T, L, **kwargs):
        """ Initial a set of random spatiotemporal Fourier modes

        Parameters
        ----------
        **kwargs
            time_scale : int
                The number of temporal frequencies to keep after truncation.
            space_scale : int
                The number of spatial frequencies to get after truncation.
        Returns
        -------
        self :
            OrbitKS whose state has been modified to be a set of random Fourier modes.

        Notes
        -----
        Anecdotal evidence suggests that "worse" initial conditions converge more often to solutions of the
        predetermined symmetry group. In other words it's better to start far away from the chaotic attractor
        because then it is less likely to start near equilibria. Spatial scale currently unused, still testing
        for the best random fields.

        """
        spectrum_type = kwargs.get('spectrum', 'random')
        if T == 0.:
            self.T = 20 + 160*np.random.rand()
        else:
            self.T = T
        if L == 0.:
            self.L = 22 + 44*np.random.rand()
        else:
            self.L = L
        self.N = kwargs.get('N', np.max([32, 2**(int(np.log2(self.T)-1))]))
        self.M = kwargs.get('M', np.max([2**(int(np.log2(self.L))), 32]))
        self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
        time_scale = np.min([kwargs.get('time_scale', self.n), self.n])
        space_scale = np.min([kwargs.get('space_scale', self.m), self.m])
        if spectrum_type == 'gaussian':
            rmodes = np.random.randn(self.N-1, self.m)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0**mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier[time_scale:, :] = 0
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)
        else:
            # Account for different sized spectra
            rmodes = np.random.randn(self.N-1, self.m)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0**mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier[time_scale:, :] = 0
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)

        self.mode_shape = rmodes.shape
        self.convert(to='field', inplace=True)
        tmp = self // (1.0/4.0)
        self.state = tmp.state
        self.convert(to='modes', inplace=True)
        return self

    def time_transform_matrix(self):
        """ Inverse Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a set of spatial modes

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        dft_mat = rfft(np.eye(self.N), norm='ortho', axis=0)
        time_dft_mat = np.concatenate((dft_mat[:-1, :].real,
                                       dft_mat[1:-1, :].imag), axis=0)
        ab_time_dft_mat = np.insert(time_dft_mat,
                                    np.arange(time_dft_mat.shape[1]),
                                    np.zeros([time_dft_mat.shape[0], time_dft_mat.shape[1]]),
                                    axis=1)
        return np.kron(ab_time_dft_mat, np.eye(self.m))

    def inv_time_transform_matrix(self):
        """ Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        idft_mat_real = irfft(np.eye(self.N//2 + 1), norm='ortho', axis=0)
        idft_mat_imag = irfft(1j*np.eye(self.N//2 + 1), norm='ortho', axis=0)
        time_idft_mat = np.concatenate((idft_mat_real[:, :-1],
                                        idft_mat_imag[:, 1:-1]), axis=1)
        ab_time_idft_mat = np.insert(time_idft_mat,
                                     np.arange(time_idft_mat.shape[0]),
                                     np.zeros([time_idft_mat.shape[0], time_idft_mat.shape[1]]),
                                     axis=0)
        return np.kron(ab_time_idft_mat, np.eye(self.m))

    def time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = rfft(self.state, norm='ortho', axis=0)
        modes_real = modes.real[:-1, -self.m:]
        modes_imag = modes.imag[1:-1, -self.m:]
        spacetime_modes = np.concatenate((modes_real, modes_imag), axis=0)

        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', T=self.T, L=self.L, S=self.S)

    def inv_time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.

        modes = self.state
        time_real = modes[:-self.n, :]
        time_imaginary = 1j*np.concatenate((np.zeros([1, self.m]), modes[-self.n:, :]), axis=0)
        spacetime_modes = np.concatenate((time_real + time_imaginary, np.zeros([1, self.m])), axis=0)
        imaginary_space_modes = irfft(spacetime_modes, norm='ortho', axis=0)
        space_modes = np.concatenate((np.zeros(imaginary_space_modes.shape), imaginary_space_modes), axis=1)

        if inplace:
            self.state = space_modes
            self.state_type = 's_modes'
            return self
        else:
            return self.__class__(state=space_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def to_fundamental_domain(self, inplace=False, **kwargs):
        """ Overwrite of parent method """
        half = kwargs.get('half', 'left')
        if half == 'left':
            fundamental_domain = self.__class__(state=self.convert(to='field').state[:, :-int(self.M//2)],
                                                state_type='field', T=self.T, L=self.L / 2.0)
        else:
            fundamental_domain = self.__class__(state=self.convert(to='field').state[:, -int(self.M//2):],
                                                state_type='field', T=self.T, L=self.L / 2.0)
        return fundamental_domain


class RelativeOrbitKS(OrbitKS):

    def __init__(self, state=None, state_type='field', T=0., L=0., S=0., frame='comoving', **kwargs):
        try:
            if state is not None:
                shp = state.shape
                self.state = state
                self.state_type = state_type
                if state_type == 'modes':
                    # This separate behavior is for Antisymmetric and ShiftReflection Tori;
                    # This avoids having to define subclass method with repeated code.
                    self.N, self.M = shp[0] + 1, shp[1] + 2
                elif state_type == 'field':
                    self.N, self.M = shp
                elif state_type == 's_modes':
                    self.N, self.M = shp[0], shp[1] + 2

                self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
                self.T, self.L = T, L
            else:
                self.state_type = 'modes'
                self.random_initial_condition(T, L, **kwargs)
                self.convert(to='field', inplace=True)
                self.state = (self // (1.0/4.0)).state
                self.convert(to='modes', inplace=True)
        except ValueError:
            print('Incompatible type provided for field or modes: 2-D NumPy arrays only')
        self.mode_shape = (self.N-1, self.M-2)
        # For uniform save format
        self.frame = frame
        self.S = S

    def calculate_shift(self, inplace=False):
        """ Calculate the phase difference between the spatial modes at t=0 and t=T """
        s_modes = self.convert(to='s_modes').state
        modes1 = s_modes[-1, :].reshape(-1, 1)
        modes0 = s_modes[0, :].reshape(-1, 1)
        angle = np.arccos(np.dot(np.transpose(modes1), modes0)/(np.linalg.norm(modes1)*np.linalg.norm(modes0)))
        # Just to prevent over-winding
        angle = np.sign(angle)*np.mod(angle, 2*pi)
        shift = float((self.L / (2 * pi)) * angle)
        if inplace:
            self.S = shift
            return None
        else:
            return shift

    def comoving_mapping_component(self):
        """ Co-moving frame component of spatiotemporal mapping """
        return -1.0 * (self.S / self.T)*self.dx()

    def comoving_matrix(self):
        """ Operator that constitutes the co-moving frame term """
        return -1.0 * (self.S / self.T)*self.dx_matrix()

    def change_reference_frame(self):
        """ Transform to (or from) the co-moving frame depending on the current reference frame

        Parameters
        ----------
        inplace : bool
            Whether to perform in-place or not

        Returns
        -------
        RelativeOrbitKS :
            RelativeOrbitKS in transformed reference frame.
        """
        if self.frame == 'comoving':
            shift = self.S
        else:
            shift = -1.0*self.S
        original_state = self.state_type
        s_modes = self.convert(to='s_modes').state
        time_vector = np.flipud(np.linspace(0, self.T, num=self.N, endpoint=True)).reshape(-1, 1)
        translation_per_period = -1.0 * self.S / self.T
        time_dependent_translations = translation_per_period*time_vector
        thetak = time_dependent_translations.reshape(-1, 1)*self.wave_vector(self.parameters).ravel()
        cosine_block = np.cos(thetak)
        sine_block = np.sin(thetak)
        real_modes = s_modes[:, :-self.m]
        imag_modes = s_modes[:, -self.m:]
        frame_rotated_s_modes_real = (np.multiply(real_modes, cosine_block)
                                      - np.multiply(imag_modes, sine_block))
        frame_rotated_s_modes_imag = (np.multiply(real_modes, sine_block)
                                      + np.multiply(imag_modes, cosine_block))
        frame_rotated_s_modes = np.concatenate((frame_rotated_s_modes_real, frame_rotated_s_modes_imag), axis=1)
        frame_change_ = self.__class__(state=frame_rotated_s_modes, state_type='s_modes',
                                       T=self.T, L=self.L, S=-1.0*self.S).convert(to=original_state)
        return frame_change_

    def dt(self, power=1):
        """ A time derivative of the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        orbit_dtn : OrbitKS or subclass instance
            The class instance whose state is the time derivative in
            the spatiotemporal mode basis.
        """
        if self.frame == 'comoving':
            return super().dt(power=power)
        else:
            raise ValueError(
                'Attempting to compute time derivative of '+self.__str__+ 'in physical reference frame.')

    def from_fundamental_domain(self):
        return self.change_reference_frame()

    def jacobian(self, parameter_constraints=(False, False, False)):
        """ Jacobian that includes the spatial translation term for relative periodic tori

        Parameters
        ----------
        parameter_constraints : (bool, bool, bool)
            Determines whether or not the various parameters, period, spatial period, spatial shift, (T,L,S)
            are variables or not.

        Returns
        -------
        matrix :
            Jacobian matrix for relative periodic tori.
        """
        self.convert(to='modes', inplace=True)
        # The linearization matrix of the governing equations.
        jac_ = self.jac_lin() + self.jac_nonlin()

        # If period is not fixed, need to include dF/dT for changes to period.
        if not parameter_constraints[0]:
            dt = self.dt()
            ds = -1.0 * (self.S / self.T)*self.dx()
            dfdt = (-1.0 / self.T)*((dt+ds).state.reshape(-1, 1))
            jac_ = np.concatenate((jac_, dfdt), axis=1)

        # If period is not fixed, need to include dF/dL for changes to period.
        if not parameter_constraints[1]:
            orbit_field = self.convert(to='field')
            qk_matrix = self.elementwise_dxn(self.parameters)
            dx2 = np.multiply(-1.0*qk_matrix**2, self.state)
            dx4 = np.multiply(qk_matrix**4, self.state)
            s = self.comoving_mapping_component().state
            dfdl_linear = (-2.0/self.L)*dx2 + (-4.0/self.L)*dx4 + (-1.0/self.L) * s
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear
            jac_ = np.concatenate((jac_, dfdl.reshape(-1, 1)), axis=1)

        if not parameter_constraints[2]:
            dfds = (-1.0 / self.T)*self.dx().state
            jac_ = np.concatenate((jac_, dfds.reshape(-1, 1)), axis=1)

        return jac_

    def jac_lin(self):
        """ Extension of the OrbitKS method that includes the term for spatial translation symmetry"""
        return super().jac_lin() + self.comoving_matrix()

    def matvec(self, other, parameter_constraints=(False, False, False), preconditioning=True, **kwargs):
        """ Extension of parent class method

        Parameters
        ----------
        other : RelativeOrbitKS
            RelativeOrbitKS instance whose state represents the vector in the matrix-vector multiplication.
        parameter_constraints : tuple of bool
            Determines whether to include period and spatial period
            as variables.
        preconditioning : bool
            Whether or not to apply (left) preconditioning P (Ax)

        Returns
        -------
        RelativeOrbitKS
            RelativeOrbitKS whose state and other parameters result from the matrix-vector product.

        Notes
        -----
        Equivalent to computation of (v_t + v_xx + v_xxxx + phi * v_x) + d_x (u .* v)
        The reason for all of the repeated code is that the co-moving terms re-use the same matrices
        as the other terms; this prevents additional function calls.
        """

        assert (self.state_type == 'modes') and (other.state_type == 'modes')
        matvec_orbit = super().matvec(self, other, parameter_constraints=parameter_constraints,
                                      preconditioning=preconditioning)
        modes = self.state
        other_modes = other.state

        elementwise_dx = self.elementwise_dxn(self.parameters)
        comoving_modes = -1.0 * (self.S / self.T) * swap_modes(np.multiply(elementwise_dx, other_modes))
        # this is needed unless all parameters are fixed, but that isn't ever a realistic choice.
        modes_dx = swap_modes(np.multiply(elementwise_dx, modes))

        if not parameter_constraints[0]:
            comoving_modes += other.T * (-1.0 / self.T) * (-1.0 * self.S / self.T) * modes_dx

        if not parameter_constraints[1]:
            # Derivative of mapping with respect to T is the same as -1/T * u_t
            comoving_modes += other.L * (-1.0 / self.L) * (-1.0 * self.S / self.T) * modes_dx

        if not parameter_constraints[2]:
            comoving_modes += other.S * (-1.0 / self.T) * modes_dx

        if preconditioning:
            p_matrix = 1.0 / (np.abs(self.elementwise_dtn(self.parameters))
                              + self.elementwise_dxn(self.parameters, power=2)
                              + self.elementwise_dxn(self.parameters, power=4))
            comoving_modes = np.multiply(comoving_modes, p_matrix)

        matvec_orbit.state += comoving_modes
        return matvec_orbit

    def rmatvec(self, other, parameter_constraints=(False, False, False), preconditioning=True, **kwargs):
        """ Extension of the parent method to RelativeOrbitKS """
        # For specific computation of the linear component instead
        # of arbitrary derivatives we can optimize the calculation by being specific.
        wj_matrix = self.elementwise_dtn(self.parameters)
        qk_matrix = self.elementwise_dxn(self.parameters)
        elementwise_qk2 = -1.0*qk_matrix**2
        elementwise_qk4 = qk_matrix**4

        dt = -1.0 * swap_modes(np.multiply(wj_matrix, other.state), dimension='time')
        dx2 = np.multiply(elementwise_qk2, other.state)
        dx4 = np.multiply(elementwise_qk4, other.state)
        s = (self.S / self.T)*swap_modes(np.multiply(qk_matrix, other.state))
        linear_component = dt + dx2 + dx4 + s
        orbit_field = self.convert(to='field')
        nonlinear_component = orbit_field.rnonlinear(other).state
        orbit_rmatvec = self.__class__(state=(linear_component + nonlinear_component))

        if not parameter_constraints[0]:
            dt_self = swap_modes(np.multiply(wj_matrix, self.state), dimension='time')
            s_self = (-1.0 * self.S / self.T)*swap_modes(np.multiply(qk_matrix, self.state))
            dfdt = (-1.0 / self.T)*(dt_self+s_self)
            orbit_rmatvec.T = np.dot(dfdt.ravel(), other.state.ravel())

        if not parameter_constraints[1]:
            dx2_self = np.multiply(elementwise_qk2, self.state)
            dx4_self = np.multiply(elementwise_qk4, self.state)
            s_self = (-1.0 * self.S / self.T) * swap_modes(np.multiply(qk_matrix, self.state))
            dfdl_linear = (-2.0/self.L)*dx2_self + (-4.0/self.L)*dx4_self + (-1.0/self.L)*s_self
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear

            # Derivative of mapping with respect to T is the same as -1/T * u_t
            orbit_rmatvec.L = np.dot(dfdl.ravel(), other.state.ravel())

        if not parameter_constraints[2]:
            orbit_rmatvec.S = ((-1.0 / self.T)
                               * np.dot(swap_modes(np.multiply(qk_matrix, self.state)).ravel(), other.state.ravel()))

        if preconditioning:
            p_matrix = 1.0 / (np.abs(wj_matrix) + qk_matrix**2 + qk_matrix**4)
            orbit_rmatvec.state = np.multiply(orbit_rmatvec.state, p_matrix)

            if not parameter_constraints[0]:
                orbit_rmatvec.T = orbit_rmatvec.T / self.T

            if not parameter_constraints[1]:
                orbit_rmatvec.L = orbit_rmatvec.L / (self.L**4)

        return orbit_rmatvec

    def random_initial_condition(self, T, L, *args, **kwargs):
        """ Extension of parent modes to include spatial-shift initialization """
        super().random_initial_condition(T, L, **kwargs)
        self.S = self.calculate_shift()
        # if args[0] == 0.0:
        #     # Assign random proportion of L with random sign as the shift if none provided.
        #     self.S = ([-1, 1][int(2*np.random.rand())])*np.random.rand()*self.L
        # else:
        #     self.S = args[0]
        return self

    def spatiotemporal_mapping(self):
        """ Extension of OrbitKS method to include co-moving frame term. """
        return super().spatiotemporal_mapping() + self.comoving_mapping_component()

    def state_vector(self):
        """ Vector which completely describes the orbit."""
        return np.concatenate((self.state.reshape(-1, 1),
                               np.array([[float(self.T)]]),
                               np.array([[float(self.L)]]),
                               np.array([[float(self.S)]])), axis=0)

    def status(self):
        """ Check if orbit has meaningful translation symmetry """

        if np.abs(self.S) < 10**-4:
            return OrbitKS(state=self.state, state_type=self.state_type, T=self.T, L=self.L).status()
        else:
            return super().status()

    def to_fundamental_domain(self):
        return self.change_reference_frame()


class ShiftReflectionOrbitKS(OrbitKS):

    def __init__(self, state=None, state_type='field', T=0., L=0., **kwargs):
        try:
            if state is not None:
                shp = state.shape
                self.state = state
                self.state_type = state_type
                if state_type == 'modes':
                    self.N, self.M = shp[0] + 1, 2*shp[1] + 2
                elif state_type == 'field':
                    self.N, self.M = shp
                elif state_type == 's_modes':
                    self.N, self.M = shp[0], shp[1]+2
                self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
                self.T, self.L = T, L
            else:
                self.random_initial_condition(T=T, L=L, **kwargs)
            self.mode_shape = (self.N-1, self.m)
            # For uniform save format
            self.S = 0.
        except ValueError:
            print('Incompatible type provided for field or modes: 2-D NumPy arrays only')

    def dx(self, power=1):
        """ Overwrite of parent method """
        qkn = self.wave_vector(self.parameters, power=power)
        if np.mod(power, 2):
            c1, c2 = so2_coefficients(power=power)
            elementwise_dxn = np.tile(np.concatenate((c1*qkn, c2*qkn), axis=1), (self.N, 1))
            dxn_s_modes = np.multiply(elementwise_dxn, self.convert(to='s_modes').state)
            dxn_s_modes = swap_modes(dxn_s_modes, dimension='space')
            return self.__class__(state=dxn_s_modes, state_type='s_modes', T=self.T, L=self.L).convert(to='s_modes')
        else:
            c, _ = so2_coefficients(power=power)
            elementwise_dxn = np.tile(c*qkn, (self.N-1, 1))
            dxn_modes = np.multiply(self.convert(to='modes').state, elementwise_dxn)
            return self.__class__(state=dxn_modes, state_type='modes', T=self.T, L=self.L).convert(to=self.state_type)

    def elementwise_dx(self):
        """ Overwrite of parent method """
        qk = self.wave_vector(self.parameters)
        return np.tile(qk, (self.N-1, 1))

    def dx_matrix(self, power=1, **kwargs):
        """ Overwrite of parent method """
        state_type = kwargs.get('state_type', self.state_type)
        # Define spatial wavenumber vector
        if state_type == 'modes':
            _, c = so2_coefficients(power=power)
            dx_n_matrix = c * np.diag(self.wave_vector(self.parameters, power=power).ravel())
            dx_matrix_complete = np.kron(np.eye(self.N-1), dx_n_matrix)
        else:
            dx_n_matrix = np.kron(so2_generator(power=power), np.diag(self.wave_vector(self.parameters,
                                                                                       power=power).ravel()))
            dx_matrix_complete = np.kron(np.eye(self.N), dx_n_matrix)
        return dx_matrix_complete

    def from_fundamental_domain(self):
        """ Reconstruct full field from discrete fundamental domain """
        field = np.concatenate((self.reflection().state, self.state), axis=0)
        return self.__class__(state=field, state_type='field', T=2*self.T, L=self.L)

    def mode_padding(self, size, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            first_half = self.state[:-self.n, :]
            second_half = self.state[-self.n:, :]
            padding_number = int((size-self.N) // 2)
            padding = np.zeros([padding_number, self.state.shape[1]])
            padded_modes = np.concatenate((first_half, padding, second_half, padding), axis=0)
        else:
            padding_number = int((size-self.M) // 2)
            padding = np.zeros([self.state.shape[0], padding_number])
            padded_modes = np.concatenate((self.state, padding), axis=1)
        return self.__class__(state=padded_modes, state_type=self.state_type, T=self.T, L=self.L)

    def mode_truncation(self, size, inplace=False, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            truncate_number = int(size // 2) - 1
            first_half = self.state[:truncate_number+1, :]
            second_half = self.state[-self.n:-self.n+truncate_number, :]
            truncated_modes = np.concatenate((first_half, second_half), axis=0)
        else:
            truncate_number = int(size // 2) - 1
            truncated_modes = self.state[:, :truncate_number]
        return self.__class__(state=truncated_modes, state_type=self.state_type, T=self.T, L=self.L)

    def random_initial_condition(self, T, L, **kwargs):
        """ Initial a set of random spatiotemporal Fourier modes

        Parameters
        ----------
        **kwargs
            time_scale : int
                The number of temporal frequencies to keep after truncation.
            space_scale : int
                The number of spatial frequencies to get after truncation.
        Returns
        -------
        self :
            OrbitKS whose state has been modified to be a set of random Fourier modes.

        Notes
        -----
        Anecdotal evidence suggests that "worse" initial conditions converge more often to solutions of the
        predetermined symmetry group. In other words it's better to start far away from the chaotic attractor
        because then it is less likely to start near equilibria. Spatial scale currently unused, still testing
        for the best random fields.

        """
        spectrum_type = kwargs.get('spectrum', 'random')
        if T == 0.:
            self.T = 20 + 100*np.random.rand(1)[0]
        else:
            self.T = T
        if L == 0.:
            self.L = 22 + 44*np.random.rand(1)[0]
        else:
            self.L = L
        self.N = kwargs.get('N', np.max([32, 2**(int(np.log2(self.T)-1))]))
        self.M = kwargs.get('M', np.max([2**(int(np.log2(self.L))), 32]))
        self.n, self.m = int(self.N // 2) - 1, int(self.M // 2) - 1
        time_scale = np.min([kwargs.get('time_scale', self.n), self.n])
        space_scale = np.min([kwargs.get('space_scale', self.m), self.m])
        # if spectrum_type == 'random':
            # Account for different sized spectra
        rmodes = np.random.randn(self.N-1, self.m)
        mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
        mollifier = 10.0**mollifier_exponents
        mollifier[:, :space_scale] = 1
        mollifier[time_scale:, :] = 0
        mollifier = np.concatenate((mollifier, mollifier), axis=0)
        mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
        self.state = np.multiply(mollifier, rmodes)
        # elif spectrum_type == 'gaussian':
        #     # Account for different sized spectra
        #     rmodes = np.random.randn(self.N-1, self.m)
        #     mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
        #     mollifier = 10.0**mollifier_exponents
        #     mollifier[:, :space_scale] = 1
        #     mollifier[time_scale:, :] = 0
        #     mollifier = np.concatenate((mollifier, mollifier), axis=0)
        #     mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
        #     self.state = np.multiply(mollifier, rmodes)
        self.mode_shape = rmodes.shape
        self.convert(to='field', inplace=True)
        tmp = self // (1.0/4.0)
        self.state = tmp.state
        self.convert(to='modes', inplace=True)
        return self

    def time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = rfft(self.state, norm='ortho', axis=0)
        modes_real = modes.real[:-1, :-self.m] + modes.real[:-1, -self.m:]
        modes_imag = modes.imag[1:-1, :-self.m] + modes.imag[1:-1, -self.m:]
        spacetime_modes = np.concatenate((modes_real, modes_imag), axis=0)

        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', T=self.T, L=self.L, S=self.S)

    def inv_time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = self.state
        even_indices = np.arange(0, self.N//2, 2)
        odd_indices = np.arange(1, self.N//2, 2)
        time_real = modes[:-self.n, :]
        time_imaginary = 1j*np.concatenate((np.zeros([1, self.m]), modes[-self.n:, :]), axis=0)
        spacetime_modes = np.concatenate((time_real + time_imaginary, np.zeros([1, self.m])), axis=0)
        real_space, imaginary_space = spacetime_modes.copy(), spacetime_modes.copy()
        real_space[even_indices, :] = 0
        imaginary_space[odd_indices, :] = 0
        space_modes = irfft(np.concatenate((real_space, imaginary_space), axis=1), norm='ortho', axis=0)

        if inplace:
            self.state = space_modes
            self.state_type = 's_modes'
            return self
        else:
            return self.__class__(state=space_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def time_transform_matrix(self):
        """

        Notes
        -----
        This function and its inverse are the most confusing parts of the code by far.
        The reason is because in order to implement a symmetry invariant time RFFT that retains no
        redundant modes (constrained to be zero) we need to actually combine two halves of the
        full transform in an awkward way. the real and imaginary spatial mode components, when
        mapped to spatiotemporal modes, retain odd and even frequency indexed modes respectively.

        The best way I have come to think about this is that there are even/odd indexed Fourier transforms
        which each apply to half of the variables, but in order to take a matrix representation we need
        to "shuffle" these two operators in order to be compatible with the way the arrays are formatted.

        Compute FFT matrix by acting on the identity matrix columns (or rows, doesn't matter).
        This is a matrix that takes in a signal (time series) and outputs a set of real valued
        Fourier coefficients in the format [a_0, a_1, b_1, a_2, b_2, a_3, b_3, ... , a_n, b_n, a(N//2)]
        a = real component, b = imaginary component if complex valued Fourier transform was computed instead
        (up to some normalization factor).
        """
        ab_transform_formatter = np.zeros((self.N-1, 2*self.N), dtype=int)
        # binary checkerboard matrix
        ab_transform_formatter[1:-self.n:2, ::2] = 1
        ab_transform_formatter[:-self.n:2, 1::2] = 1
        ab_transform_formatter[-self.n+1::2, 1::2] = 1
        ab_transform_formatter[-self.n::2, ::2] = 1

        full_dft_mat = rfft(np.eye(self.N), norm='ortho', axis=0)
        time_dft_mat = np.concatenate((full_dft_mat[:-1, :].real, full_dft_mat[1:-1, :].imag), axis=0)
        ab_time_dft_matrix = np.insert(time_dft_mat, np.arange(time_dft_mat.shape[1]), time_dft_mat, axis=1)
        full_time_transform_matrix = np.kron(ab_time_dft_matrix*ab_transform_formatter, np.eye(self.m))
        return full_time_transform_matrix

    def inv_time_transform_matrix(self):
        """ Overwrite of parent method """

        ab_transform_formatter = np.zeros((2*self.N, self.N-1), dtype=int)
        ab_transform_formatter[::2, 1:-self.n:2] = 1
        ab_transform_formatter[1::2, :-self.n:2] = 1
        ab_transform_formatter[1::2, -self.n+1::2] = 1
        ab_transform_formatter[::2, -self.n::2] = 1

        imag_idft_matrix = irfft(1j*np.eye((self.N//2)+1), norm='ortho', axis=0)
        real_idft_matrix = irfft(np.eye((self.N//2)+1), norm='ortho', axis=0)
        time_idft_matrix = np.concatenate((real_idft_matrix[:, :-1], imag_idft_matrix[:, 1:-1]), axis=1)
        ab_time_idft_matrix = np.insert(time_idft_matrix, np.arange(time_idft_matrix.shape[0]),
                                        time_idft_matrix, axis=0)

        full_inv_time_transform_matrix = np.kron(ab_time_idft_matrix*ab_transform_formatter,
                                                 np.eye(self.mode_shape[1]))
        return full_inv_time_transform_matrix

    def to_fundamental_domain(self, half='bottom'):
        """ Overwrite of parent method """
        field = self.convert(to='field').state
        if half == 'bottom':
            domain = field[-int(self.N // 2):, :]
        else:
            domain = field[:-int(self.N // 2), :]
        return self.__class__(state=domain, state_type='field', T=self.T / 2.0, L=self.L)


class EquilibriumOrbitKS(AntisymmetricOrbitKS):

    def __init__(self, state=None, state_type='field', T=0., L=0., S=0., **kwargs):
        try:
            if state is not None:
                shp = state.shape
                self.state = state
                self.state_type = state_type
                if state_type == 'modes':
                    self.N, self.M = shp[0], 2*shp[1] + 2
                elif state_type == 'field':
                    self.N, self.M = shp
                elif state_type == 's_modes':
                    self.N, self.M = shp[0], shp[1]+2
            else:
                self.random_initial_condition(L=L, **kwargs)
            self.n, self.m = 1, int(self.M // 2) - 1
            self.mode_shape = (1, self.m)
        except ValueError:
            print('Incompatible type provided for field or modes: 2-D NumPy arrays only')

        self.L = L
        # For uniform save format
        self.T = T
        self.S = S

    def state_vector(self):
        """ Overwrite of parent method """
        return np.concatenate((self.state.reshape(-1, 1),
                               np.array([[float(self.L)]])), axis=0)

    def dx(self, power=1):
        """ Overwrite of parent method """
        qkn = self.wave_vector(self.parameters, power=power)
        c1, c2 = so2_coefficients(power=power)
        elementwise_dxn = np.concatenate((c1*qkn, c2*qkn), axis=1)
        dxn_s_modes = np.multiply(elementwise_dxn, self.convert(to='s_modes').state)
        dxn_s_modes = swap_modes(dxn_s_modes, dimension='space')
        return self.__class__(state=dxn_s_modes, state_type='s_modes', L=self.L)

        qkn = self.wave_vector(self.parameters, power=power)
        if np.mod(order, 2):
            c1, c2 = so2_coefficients(power=power)
            elementwise_dxn = np.tile(np.concatenate((c1*qkn, c2*qkn), axis=1), (self.N, 1))
            dxn_s_modes = np.multiply(elementwise_dxn, self.convert(to='s_modes').state)
            dxn_s_modes = swap_modes(dxn_s_modes, dimension='space')
            return self.__class__(state=dxn_s_modes, state_type='s_modes', T=self.T, L=self.L)
        else:
            c, _ = so2_coefficients(power=power)
            elementwise_dxn = np.tile(c*qkn, (self.N-1, 1))
            dxn_modes = np.multiply(self.convert(to='modes').state, elementwise_dxn)
            return self.__class__(state=dxn_modes, state_type='modes', T=self.T, L=self.L)

    def dx_matrix(self, power=1, **kwargs):
        """ Overwrite of parent method """
        state_type = kwargs.get('state_type', self.state_type)
        # Define spatial wavenumber vector
        if state_type == 'modes':
            dx_n_matrix = np.diag(self.wave_vector(self.parameters).reshape(-1)**order)
        else:
            dx_n_matrix = np.kron(so2_generator(power=power), np.diag(self.wave_vector(self.parameters).reshape(-1)**order))
        return dx_n_matrix

    def elementwise_dx(self):
        """ Overwrite of parent method """
        qk = self.wave_vector(self.parameters)
        return -1.0*qk

    def from_fundamental_domain(self, inplace=False, **kwargs):
        """ Overwrite of parent method """
        half = kwargs.get('half', 'left')
        if half == 'left':
            full_field = np.concatenate((self.state, self.reflection().state), axis=1)
        else:
            full_field = np.concatenate((self.reflection().state, self.state), axis=1)
        return self.__class__(state=full_field, state_type='field', L=2.0*self.L)

    def mode_padding(self, size, inplace=False, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            padded_modes = np.tile(self.state[-1, :].reshape(1, -1), (size, 1))
            eqv = EquilibriumOrbitKS(state=padded_modes, L=self.L)
        else:
            padding_number = int((size-self.M) // 2)
            padding = np.zeros([self.state.shape[0], padding_number])
            eqv_modes = self.convert(to='modes').state
            padded_modes = np.concatenate((eqv_modes, padding), axis=1)
            eqv = EquilibriumOrbitKS(state=padded_modes, L=self.L)
        return eqv

    def mode_truncation(self, size, inplace=False, dimension='space'):
        """ Overwrite of parent method """
        if dimension == 'time':
            truncated_modes = self.state[-size:, :]
            return EquilibriumOrbitKS(state=truncated_modes, L=self.L)
        else:
            truncate_number = int(size // 2) - 1
            truncated_modes = self.state[:, :truncate_number]
            return EquilibriumOrbitKS(state=truncated_modes, state_type=self.state_type, L=self.L)

    def precondition(self, current, parameter_constraints=True, **kwargs):
        """ Overwrite of parent method """
        qk_matrix = self.elementwise_dxn(self.parameters)
        p_matrix = qk_matrix**2 + qk_matrix**4
        current.state = np.divide(current.state, p_matrix)

        if not parameter_constraints:
            current.L = current.L/(self.L**4)
        return current

    def nonlinear(self, other):
        """ Overwrite of parent method """
        orbit_nonlinear = self.convert(to='field').statemul(other.convert(to='field'))
        return 0.5 * orbit_nonlinear.dx().convert(to='modes')

    def random_initial_condition(self, L, **kwargs):
        """ Initial a set of random spatiotemporal Fourier modes

        Parameters
        ----------
        **kwargs
            time_scale : int
                The number of temporal frequencies to keep after truncation.
            space_scale : int
                The number of spatial frequencies to get after truncation.
        Returns
        -------
        self :
            OrbitKS whose state has been modified to be a set of random Fourier modes.

        Notes
        -----
        Anecdotal evidence suggests that "worse" initial conditions converge more often to solutions of the
        predetermined symmetry group. In other words it's better to start far away from the chaotic attractor
        because then it is less likely to start near equilibria. Spatial scale currently unused, still testing
        for the best random fields.

        """
        spectrum_type = kwargs.get('spectrum', 'random')
        if L == 0.:
            self.L = 22 + 44*np.random.rand(1)
        else:
            self.L = L
        self.N = 1
        self.n = 1
        self.M = kwargs.get('M', np.max([2**(int(np.log2(self.L))), 32]))
        self.m = int(self.M // 2) - 1
        space_scale = np.min([kwargs.get('space_scale', self.m), self.m])
        if spectrum_type == 'gaussian':
            rmodes = np.random.randn(self.N-1, self.m)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0**mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)
        else:
            rmodes = np.random.randn(self.N-1, self.m)
            mollifier_exponents = space_scale + -1 * np.tile(np.arange(0, self.m)+1, (self.n, 1))
            mollifier = 10.0**mollifier_exponents
            mollifier[:, :space_scale] = 1
            mollifier = np.concatenate((mollifier, mollifier), axis=0)
            mollifier = np.concatenate((np.ones([1, ]), mollifier), axis=0)
            self.state = np.multiply(mollifier, rmodes)

        self.mode_shape = rmodes.shape
        self.convert(to='field', inplace=True)
        tmp = self // (1.0/4.0)
        self.state = tmp.state
        self.convert(to='modes', inplace=True)
        return self

    def rnonlinear(self, other):
        """ Overwrite of parent method """
        return -1.0*self.convert(to='field').statemul(other.dx().convert(to='field')).convert(to='modes')

    def rmatvec(self, other, parameter_constraints=False, preconditioning=True, **kwargs):
        """ Overwrite of parent method """
        # For specific computation of the linear component instead
        # of arbitrary derivatives we can optimize the calculation by being specific.
        qk_matrix = self.elementwise_dxn(self.parameters)
        elementwise_qk2 = -1.0*qk_matrix**2
        elementwise_qk4 = qk_matrix**4
        dx2 = np.multiply(elementwise_qk2, other.state)
        dx4 = np.multiply(elementwise_qk4, other.state)
        linear_component = dx2 + dx4
        orbit_linear = self.__class__(state=linear_component, T=self.T, L=self.L, S=self.S)
        orbit_field = self.convert(to='field')

        orbit_rmatvec = orbit_linear + orbit_field.rnonlinear(other)

        if not parameter_constraints:
            dx2_self = np.multiply(elementwise_qk2, self.state)
            dx4_self = np.multiply(elementwise_qk4, self.state)
            dfdl_linear = ((-2.0/self.L)*dx2_self + (-4.0/self.L)*dx4_self)
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear
            orbit_rmatvec.L = np.dot(dfdl.ravel(), other.state.ravel())

        if preconditioning:
            p_matrix = 1.0 / (qk_matrix**2 + qk_matrix**4)
            orbit_rmatvec.state = np.multiply(orbit_rmatvec.state, p_matrix)

            if not parameter_constraints:
                orbit_rmatvec.L = orbit_rmatvec.L/(self.L**4)

        return orbit_rmatvec

    def spatiotemporal_mapping(self):
        """ Overwrite of parent method """
        # For specific computation of the linear component instead
        # of arbitrary derivatives we can optimize the calculation by being specific.
        qk_matrix = self.elementwise_dxn(self.parameters)
        elementwise_dx2dx4 = -1.0*qk_matrix**2 + qk_matrix**4
        linear = np.multiply(elementwise_dx2dx4, self.state)
        # Convert state information to field inplace; derivative operation switches this back to modes?
        orbit_field = self.convert(to='field')
        nonlinear = orbit_field.nonlinear(orbit_field).state
        orbit_mapping = self.__class__(state=(linear + nonlinear), L=self.L)
        return orbit_mapping

    def inv_time_transform_matrix(self):
        """ Overwrite of parent method

        Notes
        -----
        Originally this transform just selected the antisymmetric spatial modes (imaginary component),
        but in order to be consistent with all other time transforms, I will actually apply the normalization
        constant associated with a forward in time transformation. The reason for this is for comparison
        of states between different subclasses.
        """
        return (1./np.sqrt(self.N)) * np.concatenate((0*np.eye(self.m), np.eye(self.m)), axis=0)

    def time_transform_matrix(self):
        """ Overwrite of parent method """
        return (1./np.sqrt(self.N)) * np.concatenate((0*np.eye(self.m), np.eye(self.m)), axis=1)

    def time_transform(self, inplace=False):
        """ Overwrite of parent method

        Notes
        -----
        Taking the RFFT, with orthogonal normalization, of a constant time series defined on N points is equivalent
        to multiplying the constant value by sqrt(N). This is because the transform sums over N repeats of the same
        value and then divides by 1/sqrt(N). i.e. (N * C)/sqrt(N) = sqrt(N) * C. Therefore we can save some time by
        just doing this without calling the rfft function.
        """
        # Select the nonzero (imaginary) components of modes and transform in time (w.r.t. axis=0).
        spacetime_modes = np.sqrt(self.N) * self.state[-1, -self.m:].reshape(1, -1)
        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return EquilibriumOrbitKS(state=spacetime_modes, state_type='modes', L=self.L)

    def inv_time_transform(self, inplace=False):
        """ Overwrite of parent method

        Notes
        -----
        Taking the IRFFT, with orthogonal normalization is equivalent to dividing by the normalization constant; because
        there would only be
        """
        real = np.zeros(self.state.shape)
        imaginary = self.state
        spatial_modes = np.concatenate((real, imaginary), axis=1)
        if inplace:
            self.state = spatial_modes
            self.state_type = 's_modes'
            return self
        else:
            return EquilibriumOrbitKS(state=spatial_modes, state_type='s_modes', L=self.L)

    def time_transform_matrix(self):
        """ Inverse Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a set of spatial modes

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        Formatted such that the input is u_k(t) (spatial modes). The "original" DFT matrix would
        """
        dft_mat = rfft(np.eye(self.N), norm='ortho', axis=0)
        time_dft_mat = np.concatenate((dft_mat[:-1, :].real,
                                       dft_mat[1:-1, :].imag), axis=0)
        ab_time_dft_mat = np.insert(time_dft_mat,
                                    np.arange(time_dft_mat.shape[1]),
                                    np.zeros([time_dft_mat.shape[0], time_dft_mat.shape[1]]),
                                    axis=1)
        return np.kron(ab_time_dft_mat, np.eye(self.m))

    def inv_time_transform_matrix(self):
        """ Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        idft_mat_real = irfft(np.eye(self.N//2 + 1), norm='ortho', axis=0)
        idft_mat_imag = irfft(1j*np.eye(self.N//2 + 1), norm='ortho', axis=0)
        time_idft_mat = np.concatenate((idft_mat_real[:, :-1],
                                        idft_mat_imag[:, 1:-1]), axis=1)
        ab_time_idft_mat = np.insert(time_idft_mat,
                                     np.arange(time_idft_mat.shape[0]),
                                     np.zeros([time_idft_mat.shape[0], time_idft_mat.shape[1]]),
                                     axis=0)
        return np.kron(ab_time_idft_mat, np.eye(self.m))

    def time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = rfft(self.state, norm='ortho', axis=0)
        modes_real = modes.real[:-1, -self.m:]
        modes_imag = modes.imag[1:-1, -self.m:]
        spacetime_modes = np.concatenate((modes_real, modes_imag), axis=0)

        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', T=self.T, L=self.L, S=self.S)

    def inv_time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.

        modes = self.state
        time_real = modes[:-self.n, :]
        time_imaginary = 1j*np.concatenate((np.zeros([1, self.m]), modes[-self.n:, :]), axis=0)
        spacetime_modes = np.concatenate((time_real + time_imaginary, np.zeros([1, self.m])), axis=0)
        imaginary_space_modes = irfft(spacetime_modes, norm='ortho', axis=0)
        space_modes = np.concatenate((np.zeros(imaginary_space_modes.shape), imaginary_space_modes), axis=1)

        if inplace:
            self.state = space_modes
            self.state_type = 's_modes'
            return self
        else:
            return self.__class__(state=space_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def to_fundamental_domain(self, half='left', **kwargs):
        """ Overwrite of parent method """
        if half == 'left':
            return EquilibriumOrbitKS(state=self.convert(to='field').state[:, :-int(self.M//2)],
                                    state_type='field', L=self.L / 2.0)
        else:
            return EquilibriumOrbitKS(state=self.convert(to='field').state[:, -int(self.M//2):],
                                    state_type='field', L=self.L / 2.0)


class RelativeEquilibriumOrbitKS(RelativeOrbitKS):

    def __init__(self, state=None, state_type='field', T=0., L=0., S=0., frame='comoving', **kwargs):
        super().__init__(state=state, state_type=state_type, T=T, L=L, S=S, **kwargs)
        self.frame = frame
        self.mode_shape = (1, self.M-2)


    def calculate_shift(self, inplace=False):
        """ Calculate the phase difference between the spatial modes at t=0 and t=T """
        s_modes = self.convert(to='s_modes').state
        modes1 = s_modes[-1, :].reshape(-1, 1)
        modes0 = s_modes[0, :].reshape(-1, 1)
        angle = np.arccos(np.dot(np.transpose(modes1), modes0)/(np.linalg.norm(modes1)*np.linalg.norm(modes0)))
        # Just to prevent over-winding
        angle = np.sign(angle)*np.mod(angle, 2*pi)
        shift = float((self.L / (2 * pi)) * angle)
        if inplace:
            self.S = shift
            return None
        else:
            return shift

    def comoving_mapping_component(self):
        """ Co-moving frame component of spatiotemporal mapping """
        return -1.0 * (self.S / self.T)*self.dx()

    def comoving_matrix(self):
        """ Operator that constitutes the co-moving frame term """
        return -1.0 * (self.S / self.T)*self.dx_matrix()

    def dt(self, power=1):
        """ A time derivative of the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        orbit_dtn : OrbitKS or subclass instance
            The class instance whose state is the time derivative in
            the spatiotemporal mode basis.
        """
        if self.frame == 'comoving':
            return np.zeros(self.state.shape)
        else:
            raise ValueError(
                'Attempting to compute time derivative of '+self.__str__+ 'in physical reference frame.')

    def dt_matrix(self, power=1):
        """ The time derivative matrix operator for the current state.

        Parameters
        ----------
        power :int
            The order of the derivative.

        Returns
        ----------
        wk_matrix : matrix
            The operator whose matrix-vector product with spatiotemporal
            Fourier modes is equal to the time derivative. Used in
            the construction of the Jacobian operator.

        Notes
        -----
        Before the kronecker product, the matrix dt_n_matrix is the operator which would correctly take the
        time derivative of a single set of N-1 temporal modes. Because we have space as an extra dimension,
        we need a number of copies of dt_n_matrix equal to the number of spatial frequencies.
        """
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        dt_n_matrix = np.kron(so2_generator(power=power), np.diag(self.frequency_vector(self.parameters,
                                                                                        power=order).ravel()))
        # Zeroth frequency was not included in frequency vector.
        dt_n_matrix = block_diag([[0]], dt_n_matrix)
        # Take kronecker product to account for the number of spatial modes.
        spacetime_dtn = np.kron(dt_n_matrix, np.eye(self.mode_shape[1]))
        return spacetime_dtn

    def elementwise_dx(self):
        return super().elementwise_dx()[0, :].reshape(1, -1)

    def from_fundamental_domain(self):
        """ For compatibility purposes with plotting and other utilities """
        return self.change_reference_frame()

    def jacobian(self, parameter_constraints=(False, False, False)):
        """ Jacobian that includes the spatial translation term for relative periodic tori

        Parameters
        ----------
        parameter_constraints : (bool, bool, bool)
            Determines whether or not the various parameters, period, spatial period, spatial shift, (T,L,S)
            are variables or not.

        Returns
        -------
        matrix :
            Jacobian matrix for relative periodic tori.
        """
        self.convert(to='modes', inplace=True)
        # The linearization matrix of the governing equations.
        jac_ = self.jac_lin() + self.jac_nonlin()

        # If period is not fixed, need to include dF/dT for changes to period.
        if not parameter_constraints[0]:
            ds = -1.0 * (self.S / self.T)*self.dx()
            dfdt = (-1.0 / self.T)*(ds.state.reshape(-1, 1))
            jac_ = np.concatenate((jac_, dfdt), axis=1)

        # If period is not fixed, need to include dF/dL for changes to period.
        if not parameter_constraints[1]:
            orbit_field = self.convert(to='field')
            qk_matrix = self.elementwise_dxn(self.parameters)
            dx2 = np.multiply(-1.0*qk_matrix**2, self.state)
            dx4 = np.multiply(qk_matrix**4, self.state)
            s = self.comoving_mapping_component().state
            dfdl_linear = (-2.0/self.L)*dx2 + (-4.0/self.L)*dx4 + (-1.0/self.L) * s
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear
            jac_ = np.concatenate((jac_, dfdl.reshape(-1, 1)), axis=1)

        if not parameter_constraints[2]:
            dfds = (-1.0 / self.T)*self.dx().state
            jac_ = np.concatenate((jac_, dfds.reshape(-1, 1)), axis=1)

        return jac_

    def jac_lin(self):
        """ Extension of the OrbitKS method that includes the term for spatial translation symmetry"""
        return self.dx_matrix(power=2) + self.dx_matrix(power=4) + self.comoving_matrix()

    def matvec(self, other, parameter_constraints=(False, False, False), preconditioning=True, **kwargs):
        """ Extension of parent class method

        Parameters
        ----------
        other : RelativeOrbitKS
            RelativeOrbitKS instance whose state represents the vector in the matrix-vector multiplication.
        parameter_constraints : tuple of bool
            Determines whether to include period and spatial period
            as variables.
        preconditioning : bool
            Whether or not to apply (left) preconditioning P (Ax)

        Returns
        -------
        RelativeOrbitKS
            RelativeOrbitKS whose state and other parameters result from the matrix-vector product.

        Notes
        -----
        Equivalent to computation of (v_t + v_xx + v_xxxx + phi * v_x) + d_x (u .* v)
        The reason for all of the repeated code is that the co-moving terms re-use the same matrices
        as the other terms; this prevents additional function calls.

        Additionally, note that for the partial derivatives w.r.t. params, T,L,S, we still need the derivative
        of d(u_t + u_xx + u_xxxx + uu_x)/dPARAM;
        """
        qk_matrix = self.elementwise_dxn(self.parameters)
        elementwise_qk2 = -1.0*qk_matrix**2
        elementwise_qk4 = qk_matrix**4

        dx2 = np.multiply(elementwise_qk2, other.state)
        dx4 = np.multiply(elementwise_qk4, other.state)
        s = -1.0 * (self.S / self.T)*swap_modes(np.multiply(qk_matrix, other.state))

        linear_term = dx2 + dx4 + s
        orbit_linear = self.__class__(state=linear_term, state_type='modes', T=self.T, L=self.L, S=self.S)
        orbit_field = self.convert(to='field')

        # Convert state information to field inplace; derivative operation switches this back to modes?
        orbit_matvec = orbit_linear + 2 * orbit_field.nonlinear(other)

        if not parameter_constraints[0]:
            # the partial derivations w.r.t. params
            s_self = (-1.0 * self.S / self.T)*swap_modes(np.multiply(qk_matrix, self.state))
            dfdt = (-1.0 / self.T)*(s_self)
            orbit_matvec.state = orbit_matvec.state + other.T*dfdt

        if not parameter_constraints[1]:
            dx2_self = np.multiply(elementwise_qk2, self.state)
            dx4_self = np.multiply(elementwise_qk4, self.state)
            s_self = (-1.0 * self.S / self.T)*swap_modes(np.multiply(qk_matrix, self.state))
            dfdl_linear = (-2.0/self.L)*dx2_self + (-4.0/self.L)*dx4_self + (-1.0/self.L)*s_self
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear
            # Derivative of mapping with respect to T is the same as -1/T * u_t
            orbit_matvec.state = orbit_matvec.state + other.L*dfdl

        if not parameter_constraints[2]:
            orbit_matvec.state = (orbit_matvec.state
                                  + other.S*(-1.0 / self.T)*swap_modes(np.multiply(qk_matrix, self.state)))

        if preconditioning:
            p_matrix = 1.0 / (qk_matrix**2 + qk_matrix**4)
            orbit_matvec.state = np.multiply(orbit_matvec.state, p_matrix)

        return orbit_matvec

    def rmatvec(self, other, parameter_constraints=(False, False, False), preconditioning=True, **kwargs):
        """ Extension of the parent method to RelativeOrbitKS """
        # For specific computation of the linear component instead
        # of arbitrary derivatives we can optimize the calculation by being specific.
        qk_matrix = self.elementwise_dxn(self.parameters)
        elementwise_qk2 = -1.0*qk_matrix**2
        elementwise_qk4 = qk_matrix**4

        dx2 = np.multiply(elementwise_qk2, other.state)
        dx4 = np.multiply(elementwise_qk4, other.state)
        s = (self.S / self.T) * swap_modes(np.multiply(qk_matrix, other.state))
        linear_component = dx2 + dx4 + s
        orbit_field = self.convert(to='field')
        nonlinear_component = orbit_field.rnonlinear(other).state
        orbit_rmatvec = self.__class__(state=(linear_component + nonlinear_component))

        if not parameter_constraints[0]:
            s_self = (-1.0 * self.S / self.T)*swap_modes(np.multiply(qk_matrix, self.state))
            dfdt = (-1.0 / self.T)*(s_self)
            orbit_rmatvec.T = np.dot(dfdt.ravel(), other.state.ravel())

        if not parameter_constraints[1]:
            dx2_self = np.multiply(elementwise_qk2, self.state)
            dx4_self = np.multiply(elementwise_qk4, self.state)
            s_self = (-1.0 * self.S / self.T) * swap_modes(np.multiply(qk_matrix, self.state))
            dfdl_linear = (-2.0/self.L)*dx2_self + (-4.0/self.L)*dx4_self + (-1.0/self.L)*s_self
            dfdl_nonlinear = (-1.0/self.L) * orbit_field.nonlinear(orbit_field).state
            dfdl = dfdl_linear + dfdl_nonlinear

            # Derivative of mapping with respect to T is the same as -1/T * u_t
            orbit_rmatvec.L = np.dot(dfdl.ravel(), other.state.ravel())

        if not parameter_constraints[2]:
            orbit_rmatvec.S = ((-1.0 / self.T)
                               * np.dot(swap_modes(np.multiply(qk_matrix, self.state)).ravel(), other.state.ravel()))

        if preconditioning:
            p_matrix = 1.0 / (qk_matrix**2 + qk_matrix**4)
            orbit_rmatvec.state = np.multiply(orbit_rmatvec.state, p_matrix)

            if not parameter_constraints[0]:
                orbit_rmatvec.T = orbit_rmatvec.T / self.T

            if not parameter_constraints[1]:
                orbit_rmatvec.L = orbit_rmatvec.L / (self.L**4)

        return orbit_rmatvec

    def random_initial_condition(self, T, L, *args, **kwargs):
        """ Extension of parent modes to include spatial-shift initialization """
        super().random_initial_condition(T, L, **kwargs)
        self.calculate_shift(inplace=True)
        if args[0] == 0.0:
            # Assign random proportion of L with random sign as the shift if none provided.
            self.S = ([-1, 1][int(2*np.random.rand())])*np.random.rand()*self.L
        else:
            self.S = args[0]
        return self

    def spatiotemporal_mapping(self):
        """ Extension of OrbitKS method to include co-moving frame term. """
        return super().spatiotemporal_mapping() + self.comoving_mapping_component()

    def state_vector(self):
        """ Vector which completely describes the orbit."""
        return np.concatenate((self.state.reshape(-1, 1),
                               np.array([[float(self.T)]]),
                               np.array([[float(self.L)]]),
                               np.array([[float(self.S)]])), axis=0)

    def status(self):
        """ Check if orbit has meaningful translation symmetry """
        if np.abs(self.S) < 10**-4:
            return OrbitKS(state=self.state, state_type=self.state_type, T=self.T, L=self.L).status()
        else:
            return super().status()

    def inv_time_transform_matrix(self):
        """ Overwrite of parent method

        Notes
        -----
        Originally this transform just selected the antisymmetric spatial modes (imaginary component),
        but in order to be consistent with all other time transforms, I will actually apply the normalization
        constant associated with a forward in time transformation. The reason for this is for comparison
        of states between different subclasses.
        """
        (1./np.sqrt(self.N)) * np.concatenate((0*np.eye(self.m), np.eye(self.m)), axis=0)
        return

    def time_transform_matrix(self):
        """ Overwrite of parent method """
        return (1./np.sqrt(self.N)) * np.concatenate((0*np.eye(self.m), np.eye(self.m)), axis=1)

    def time_transform(self, inplace=False):
        """ Overwrite of parent method """
        # Select the nonzero (imaginary) components of modes and transform in time (w.r.t. axis=0)
        spacetime_modes = (1./np.sqrt(self.N)) * self.state
        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', L=self.L)

    def inv_time_transform(self, inplace=False):
        """ Overwrite of parent method """
        real = np.zeros(self.state.shape)
        imaginary = self.state
        spatial_modes = np.concatenate((real, imaginary), axis=1)
        if inplace:
            self.state = spatial_modes
            self.state_type = 's_modes'
            return self
        else:
            return EquilibriumOrbitKS(state=spatial_modes, state_type='s_modes', L=self.L)

    def time_transform_matrix(self):
        """ Inverse Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a set of spatial modes

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        dft_mat = rfft(np.eye(self.N), norm='ortho', axis=0)
        time_dft_mat = np.concatenate((dft_mat[:-1, :].real,
                                       dft_mat[1:-1, :].imag), axis=0)
        ab_time_dft_mat = np.insert(time_dft_mat,
                                    np.arange(time_dft_mat.shape[1]),
                                    np.zeros([time_dft_mat.shape[0], time_dft_mat.shape[1]]),
                                    axis=1)
        return np.kron(ab_time_dft_mat, np.eye(self.m))

    def inv_time_transform_matrix(self):
        """ Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.
        """
        idft_mat_real = irfft(np.eye(self.N//2 + 1), norm='ortho', axis=0)
        idft_mat_imag = irfft(1j*np.eye(self.N//2 + 1), norm='ortho', axis=0)
        time_idft_mat = np.concatenate((idft_mat_real[:, :-1],
                                        idft_mat_imag[:, 1:-1]), axis=1)
        ab_time_idft_mat = np.insert(time_idft_mat,
                                     np.arange(time_idft_mat.shape[0]),
                                     np.zeros([time_idft_mat.shape[0], time_idft_mat.shape[1]]),
                                     axis=0)
        return np.kron(ab_time_idft_mat, np.eye(self.m))

    def time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.
        modes = rfft(self.state, norm='ortho', axis=0)
        modes_real = modes.real[:-1, -self.m:]
        modes_imag = modes.imag[1:-1, -self.m:]
        spacetime_modes = np.concatenate((modes_real, modes_imag), axis=0)

        if inplace:
            self.state = spacetime_modes
            self.state_type = 'modes'
            return self
        else:
            return self.__class__(state=spacetime_modes, state_type='modes', T=self.T, L=self.L, S=self.S)

    def inv_time_transform(self, inplace=False):
        """ Spatial Fourier transform

        Parameters
        ----------
        inplace : bool
            Whether or not to perform the operation "in place" (overwrite the current state if True).

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        # Take rfft, accounting for unitary normalization.

        modes = self.state
        time_real = modes[:-self.n, :]
        time_imaginary = 1j*np.concatenate((np.zeros([1, self.m]), modes[-self.n:, :]), axis=0)
        spacetime_modes = np.concatenate((time_real + time_imaginary, np.zeros([1, self.m])), axis=0)
        imaginary_space_modes = irfft(spacetime_modes, norm='ortho', axis=0)
        space_modes = np.concatenate((np.zeros(imaginary_space_modes.shape), imaginary_space_modes), axis=1)

        if inplace:
            self.state = space_modes
            self.state_type = 's_modes'
            return self
        else:
            return self.__class__(state=space_modes, state_type='s_modes', T=self.T, L=self.L, S=self.S)

    def to_fundamental_domain(self):
        return self.change_reference_frame()

