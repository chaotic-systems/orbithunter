from math import pi
from ..core import Orbit
from scipy.fft import rfft, irfft, rfftfreq
from scipy.linalg import block_diag
from scipy.sparse.linalg import LinearOperator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings


__all__ = [
    "OrbitKS",
    "RelativeOrbitKS",
    "ShiftReflectionOrbitKS",
    "AntisymmetricOrbitKS",
    "EquilibriumOrbitKS",
    "RelativeEquilibriumOrbitKS",
]


class OrbitKS(Orbit):
    """
    Base class for orbits of the Kuramoto-Sivashinsky equation.

    The Kuramoto-Sivashinsky equation is a fourth order partial differential equation that serves as a simplified
    testing ground for the more complicated Navier-Stokes equation. It's form in configuration space, where the
    state variable :math:`u(t,x)` is typically imagined as a velocity field of a laminar flame front.
    It's spatiotemporal version with doubly periodic boundary conditions takes the form

    $u_t + u_{xx} + u_{xxxx} + 1/2(u^2)_x = 0$

    with boundary conditions

    $u(t, x) = u(t + T, x) = u(t, x+L) = u(t+T, x+L)$.

    This class and its subclasses is used to find solutions to the system of differential algebraic equations (DAEs)
    which result from applying a discrete Fourier transform in both space and time.

    The 'state' in configuration space is ordered such that when in the physical basis, the last row
    corresponds to 't=0'. This results in an extra negative sign when computing time derivatives.
    This convention was chosen because it is conventional to display positive time as 'up'. This convention prevents
    errors due to flipping fields up and down.

    To define an orbit, the configuration space (spatiotemporal dimensions) or tile must be defined. The
    unconventional approach of this package is to keep these domain dimensions as free variables.

    The only additional parameter beyond the dimensions is a spatial shift parameter for solutions with continuous
    spatial translation symmetry; it only applies to :class:`orbithunter.ks.orbits.RelativeOrbitKS`
    and :class:`orbithunter.ks.orbits.RelativeEquilibriumOrbitKS`. Its inclusion in the class
    :class:`orbithunter.ks.orbits.OrbitKS` is due to the ability to convert between Orbit types.
    The various subclasses represent symmetry invariant subspaces. Due to the nature of
    subspaces, it is numerically possible to find, for example, solutions with spatial reflection symmetry using
    OrbitKS. The discrete symmetry invariant orbits are literaly subspaces of solutions; any subclass member
    can be found using its parent class.

    Historically, only `adj` and `lstsq` were used, in combination, for OrbitKS and its subclasses:
    All possible methods include:

    - 'adj'
    - 'newton_descent'
    - 'lstsq'
    - 'lsqr'
    - 'lsmr'
    - 'bicg'
    - 'bicgstab'
    - 'gmres'
    - 'lgmres'
    - 'cg'
    - 'cgs'
    - 'qmr'
    - 'minres'
    - 'gcrotmk'
    - 'cg_min'
    - 'bfgs'
    - 'newton-cg'
    - 'l-bfgs-b'
    - 'tns'
    - 'slsqp'

    .. warning::
        If dimensions change by dramatic/nonsensible amounts then preconditioning=True can be used with certain methods
        (most notably, 'adj') to account for very large parameter gradients.

    .. warning::
        The following are supported but NOT recommended for the KSE.

        - 'nelder-mead' (very very slow)
        - 'powell' (very slow)
        - 'cobyla' (slow),

    See Also
    --------
    :class:`orbithunter.core.Orbit`

    """

    def periodic_dimensions(self):
        """
        Bools indicating whether or not dimension is periodic.

        Returns
        -------
        tuple of bool
            Flags for whether dimensions are periodic or not.

        Notes
        -----
        Non-static method to match the signature of :class:`orbithunter.core.Orbit`

        """
        return True, True

    @staticmethod
    def _default_shape():
        """
        The default array shape when dimensions are not specified.

        Returns
        -------
        tuple of int :
            Tuple containing the default shape.

        Notes
        -----
        (32, 32) is used because this is the shape that generally works for small tile sizes.

        """
        return 32, 32

    @staticmethod
    def minimal_shape_increments():
        """
        The smallest valid increment to change the discretization by.

        Returns
        -------
        tuple of int :
            The smallest valid increments to changes in discretization size which retain all functionality.

        """
        return 2, 2

    @staticmethod
    def minimal_shape():
        """
        The smallest possible compatible discretization to have full functionality.

        Returns
        -------
        tuple of int :
            The minimal shape for this class.
        """

        return 2, 4

    @staticmethod
    def bases_labels():
        """
        Labels of the different bases produced by transforms.

        Returns
        -------
        tuple of str

        """
        return "field", "spatial_modes", "modes"

    @staticmethod
    def discretization_labels():
        """
        Strings to use to label dimensions/periods

        Returns
        -------
        tuple of str
            The labels for each dimension's number of collocation points (dimension in field basis)

        """
        return "n", "m"

    @staticmethod
    def parameter_labels():
        """
        Labels of all parameters

        Returns
        -------
        tuple of str

        Notes
        -----
        The parameter 's' is never used outside of RelativeOrbitKS and RelativeEquilibriumKS, however, it is included
        for compatibility between conversion between symmetry types.

        """
        return "t", "x", "s"

    @staticmethod
    def dimension_labels():
        """
        Strings to use to label dimensions/periods.

        """
        return "t", "x"

    def orbit_vector(self):
        """
        Vector which completely specifies the orbit, contains state information and parameters.

        Returns
        -------
        np.ndarray :
            Column vector array comprised of all (valid) state variables (self.state and self.parameters). Shift 's'
            is never valid for this class and hence not included.

        """
        return np.concatenate(
            (
                self.state.reshape(-1, 1),
                np.array([[float(self.t)]]),
                np.array([[float(self.x)]]),
            ),
            axis=0,
        )

    def transform(self, to=None, array=False, inplace=False):
        """
        Transform current state to a different basis.

        Parameters
        ----------
        to : str
            One of the following: 'field', 'spatial_modes', 'modes'. Specifies the basis which the orbit will be
            converted to.
        array : bool
            Whether to return np.ndarray or OrbitKS instance

        Raises
        ----------
        ValueError
            Raised if the provided basis is unrecognizable or state is an empty array.

        Returns
        ----------
        OrbitKS :
            The class instance in the new basis.

        Notes
        -----
        This method is just a wrapper for different Fourier transforms. It's purpose is to remove the
        need for the user to keep track of the basis by hand. Transforms should never be used directly. For a state in
        the field basis, self.transform(to='modes') is equivalent to self._space_transform()._time_transform()
        If 'to'==self.basis then self (NOT a copy) is returned. While this could cause unintentional overwrites,
        the user should not be transforming to a basis it is already in anyway. However, for practical purposes,
        remembering the basis at all times can be troublesome and so not error is raised if this is true.

        """
        if self.state is None:
            raise ValueError(
                "Trying to transform an unpopulated {} instance.".format(str(self))
            )
        elif self.basis is None:
            raise ValueError(
                "Trying to transform state with unknown basis".format(str(self))
            )

        if to == "field":
            if self.basis == "spatial_modes":
                return self._inv_space_transform(array=array, inplace=inplace)
            elif self.basis == "modes":
                return self._inv_spacetime_transform(array=array, inplace=inplace)
            else:
                if array:
                    return self.state
                else:
                    return self
        elif to == "spatial_modes":
            if self.basis == "field":
                return self._space_transform(array=array, inplace=inplace)
            elif self.basis == "modes":
                return self._inv_time_transform(array=array, inplace=inplace)
            else:
                if array:
                    return self.state
                else:
                    return self
        elif to == "modes":
            if self.basis == "spatial_modes":
                return self._time_transform(array=array, inplace=inplace)
            elif self.basis == "field":
                return self._spacetime_transform(array=array, inplace=inplace)
            else:
                if array:
                    return self.state
                else:
                    return self
        else:
            raise ValueError("Trying to transform to unrecognized basis.")

    def dt(self, order=1, array=False, **kwargs):
        """
        Spectral time derivatives of the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.
        array : bool
            Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
            Orbit instances.

        Returns
        ----------
        orbit_dtn : OrbitKS
            The class instance whose state is the time derivative in the spatiotemporal mode basis.

        Notes
        -----
        It is very often the case that the derivative in the field basis is desired. Instead of forcing
        the user to write self.transform(to='modes').dt().transform(to='field'), the functions are written
        to incur extra overhead by transforming behind the scenes. To avoid unnecessary slow down, the OrbitKS
        instances must be in the modes basis to compute self.eqn()

        """
        if kwargs.get("inplace", False):
            return_basis = kwargs.get("return_basis", self.basis)
            # Need mode basis to compute derivatives
            self.transform(to="modes", array=True, inplace=True)
            # Elementwise multiplication of modes with frequencies, this is the derivative. Uses numpy broadcasting.
            self.state = temporal_frequencies(self.t, self.n, order) * self.state

            # If the order of the derivative is odd, then imaginary component and real components switch. Need to
            # account for this for our real-valued transforms.
            if np.mod(order, 2):
                self.state = swap_modes(self.state, axis=0)

            # To for numerical efficiency, NumPy arrays can be returned.
            if array:
                return self.transform(to=return_basis, inplace=True).state
            else:
                return self.transform(to=return_basis, inplace=True)
        else:
            # Need mode basis to compute derivatives
            modes = self.transform(to="modes", array=True)
            # Elementwise multiplication of modes with frequencies, this is the derivative. Uses numpy broadcasting.
            dtn_modes = temporal_frequencies(self.t, self.n, order) * modes

            # If the order of the derivative is odd, then imaginary component and real components switch. Need to
            # account for this for our real-valued transforms.
            if np.mod(order, 2):
                dtn_modes = swap_modes(dtn_modes, axis=0)

            # To for numerical efficiency, NumPy arrays can be returned.
            if array:
                return dtn_modes
            else:
                # return the derivative in an instance whose basis is equivalent to the original basis of self.
                orbit_dtn = self.__class__(
                    **{**vars(self), "state": dtn_modes, "basis": "modes"}
                )
                return orbit_dtn.transform(to=self.basis)

    def dx(self, **kwargs):
        """
        Spatial derivative of the current state.

        Parameters
        ----------
        kwargs :
            order :int
                The order of the derivative.
            array : bool
                Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
                Orbit instances.
            computation_basis : str
                The basis in which to compute the derivative.
            return_basis : str
                Which basis to return the ShiftReflectionOrbitKS in, if array=False.


        Returns
        ----------
        orbit_dxn : OrbitKS or subclass instance
            Class instance whose spatiotemporal state represents the spatial derivative in the
            the basis of the original state.

        Notes
        -----
        It is very often the case that the derivative in the field basis is desired. Instead of forcing
        the user to write self.transform(to='modes').dt().transform(to='field'), the functions are written
        to incur extra overhead by transforming behind the scenes. To avoid unnecessary slow down, the OrbitKS
        instances must be in the modes basis to compute self.eqn()

        """
        # can compute spatial derivative in spatial mode or spatiotemporal mode basis. spatial_modes basis is required
        # for orbits with discrete symmetry as they are orthogonal to the dx() direction.
        computation_basis = kwargs.get("computation_basis", "modes")
        inplace = kwargs.get("inplace", False)
        order = kwargs.get("order", 1)
        array = kwargs.get("array", False)
        return_basis = kwargs.get("return_basis", self.basis)

        if inplace:
            self.transform(to=computation_basis, array=True, inplace=True)
            self.state = (
                spatial_frequencies(self.x, self.m, order)[:, : self.state.shape[1]]
                * self.state
            )
            # If the order of the differentiation is odd, need to swap imaginary and real components.
            if np.mod(order, 2):
                self.state = swap_modes(self.state, axis=1)

            self.transform(to=return_basis, inplace=True)
            if array:
                return self.state
            else:
                return self

        else:
            if computation_basis == "spatial_modes":
                modes = self.transform(to="spatial_modes", array=True)
                dxn_modes = spatial_frequencies(self.x, self.m, order) * modes
            elif computation_basis == "modes":
                modes = self.transform(to="modes", array=True)
                # Slicing is a correction which only affects discrete symmetry orbits.
                dxn_modes = (
                    spatial_frequencies(self.x, self.m, order)[:, : modes.shape[1]]
                    * modes
                )
            else:
                raise ValueError(
                    f"{str(self)}.dx(computation_basis={computation_basis}); invalid basis for spectral differentiation. "
                )

            # If the order of the differentiation is odd, need to swap imaginary and real components.
            if np.mod(order, 2):
                dxn_modes = swap_modes(dxn_modes, axis=1)

            if array:
                return dxn_modes
            else:
                orbit_dxn = self.__class__(
                    **{**vars(self), "state": dxn_modes, "basis": computation_basis}
                )
                return orbit_dxn.transform(to=return_basis, inplace=True)

    def eqn(self, **kwargs):
        """
        Instance whose state is the Kuramoto-Sivashinsky equation evaluated at the current state

        kwargs :
            Unused, to match signature of Orbit

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is the spatiotamporal fourier modes resulting from the calculation of the K-S equation:
            OrbitKS.state = u_t + u_xx + u_xxxx + 1/2 (u^2)_x

        """
        # to be efficient, should be in modes basis.
        assert (
            self.basis == "modes"
        ), "Convert to spatiotemporal Fourier mode basis before computing K-S equation DAEs."

        # to avoid two IFFT calls, convert before nonlinear product
        orbit_field = self.transform(to="field")

        # Compute the Kuramoto-sivashinsky equation; linear components differ between subclasses.
        mapping_modes = self._eqn_linear_component(array=True) + orbit_field._nonlinear(
            orbit_field, array=True
        )

        return self.__class__(
            **{**vars(self), "state": mapping_modes, "basis": "modes"}
        )

    def jacobian(self, **kwargs):
        """
        Jacobian matrix evaluated at the current state.

        kwargs :
            Unused, included to match signature in Orbit class.

        Returns
        -------
        jac_ : 2-d ndarray
            Jacobian matrix whose columns are derivatives with respect to all unconstrained state variables;
            including periods. Has dimensions dependent on number of spatiotemporal modes and free parameters,
            (self.shapes()[-1].size, self.shapes()[-1].size + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(self.constraints)

        Notes
        -----
        Original implementation was pretty, but very inefficient. This now computes the Jacobian matrix,
        minimizing the amount of allocated memory by overwriting and performing matrix-free implementations of
        Fourier transform matrix operations. Computes the Jacobian
        $J = D_t + D_xx + D_xxxx + F_t D_x F_x Diag(u) F_x^{-1} F_t^{-1}$ in the following steps:

        """
        assert (
            self.basis == "modes"
        ), "Convert to spatiotemporal Fourier mode basis before computing Jacobian"
        field_size, smode_size, mode_size = (np.prod(shp) for shp in self.shapes())
        # Begin with nonlinear term. Apply matrix operators in matrix-free fashion. begin with diag(u)
        J = np.diag(self.transform(to="field", array=True).ravel()).reshape(-1, self.m)
        # By creatively reshaping J, can apply FFTs to 3-d tensor.
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        ).transform(to="spatial_modes", array=True, inplace=True)
        J = J.reshape(field_size, smode_size).T.reshape(-1, self.m)

        # After transforming the columns, transpose and transform again to get F_x Diag(u) F_x^{-1}
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        )

        # Reshape back into a matrix, and then into another 3-d tensor after transposing back, to take derivative.
        J = J.transform(to="spatial_modes", array=True, inplace=True).reshape(
            smode_size, smode_size
        )
        J = J.T.reshape(-1, self.m - 2)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "basis": "spatial_modes",
                "discretization": (J.shape[0], self.m),
            }
        )
        J = J.dx(array=True, computation_basis="spatial_modes", inplace=True)

        # At this point J represents D_x F_x Diag(u) F_x^{-1}; reshape into 3-d tensor again and apply
        # time transforms
        J = J.reshape(-1, smode_size).T.reshape(self.n, self.m - 2, -1)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        # reshape, transpose and transform.
        J = (
            J.reshape((*self.shapes()[2], -1))
            .reshape(-1, smode_size)
            .T.reshape(self.n, self.m - 2, -1)
        )
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        J = J.reshape((*self.shapes()[2], -1)).reshape(J.shape[-1], J.shape[-1]).T

        # Produce the linear term; spatial derivatives are diagonal; time is more complicated due to SO(2) operator.
        e = np.ones(self.shapes()[2])
        dx2 = self.__class__(**{**vars(self), "state": e, "basis": "modes"}).dx(order=2)
        J[np.diag_indices(J.shape[0])] += dx2.state.ravel() + dx2.state.ravel() ** 2
        e = e.ravel()
        # For time, get the correct frequency by taking the derivative of individual elements; swapping of real
        # and imaginary components handled by .dt()
        for i in range(J.shape[0]):
            e *= 0
            e[i] = 1.0
            J[:, i] += (
                self.__class__(
                    **{**vars(self), "state": e.reshape(self.shape), "basis": "modes"}
                )
                .dt()
                .state.ravel()
            )
        J = self._jacobian_parameter_derivatives_concat(J)
        return J

    def matvec(self, other, **kwargs):
        """
        Matrix-vector product of a vector with the Jacobian of the current state.

        Parameters
        ----------
        other : OrbitKS
            OrbitKS instance whose state represents the vector in the matrix-vector multiplication.
        kwargs : dict
            Included to match Orbit signature

        Returns
        -------
        OrbitKS :
            OrbitKS whose state and other parameters result from the matrix-vector product.

        Notes
        -----
        Equivalent to computation of v_t + v_xx + v_xxxx + d_x (u .* v), where v is the state of 'other'.

        """
        assert (self.basis == "modes") and (other.basis == "modes")
        self_field = self.transform(to="field")
        # The correct derivative of the vector in the matrix vector product needs the current state parameters in
        # self but the state stored in other.
        other_mode_component = other.__class__(
            **{**vars(self), "state": other.state, "basis": other.basis}
        )
        other_field = other_mode_component.transform(to="field")

        # Factor of two corrects the 1/2 u^2 from differentiation of nonlinear term.
        matvec_modes = other_mode_component._eqn_linear_component(
            array=True
        ) + 2 * self_field._nonlinear(other_field, array=True)

        if not self.constraints["t"]:
            # Compute the product of the partial derivative with respect to T with the vector's value of T.
            # This is only relevant when other.t an incremental value dt from a numerical method.
            matvec_modes += other.t * (-1.0 / self.t) * self.dt(array=True)

        if not self.constraints["x"]:
            # Compute the product of the partial derivative with respect to L with the vector's value of L.
            # This is only relevant when other.x an incremental value dx from a numerical method.
            dfdl = (
                (-2.0 / self.x) * self.dx(order=2, array=True)
                + (-4.0 / self.x) * self.dx(order=4, array=True)
                + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
            )
            matvec_modes += other.x * dfdl

        return self.__class__(**{**vars(self), "state": matvec_modes, "basis": "modes"})

    def rmatvec(self, other, **kwargs):
        """
        Matrix-vector product with the adjoint of the Jacobian

        Parameters
        ----------
        other : OrbitKS
            OrbitKS whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_rmatvec : OrbitKS
            OrbitKS with values representative of the adjoint-vector product

        Notes
        -----
        The adjoint vector product in this case is defined as J^T * v,  where J is the jacobian matrix. Equivalent to
        evaluation of -v_t + v_xx + v_xxxx  - (u .* v_x). In regards to preconditioning (which is very useful
        for certain numerical methods, right preconditioning and left preconditioning switch meanings when the
        jacobian is transposed. i.e. Right preconditi oning of the Jacobian can include preconditioning of the state
        parameters (which in this case are usually incremental corrections dt, dx, ds);
        this corresponds to LEFT preconditioning of the adjoint.

        The derivatives always occur with respect to the parameters of u. therefore, the evaluation of
        _rmatvec_linear_component and _rnonlinear require the parameters from self.

        """
        assert (self.basis == "modes") and (other.basis == "modes")
        # store the state in the field basis for the pseudospectral products
        self_field = self.transform(to="field")
        other_modes = other.__class__(**{**vars(other), "parameters": self.parameters})
        rmatvec_modes = other_modes._rmatvec_linear_component(
            array=True
        ) + self_field._rnonlinear(other_modes, array=True)

        # parameters are derived by multiplying partial derivatives w.r.t. parameters with the other orbit.
        rmatvec_params = self._rmatvec_parameters(self_field, other_modes)
        return self.__class__(
            **{
                **vars(self),
                "state": rmatvec_modes,
                "basis": "modes",
                "parameters": rmatvec_params,
            }
        )

    def costgrad(self, *args, **kwargs):
        """
        Derivative of $1/2 |F|^2$

        Parameters
        ----------
        eqn : OrbitKS
            Orbit instance whose state equals DAE evaluated with respect to current state, i.e. F(v)
        kwargs :
            Any keyword arguments relevant for rmatvec, eqn, or 'preconditioning'.

        Returns
        -------
        gradient :
            OrbitKS instance whose state contains $(dF/dv)^T * F  = J^T F$

        Notes
        -----
        In this case, "preconditioning" is numerical rescaling of the gradient used as a numerical tool in descent
        methods.

        """

        if args:
            eqn = args[0]
        else:
            eqn = self.eqn()

        grad = self.rmatvec(eqn, **kwargs)
        if kwargs.get("preconditioning", False):
            # This preconditions with respect to the current state. not J^T F
            grad = grad.precondition(pmult=self.preconditioning_parameters())

        return grad

    def plot(
        self, show=True, save=False, padding=False, fundamental_domain=False, **kwargs
    ):
        """
        Plot the velocity field as a 2-d density plot using matplotlib's imshow

        Parameters
        ----------
        show : bool
            Whether or not to display the figure
        save : bool
            Whether to save the figure
        padding : bool
            Whether or not to interpolate more points before plotting. Done numerically instead of via plt.imshow
        fundamental_domain : bool
            Whether to plot only the fundamental domain or not.
        **kwargs :
            padding_shape : (int, int)
                The field discretization to plot, will be used instead of default padding if padding is enabled.
            filename : str
                The (custom) save name of the figure, if save==True. Save name will be populated otherwise.
            extension : str
                The file extension to save as, .png, .pdf, etc. Values supported by matplotlib only.
            figsize : (int, int)
                The matplotlib figure size.

        Notes
        -----
        Many of the defaults are experiential quantities; for comparison, i.e. larger domains require larger figures,
        however, to avoid getting too large or small, a number of defaults are set in place to make sure that the
        labels and scales are sensible.

        The time axis for EquilibriumOrbitKS is labeled by infinity to indicate that they do not change over time.

        """
        plt.rcParams.update(
            {"font.serif": ["Palatino"],}
        )
        if padding:
            padding_shape = kwargs.get("padding_shape", (16 * self.n, 16 * self.m))
            plot_orbit = self.resize(padding_shape)
        else:
            plot_orbit = self.copy()

        if fundamental_domain:
            plot_orbit = plot_orbit.to_fundamental_domain().transform(to="field")
        else:
            # The fundamental domain is never used in computation, so no operations are required if we do not want
            # to plot the fundamental domain explicitly.
            plot_orbit = plot_orbit.transform(to="field")

        # The following creates custom tick labels and accounts for some pathological cases
        # where the period is too small (only a single label) or too large (many labels, overlapping due
        # to font size) Default label tick size is 10 for time and the fundamental frequency, 2 pi sqrt(2) for space.

        # Create time ticks, with the separation
        if plot_orbit.t > 10:
            timetick_step = np.max(
                [
                    np.min(
                        [
                            100,
                            (
                                5
                                * 2
                                ** (np.max([int(np.log2(plot_orbit.t // 2)) - 3, 1]))
                            ),
                        ]
                    ),
                    5,
                ]
            )
            yticks = np.arange(0, plot_orbit.t, timetick_step)
            ylabels = np.array([str(int(y)) for y in yticks])
        elif 0 < plot_orbit.t <= 10:
            scaled_T = np.round(plot_orbit.t, 1)
            yticks = np.array([0, plot_orbit.t])
            ylabels = np.array(["0", str(scaled_T)])
        else:
            plot_orbit.t = np.min([plot_orbit.x, 1])
            yticks = np.array([0, plot_orbit.t])
            ylabels = np.array(["0", "0"])

        if plot_orbit.x > 2 * pi * np.sqrt(2):
            xmult = (plot_orbit.x // 64) + 1
            xscale = xmult * 2 * pi * np.sqrt(2)
            xticks = np.arange(0, plot_orbit.x, xscale)
            xlabels = [str(int((xmult * x) // xscale)) for x in xticks]
        else:
            scaled_L = np.round(plot_orbit.x / (2 * pi * np.sqrt(2)), 1)
            xticks = np.array([0, plot_orbit.x])
            xlabels = np.array(["0", str(scaled_L)])

        default_figsize = (
            min([max([0.25, 0.15 * plot_orbit.x ** 0.7]), 16]),
            min([max([0.25, 0.15 * plot_orbit.t ** 0.7]), 16]),
        )

        # # this allows for local non-zero galilean velocity to be more easily displayed
        maxval = np.round(
            np.abs(
                np.array(
                    [plot_orbit.state.ravel().min(), plot_orbit.state.ravel().max()]
                )
            ).max(),
            1,
        )
        cbarticks = [-maxval, maxval]
        cbarticklabels = [str(i) for i in np.round(cbarticks, 1)]

        figsize = kwargs.get("figsize", default_figsize)
        extentL, extentT = np.min([15, figsize[0]]), np.min([15, figsize[1]])
        scaled_font = np.max([int(np.min([20, np.mean(figsize)])), 10])
        plt.rcParams.update({"font.size": scaled_font})

        fig, ax = plt.subplots(figsize=(extentL, extentT))
        image = ax.imshow(
            plot_orbit.state,
            extent=[0, extentL, 0, extentT],
            cmap="jet",
            interpolation="none",
            aspect="auto",
            vmin=-maxval,
            vmax=maxval,
        )

        xticks = (xticks / plot_orbit.x) * extentL
        yticks = (yticks / plot_orbit.plotting_dimensions()[0][1]) * extentT

        # Include custom ticks and tick labels
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels, ha="left")
        ax.set_yticklabels(ylabels, va="center")
        ax.grid(True, linestyle="dashed", color="k", alpha=0.8)

        fig.subplots_adjust(right=0.95)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.075, pad=0.1)
        cbar = plt.colorbar(image, cax=cax, ticks=cbarticks)
        cbar.ax.set_yticklabels(cbarticklabels, fontdict={"fontsize": scaled_font})

        if save or kwargs.get("filename", None):
            extension = kwargs.get("extension", ".png")
            filename = kwargs.get("filename", None) or self.filename(
                extension=extension
            )
            # Create save name if one doesn't exist.
            if filename.endswith(".h5"):
                filename = "".join([filename.split(".h5")[0], extension])

            if fundamental_domain:
                # Need to rename fundamental domain or else it will overwrite, of course there
                # is no such thing for solutions without any symmetries.
                filename_tmp = filename.split(".")
                filename = "".join([filename_tmp[0], "_fdomain.", filename_tmp[-1]])

            # If filename is provided as an absolute path it overrides the value of 'directory'.
            filename = os.path.abspath(os.path.join(filename))
            if kwargs.get("verbose", False):
                print("Saving figure to {}".format(filename))
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)

        if show:
            plt.show()
        plt.close()
        return None

    def mode_plot(self, show=True, save=False, scale="log", **kwargs):
        """
        Plot the spatiotemporal Fourier spectrum as a 2-d density plot using matplotlib's imshow

        Parameters
        ----------
        show : bool
            Whether or not to display the figure
        save : bool
            Whether to save the figure
        scale : str
            Whether or not to plot using transformation of `np.log10(self.state)`
        **kwargs :
            filename : str
                The (custom) save name of the figure, if save==True. Save name will be populated otherwise.
            verbose : bool
                If True then prints the location of the file saving, if attempted.
            extension : str
                The file extension to save as, .png, .pdf, etc. Values supported by matplotlib only.

        """
        plt.rcParams.update(
            {"font.serif": ["Palatino"],}
        )
        if scale == "log":
            modes = np.abs(self.transform(to="modes").state)
            modes[modes > 0.0] = np.log10(modes[modes > 0.0])
        else:
            modes = self.transform(to="modes").state

        fig, ax = plt.subplots()
        image = ax.imshow(modes, interpolation="none", aspect="auto")

        # Custom colorbar values
        fig.subplots_adjust(right=0.95)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.075, pad=0.1)
        plt.colorbar(image, cax=cax)

        if save or kwargs.get("filename", None):
            extension = kwargs.get("extension", ".png")
            filename = kwargs.get("filename", None) or self.filename(
                extension=extension
            )
            # Create save name if one doesn't exist.
            if filename.endswith(".h5"):
                filename = "".join([filename.split(".h5")[0], extension])

            # If filename is provided as an absolute path it overrides the value of 'directory'.
            filename = os.path.abspath(os.path.join(filename))
            if kwargs.get("verbose", False):
                print("Saving figure to {}".format(filename))
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)

        if show:
            plt.show()
        plt.close()
        return None

    def preconditioning_parameters(self):
        """
        Parameters bundled for convenience

        Returns
        -------
        tuple(tuple, tuple) :
            Time and spatial parameters for usage in preconditioning

        Notes
        -----
        It is often the case that rescaling one OrbitKS with respect to another OrbitKS's parameters is desired.
        This allows calls like self.precondition(other.parameters), for example.

        """
        return (self.t, self.n), (self.x, self.m)

    def precondition(self, **kwargs):
        """
        Rescale a vector with the inverse (absolute value) of linear spatial terms

        Parameters
        ----------
        kwargs :
            pmult : tuple of tuples
                Parameters passed in the form self.preconditioning_parameters
            pexp : tuple
                Exponents for the parameter scaling, default (1, 4)

        Returns
        -------
        target : OrbitKS
            Return the OrbitKS instance, whose state and parameters have been modified by preconditioning.

        Notes
        -----
        Often we want to precondition a state derived from a mapping or rmatvec (gradient descent step),
        with respect to another orbit's (current state's) parameters. By passing parameters we can access the
        cached classmethods.

        I never preconditioned the spatial shift for relative periodic solutions so I don't include it here.

        """
        pmult = kwargs.get("pmult", self.preconditioning_parameters())
        p_multipliers = 1.0 / (
            np.abs(temporal_frequencies(*pmult[0], order=1))
            + np.abs(spatial_frequencies(*pmult[1], order=2)[:, : self.shapes()[2][1]])
            + spatial_frequencies(*pmult[1], order=4)[:, : self.shapes()[2][1]]
        )

        preconditioned_state = np.multiply(self.state, p_multipliers)
        # Precondition the change in T and L
        pexp = kwargs.get("pexp", (1, 4))
        if not self.constraints["t"]:
            # self is the orbit being preconditioned, i.e. the correction orbit; by default this is dt = dt / T
            t = self.t * (pmult[0][0] ** -pexp[0])
        else:
            t = self.t

        if not self.constraints["x"]:
            # self is the orbit being preconditioned, i.e. the correction orbit; by default this is dx = dx / L^4
            x = self.x * (pmult[1][0] ** -pexp[1])
        else:
            x = self.x

        return self.__class__(
            **{
                **vars(self),
                "state": preconditioned_state,
                "basis": "modes",
                "parameters": (t, x, self.s),
            }
        )

    def preconditioner(self, **kwargs):
        """
        Returns a diagonal preconditioner as a scipy LinearOperator instance

        Parameters
        ----------
        kwargs : dict
            Any keyword arguments that are accepted by precondition.

        Returns
        -------
        LinearOperator :
            An object with callable methods that return matrix-vector products, for `v=vector` and `A=LinearOperator`,
            A.matvec(v) and A.rmatvec(v) compute $A * v$ and $A^T * v$, respectively. Here the LinearOperator
            approximates the inverse of the operator that defines the linear component of the KSe
        """
        # To get the diagonal preconditioner, can apply preconditioning to an array of 1's, returning the multipliers.
        # The orbit vector of this instance represents the diagonal of a diagonal preconditioning matrix.
        diag_M = (
            self.__class__(
                **{**vars(self), "state": np.ones(self.shape), "parameters": (1, 1, 1)}
            )
            .precondition(**{"pmult": self.preconditioning_parameters(), **kwargs})
            .orbit_vector()
        )

        def matvec_(v):
            # v is an orbit vector,
            nonlocal diag_M
            if v.ndim != diag_M.ndim:
                diag_M = diag_M.reshape(v.shape)
            return v * diag_M

        # rmatvec = matvec because diagonal
        return LinearOperator(
            shape=(diag_M.size, diag_M.size), matvec=matvec_, rmatvec=matvec_
        )

    def reflection(self, axis=1, signed=True):
        """
        Reflect the velocity field about the spatial midpoint

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is the reflected velocity field -u(L-x,t).

        Notes
        -----
        The correction provided by np.roll is due to the location of the reflection axis in the field array.
        Numerically it is located at the (self.m//2 + 1)-th element while intuitively it should be "between"
        self.m//2 + 1 and self.m//2. Simply a consequence of implementation.

        """
        # Different points in space represented by columns of the state array
        reflected_field = -1.0 * np.roll(
            np.fliplr(self.transform(to="field").state), 1, axis=axis
        )
        return self.__class__(
            **{
                **vars(self),
                "state": reflected_field,
                "basis": "field",
                "parameters": (self.t, self.x, -1 * self.s),
            }
        ).transform(to=self.basis)

    def roll(self, shift, axis=0):
        """
        Apply numpy roll along specified axis.

        Parameters
        ----------
        shift : int or tuple of int
            Number of discrete points to rotate by
        axis : int or tuple of int
            The numpy ndarray axes along which to roll

        Returns
        -------
        OrbitKS :
            Instance with rolled state

        Notes
        -----
        In decision to maintain numpy defaults or change roll to be positive when shift is positive, the latter was
        chosen. If provided tuples of ints, shift and axis need to be the same length as to be coherent.
        This forces OrbitKS to be in the 'field' basis as the rolls can be nonsensical otherwise

        """
        return self.__class__(
            **{
                **vars(self),
                "state": np.roll(self.transform(to="field").state, shift, axis=axis),
            }
        ).transform(to=self.basis)

    def cell_shift(self, n_cell, axis=0):
        """
        Rotate by fraction of the period in either axis; nearest discrete approximate is taken.

        Parameters
        ----------
        n_cell : int or array-like
            Fraction of the domain to rotate by.
        axis : int or array-like
            The numpy ndarray axes along which to roll

        Returns
        -------
        OrbitKS :
            Instance with rolled state

        Notes
        -----
        If provided tuples of ints, shift and axis need to be the same length as to be coherent.

        """
        # To allow slicing, discretization temporarily cast as array.
        return (
            self.transform(to="field")
            .roll(
                np.sign(n_cell)
                * np.array(self.discretization)[np.array(axis)]
                // np.abs(n_cell),
                axis=axis,
            )
            .transform(to=self.basis)
        )

    def rotate(self, distance, axis=0, units="plotting"):
        """
        Rotate the velocity field in either space or time.

        Parameters
        ----------
        distance : float
            The rotation / translation amount, in dimensionless units of time or space.
        axis : int
            The axis of the ndarray (state) that rotations
        units : str
            Determines the spatial units of the provided rotation

        Returns
        -------
        OrbitKS :
            OrbitKS whose field has been rotated.

        Notes
        -----
        Due to periodic boundary conditions, translation is equivalent to rotation on a fundamental level here.
        Hence the use of 'distance' instead of 'angle'. This can be negative. Also due to the periodic boundary
        conditions, a distance equaling the entire domain length is equivalent to no rotation. I.e.
        the rotation is always modulo L or modulo T.

        The orbit only remains a converged solution if rotations coincide with collocation
        points.  i.e. multiples of L / M and T / N. The reason for this is because arbitrary rotations require
        interpolation of the field.

        Rotation breaks discrete symmetry and destroys the solution. Users encouraged to change to OrbitKS first.

        """
        if axis == 0:
            orbit_to_rotate = self.transform(to="modes")

            # angle to rotate by
            thetaj = (
                distance
                * temporal_frequencies(self.t, self.n, 1)[
                    1 : -(orbit_to_rotate.n // 2 - 1), :
                ]
            )
            cosinej = np.cos(thetaj)
            sinej = np.sin(thetaj)

            # Splitting into real and imaginary components of temporal modes
            modes_time_real = orbit_to_rotate.state[
                1 : -(orbit_to_rotate.n // 2 - 1), :
            ]
            modes_time_imaginary = orbit_to_rotate.state[
                -(orbit_to_rotate.n // 2 - 1) :, :
            ]
            # Elementwise product to account for matrix product with "2-D" rotation matrix
            rotated_real = cosinej * modes_time_real + sinej * modes_time_imaginary
            rotated_imag = -sinej * modes_time_real + cosinej * modes_time_imaginary

            time_rotated_modes = np.concatenate(
                (orbit_to_rotate.state[None, 0, :], rotated_real, rotated_imag,),
                axis=0,
            )
            return self.__class__(
                **{**vars(self), "state": time_rotated_modes, "basis": "modes"}
            ).transform(to=self.basis)
        else:
            if units == "wavelength":
                # conversion from plotting units.
                distance = distance * 2 * pi * np.sqrt(2)
            orbit_to_rotate = self.transform(to="spatial_modes")
            # angles to rotate by
            thetak = (
                distance
                * spatial_frequencies(self.x, self.m, 1).ravel()[
                    :, : -(orbit_to_rotate.m // 2 - 1)
                ]
            )
            cosinek = np.cos(thetak)
            sinek = np.sin(thetak)

            # Rotation performed on spatial modes because otherwise rotation is ill-defined for Antisymmetric and
            # Shift-reflection symmetric Orbits.
            spatial_modes_real = orbit_to_rotate.state[
                :, : -(orbit_to_rotate.m // 2 - 1)
            ]
            spatial_modes_imaginary = orbit_to_rotate.state[
                :, -(orbit_to_rotate.m // 2 - 1) :
            ]
            rotated_real = (
                cosinek * spatial_modes_real + sinek * spatial_modes_imaginary
            )
            rotated_imag = (
                -sinek * spatial_modes_real + cosinek * spatial_modes_imaginary
            )
            rotated_spatial_modes = np.concatenate((rotated_real, rotated_imag), axis=1)

            return self.__class__(
                **{
                    **vars(self),
                    "state": rotated_spatial_modes,
                    "basis": "spatial_modes",
                }
            ).transform(to=self.basis)

    def shift_reflection(self):
        """
        Return a OrbitKS with shift-reflected velocity field

        Returns
        -------
        OrbitKS :
            OrbitKS with shift-reflected velocity field

        Notes
        -----
        Shift reflection in this case is a composition of spatial reflection and temporal translation by
        half of the period. Because these are in different dimensions these operations commute.

        """
        shift_reflected_field = np.roll(
            -1.0 * np.roll(np.fliplr(self.transform(to="field").state), 1, axis=1),
            self.n // 2,
            axis=0,
        )
        return self.__class__(
            **{**vars(self), "state": shift_reflected_field, "basis": "field"}
        ).transform(to=self.basis)

    def group_orbit(self, **kwargs):
        """
        Group orbit generator.

        Yields
        ------
        An instance which is an element of the (discrete or continuous) group orbit (equivariance) of the
        current instance.

        """
        if kwargs.get("discrete", False):
            # The discrete symmetry operations which preserve reflection symmetry axis. Spatial only.
            for g_space in (
                self,
                self.reflection(),
                self.cell_shift(2, axis=1),
                self.cell_shift(2, axis=1).reflection(),
            ):
                # Half cell shifts in time to account for shift-reflection invariant orbits
                for g in [g_space, g_space.cell_shift(2, axis=0)]:
                    if kwargs.get("fundamental_domain", False):
                        yield g.to_fundamental_domain()
                    else:
                        yield g
        elif kwargs.get("continuous", False):
            rolls = kwargs.get("rolls", (1, 1))
            for N in range(0, self.n, rolls[0]):
                for M in range(0, self.m, rolls[1]):
                    if kwargs.get("fundamental_domain", False):
                        yield self.roll(N, axis=0).roll(
                            M, axis=1
                        ).to_fundamental_domain()
                    else:
                        yield self.roll(N, axis=0).roll(M, axis=1)
        else:
            # Don't need cell shifts, these are within the rotations. Arbitrary rotations require interpolation;
            # only roll preserves orbit's status as a solution.
            rolls = kwargs.get("rolls", (1, 1))
            for g in [self, self.reflection()]:
                for N in range(0, g.n, rolls[0]):
                    for M in range(0, g.m, rolls[1]):
                        if kwargs.get("fundamental_domain", False):
                            yield g.roll(N, axis=0).roll(
                                M, axis=1
                            ).to_fundamental_domain()
                        else:
                            yield g.roll(N, axis=0).roll(M, axis=1)

    def shapes(self):
        """
        State array shapes in different bases; determined by symmetry selection rules.

        """
        return (
            (self.n, self.m),
            (self.n, self.m - 2),
            (max([self.n - 1, 1]), self.m - 2),
        )

    def dimensions(self):
        """
        Tile dimensions.

        """
        return self.t, self.x

    def plotting_dimensions(self):
        """
        Dimensions according to plot labels; used in clipping.

        """
        return (0.0, self.t), (0.0, self.x / (2 * pi * np.sqrt(2)))

    def _pad(self, size, axis=0):
        """
        Increase the size of the discretization via zero-padding

        Parameters
        ----------
        size : int
            The new size of the discretization, must be an even integer
            larger than the current size of the discretization (handled by resize method).

        axis : int
            The dimension of the state that will be padded.

        Returns
        -------
        OrbitKS :
            OrbitKS instance with larger discretization.

        Notes
        -----
        Need to account for the normalization factors by multiplying by old, dividing by new.

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                # Due to formatting, can prepend and append zeros to second half as opposed to appending
                # to first and second halves.
                padding = (size - modes.n) // 2
                padding_tuple = ((padding, padding), (0, 0))
                padded_modes = np.concatenate(
                    (
                        modes.state[: -(modes.n // 2 - 1), :],
                        np.pad(modes.state[-(modes.n // 2 - 1) :, :], padding_tuple),
                    ),
                    axis=0,
                )
                padded_modes *= np.sqrt(size / modes.n)
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)

            else:
                padding = (size - modes.m) // 2
                padding_tuple = ((0, 0), (padding, padding))
                padded_modes = np.concatenate(
                    (
                        modes.state[:, : -(modes.m // 2 - 1)],
                        np.pad(modes.state[:, -(modes.m // 2 - 1) :], padding_tuple),
                    ),
                    axis=1,
                )
                padded_modes *= np.sqrt(size / modes.m)
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Decrease the size of the discretization via truncation

        Parameters
        ----------
        size : int
            The new size of the discretization, must be an even integer
            smaller than the current size of the discretization (handled by resize method).

        axis : int
            The dimension of the state that will be truncated.

        Returns
        -------
        OrbitKS :
            OrbitKS instance with smaller discretization.

        Notes
        -----
        Need to account for the normalization factors by multiplying by old, dividing by new.

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                truncate_number = int(size // 2) - 1
                # Split into real and imaginary components, truncate separately.
                first_half = modes.state[: truncate_number + 1, :]
                second_half = modes.state[
                    -(modes.n // 2 - 1) : -(modes.n // 2 - 1) + truncate_number, :
                ]
                truncated_modes = np.sqrt(size / modes.n) * np.concatenate(
                    (first_half, second_half), axis=0
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                truncate_number = int(size // 2) - 1
                # Split into real and imaginary components, truncate separately.
                first_half = self.state[:, :truncate_number]
                second_half = self.state[
                    :,
                    -(int(self.m // 2) - 1) : -(int(self.m // 2) - 1) + truncate_number,
                ]
                truncated_modes = np.sqrt(size / modes.m) * np.concatenate(
                    (first_half, second_half), axis=1
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def preprocess(self):
        """
        Check whether the orbit converged to an equilibrium or close-to-zero solution

        """
        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        field_orbit = self.transform(to="field")
        # Equilibrium is defined by having no temporal variation, i.e. time derivative is a uniformly zero.
        if self.t == 0.0:
            # If there is sufficient evidence that solution is an equilibrium, change its class
            # store T just in case we want to refer to what the period was before conversion to EquilibriumOrbitKS
            return EquilibriumOrbitKS(
                state=field_orbit.state, basis="field", parameters=self.parameters
            ).transform(to=self.basis)
        # See if the L_2 norm is beneath a threshold value, if so, replace with zeros.
        elif field_orbit.norm() < field_orbit.size * 10 ** -9:
            return EquilibriumOrbitKS(
                state=np.zeros(self.discretization),
                basis="field",
                parameters=self.parameters,
            ).transform(to=self.basis)

        elif (
            field_orbit.dt().transform(to="field").norm() < field_orbit.size * 10 ** -9
        ):
            # If there is sufficient evidence that solution is an equilibrium, change its class
            # code = 3
            # store T just in case we want to refer to what the period was before conversion to EquilibriumOrbitKS
            return EquilibriumOrbitKS(
                state=field_orbit.state, basis="field", parameters=self.parameters
            ).transform(to=self.basis)
        else:
            return self

    def to_h5(
        self,
        filename=None,
        dataname=None,
        h5mode="a",
        verbose=False,
        include_cost=True,
        **kwargs,
    ):
        """
        Export current state information to HDF5 file. See core.py for more details

        Parameters
        ----------
        filename : str or None
            If None then filename method will be called.
        dataname : str or None
            h5py group name for orbit
        h5mode : str
            The writing mode, see core.py for details
        verbose : bool
            Whether or not to print save destination.
        include_cost :
            Whether to save cost to h5 file.

        Notes
        -----
        Mainly an overload simply to get a different default behavior for include_cost.

        """
        super().to_h5(
            filename=filename,
            dataname=dataname,
            h5mode=h5mode,
            verbose=verbose,
            include_cost=include_cost,
            **kwargs,
        )

    @classmethod
    def dimension_based_discretization(cls, dimensions, **kwargs):
        """
        Return discretization size according to orbithunter conventions for the KSe.

        Parameters
        ----------
        dimensions : tuple
            tuple containing (T, L) as first two entries of tuple (i.e. self.parameters or self.dimensions)
        kwargs :
            resolution : str
                Takes values 'coarse', 'normal', 'fine', 'power'.
                These options return the according discretization sizes as described below.
            discretization : tuple of int or None
                If one wants to force or specify a particular dimension's discretization, it can be provided
                as, for example, discretization=(N, None).

        Returns
        -------
        discretization : tuple of ints
            The new spatiotemporal field discretization; number of time points
            (rows) and number of space points (columns)

        Notes
        -----
        This function should only ever be called by resize, the returned values can always be accessed by
        the appropriate attributes of the rediscretized orbit.

        """
        resolution = kwargs.get("resolution", "default")
        t, x = dimensions
        N, M = kwargs.get("discretization", (None, None))
        if N is None:
            if t == 0:
                N = cls._default_shape()[0]
            elif resolution == "coarse":
                N = np.max([2 ** (int(np.log2(t) - 2)), 16])
            elif resolution == "fine":
                N = np.max([2 ** (int(np.log2(t) + 1)), 32])
            elif resolution == "power":
                N = np.max([2 * (int(4 * t ** (1.0 / 2.0)) // 2), 32])
            else:
                N = np.max([2 ** (int(np.log2(t) - 1)), 32])
            N = max(N, cls.minimal_shape()[0])

        if M is None:
            if x == 0:
                M = cls._default_shape()[1]
            elif resolution == "coarse":
                M = np.max([2 ** (int(np.log2(x) - 1)), 16])
            elif resolution == "fine":
                M = np.max([2 ** (int(np.log2(x) + 2)), 32])
            elif resolution == "power":
                M = np.max([2 * (int(4 * x ** (1.0 / 2.0)) // 2), 32])
            else:
                M = np.max([2 ** (int(np.log2(x) + 0.5)), 32])
            M = max(M, cls.minimal_shape()[1])
        return N, M

    @classmethod
    def positive_indexing(cls):
        """
        Indicates whether numpy indexing corresponds to increasing or decreasing values configuration space variable

        Returns
        -------
        tuple :
            A tuple of bool, one for each continuous dimension which indicates whether positively increasing
            indices indicate the 'positive direction' in configuration space.

        Notes
        -----
        This is mainly for compatibility for the "upwards" time convention in the field of Physics, used for the
        Kuramoto-Sivashinsky equation. That is, the last element along the time axis, i.e. index = -1
        corresponds to t=0 if False, t=T if True, where T is the temporal period of the Orbit.

        """

        return False, True

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Default parameter ranges.

        Returns
        -------
        dict :
            Keys are parameter labels, values are intervals to sample from in parameter generation.

        """
        return {"t": (20, 200), "x": (20, 100)}

    def _default_constraints(self):
        """
        Defaults for whether or not parameters are constrained. parameter labels are forced to be constant.

        """
        return {"t": False, "x": False}

    def _populate_state(self, **kwargs):
        """
        Initialize a set of random spatiotemporal Fourier modes

        Parameters
        ----------
        **kwargs
            tscale : int
                "Mean" for temporal spectrum modulation
            xscale : int
                "Mean" for spatial spectrum modulation
            xvar : float
                "Variance" for temporal spectrum modulation
            tvar : float
                "Variance"for spatial spectrum modulation
            seed : int
                Value to seed random number generator with.

        Returns
        -------
        self :
            OrbitKS whose state has been modified to be a set of random Fourier modes.

        Notes
        -----
        These are the initial condition generators that I find the most useful. If a different method is
        desired, simply pass the array as 'state' variable to __init__.

        By initializing the shape parameters and orbit parameters, the other properties get initialized, so
        they can be referenced in what follows (). I am unsure whether or not
        this is bad practice but they could be replaced by the corresponding tuples. The reason why this is avoided
        is so this function generalizes to subclasses.

        """
        spatial_modulation = kwargs.get("spatial_modulation", "gaussian")
        temporal_modulation = kwargs.get("temporal_modulation", "time_truncation")

        tscale = kwargs.get("tscale", int(np.round(self.t / 20.0)))
        xscale = kwargs.get("xscale", int(np.round(self.x / (2 * pi * np.sqrt(2)))))
        xvar = kwargs.get("xvar", max([np.sqrt(xscale), 1]))
        tvar = kwargs.get("tvar", max([np.sqrt(tscale), 1]))
        np.random.seed(kwargs.get("seed", None))

        # also accepts discretization as kwarg
        n, m = self.dimension_based_discretization(self.dimensions(), **kwargs)
        if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
            warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
            warnings.warn(warn_str, RuntimeWarning)
        self.discretization = n, m
        # I think this is the easiest way to get symmetry-dependent Fourier mode arrays' shapes.
        # power = 2 b.c. odd powers not defined for spacetime modes for discrete symmetries.
        space_ = np.abs(
            spatial_frequencies(2 * pi, self.m, 1)[:, : self.shapes()[2][1]]
        ).astype(int)
        time_ = np.abs(temporal_frequencies(2 * pi, self.n, 1)).astype(int)
        random_modes = np.random.randn(*self.shapes()[2])

        # Anecdotal evidence shows that not enough space-time coupling produces traveling waves very often.
        # For example, scaling the spatial and temporal modes with the same gaussian profile. Therefore,
        # The following is added as a means of coupling space and time. It also mimics witnessed spectra.
        xshift = kwargs.get("xshift", "sqrt")
        if xshift == "sqrt":
            xscale = xscale + np.sqrt(time_)
        elif xshift == "log":
            xscale = xscale + np.log(1 + time_)
        elif type(xshift) in [int, float, np.float64, np.int32, np.ndarray]:
            xscale = xscale + xshift

        # Spatial then temporal modulation, done separately now.
        original_mode_norm = np.linalg.norm(random_modes)
        if spatial_modulation == "gaussian":
            modulator = np.exp(-((space_ - xscale) ** 2) / (2 * xvar))
        elif spatial_modulation == "laplace":
            modulator = np.exp(-1.0 * np.abs(space_ - xscale) / np.sqrt(xvar))
        elif spatial_modulation == "laplace_sqrt":
            modulator = np.exp(-1.0 * np.sqrt(np.abs(space_ - xscale)) / np.sqrt(xvar))
        elif spatial_modulation == "plateau_linear":
            modulator = np.divide(
                1, ((2 * pi * space_ / self.x) ** 2 - (2 * pi * space_ / self.x) ** 4)
            ) * np.ones(time_.shape)
            modulator[space_ <= xscale] = 1
        elif spatial_modulation == "exponential_linear":
            modulator = np.exp(
                (2 * pi * space_ / self.x) ** 2 - (2 * pi * space_ / self.x) ** 4
            ) * np.ones(time_.shape)
            modulator[space_ <= xscale] = 1
        elif spatial_modulation == "flat_top":
            modulator = np.exp(-((np.abs(space_ - xscale) / xvar) ** 5))
        else:
            modulator = np.ones(random_modes.shape)

        modes = np.multiply(modulator, random_modes)
        if temporal_modulation == "gaussian":
            modulator = np.exp(-((time_ - tscale) ** 2) / (2 * tvar))
        elif temporal_modulation == "laplace":
            modulator = np.exp(-1.0 * np.abs(time_ - tscale) / np.sqrt(tvar))
        elif temporal_modulation == "truncate":
            modulator = time_.copy()
            modulator[time_ > tscale] = 0
            modulator[time_ <= tscale] = 1
        else:
            modulator = np.ones(random_modes.shape)
        modes = np.multiply(modulator, modes)
        # Rescale
        self.state = (original_mode_norm / np.linalg.norm(modes)) * modes
        self.basis = "modes"

    def _parse_state(self, state, basis, **kwargs):
        """
        Instantiate state and infer shape of collocation grid from numpy array and basis

        Parameters
        ----------
        state : np.ndarray
            The current spatiotemporal Orbit state.
        basis : str
            String taking value 'field', 'spatial_modes', 'modes'. Determines how discretization is inferred from
            state array. If no state provided then this value is overwritten and None is set as the value.
        **kwargs :
            Unused, included so signature matches Orbit class.

        """
        if isinstance(state, np.ndarray):
            if len(state.shape) != 2:
                raise ValueError('"state" array must be two-dimensional')
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(0, 0)

        if self.size > 0:
            # This is essentially the inverse of .shapes() method
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            elif basis == "modes":
                # N-1, M-2
                n, m = self.shape[0] + 1, self.shape[1] + 2
            elif basis == "field":
                # N, M
                n, m = self.shape
            elif basis == "spatial_modes":
                # N, M - 2
                n, m = self.shape[0], self.shape[1] + 2
            else:
                raise ValueError(
                    'basis not recognized; must equal "field" or "spatial_modes", or "modes"'
                )
            if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
                warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
                warnings.warn(warn_str, RuntimeWarning)
            self.basis = basis
            self.discretization = n, m
        else:
            self.basis = None
            self.discretization = None

    def _eqn_linear_component(self, array=False):
        """
        Linear component of the KSe.

        Parameters
        ----------
        array : bool
            Whether to return np.ndarray or not.

        Returns
        -------
        OrbitKS or np.ndarray :
            Evaluation of the governing equations.

        Notes
        -----
        Equal to u_t + u_xx + u_xxxx

        """
        return (
            self.dt(array=array)
            + self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
        )

    def _nonlinear(self, other, array=False):
        """
        Computation of the nonlinear term of the Kuramoto-Sivashinsky equation

        Parameters
        ----------
        other : OrbitKS
            The second component of the nonlinear product.
        array : bool
            Whether to return a numpy array or OrbitKS instance

        Returns
        -------
        np.ndarray or OrbitKS
            Array or OrbitKS instance containing the nonlinear term of the KSE = 1/2 (u**2)_x

        Notes
        -----
        The nonlinear product is the name given to the elementwise product in the field basis equivalent to the
        convolution of spatiotemporal Fourier modes, the defining quality of a pseudospectral implementation.
        The matrix vector product takes the form d_x (u * v), but the "normal" usage is d_x (u * u); in the latter
        case 'other' should equal 'self', in the field basis.

        """
        # Elementwise product, both self and other should be in physical field basis.
        assert (self.basis == "field") and (other.basis == "field")
        return 0.5 * (self * other).transform(to="modes").dx(array=array)

    def _rnonlinear(self, other, array=False, **kwargs):
        """
        Computation of the nonlinear term of the adjoint Kuramoto-Sivashinsky equation

        Parameters
        ----------
        other : OrbitKS
            The second component of the nonlinear product
        array : bool
            Whether to return np.ndarray or OrbitKS.

        Returns
        -------
        np.ndarray or OrbitKS :
            Array or OrbitKS instance with -u*v_x as its state.

        Notes
        -----
        The matrix-vector product comprised of adjoint Jacobian evaluated at 'self'
        multiplied with the spatiotemporal modes from another orbit instance (typically the DAE modes)
        elementwise/vectorized operation takes the form -u * v_x. self==u, other==v.

        """
        assert self.basis == "field"
        if array:
            return -1.0 * (self * other.dx(return_basis="field")).transform(
                to="modes", array=True
            )
        else:
            # inplace is fine here because dx() is making a new array
            return -1.0 * (self * other.dx(return_basis="field")).transform(to="modes")

    def _rmatvec_parameters(self, self_field, other):
        """
        Parameter values from product with partial derivatives

        Parameters
        ----------
        self_field : OrbitKS
            The orbit in the field basis; this cuts down on redundant transforms.
        other : OrbitKS
            The adjoint/co-state variable Orbit instance.

        Returns
        -------
        parameters : tuple
            Set of parameters resulting from the last rows of the product with adjoint Jacobian.

        """
        other_modes_in_vector_form = other.state.ravel()
        if not self.constraints["t"]:
            # partial derivative with respect to period times the adjoint/co-state variable state.
            rmatvec_t = (-1.0 / self.t) * self.dt(array=True).ravel().dot(
                other_modes_in_vector_form
            )
        else:
            rmatvec_t = 0

        if not self.constraints["x"]:
            # change in L, dx, equal to DF/DL * v
            rmatvec_x = (
                (
                    (-2.0 / self.x) * self.dx(order=2, array=True)
                    + (-4.0 / self.x) * self.dx(order=4, array=True)
                    + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
                )
                .ravel()
                .dot(other_modes_in_vector_form)
            )
        else:
            rmatvec_x = 0

        return rmatvec_t, rmatvec_x, 0.0

    def _rmatvec_linear_component(self, array=False):
        """
        Linear component of the adjoint Jacobian-vector product

        Parameters
        ----------
        array : bool
            Whether or not to return ndarray

        Returns
        -------
        OrbitKS :
            Linear component of the adjoint KSE

        """
        return (
            -1.0 * self.dt(array=array)
            + self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
        )

    def _jac_lin(self):
        """
        The linear component of the Jacobian matrix.

        Returns
        -------
        OrbitKS or ndarray :
            Linear component of the Jacobian matrix.

        """
        return self._dt_matrix() + self._dx_matrix(order=2) + self._dx_matrix(order=4)

    def _jac_nonlin(self):
        """
        The nonlinear component of the Jacobian matrix.

        Returns
        -------
        _jac_nonlin : matrix
            Matrix which represents the nonlinear component of the Jacobian. The derivative of
            the nonlinear term, which is
            (D/DU) 1/2 d_x (u .* u) = (D/DU) 1/2 d_x F (diag(F^-1 u)^2)  = d_x F( diag(F^-1 u) F^-1).
            See
            Chu, K.T. A direct matrix method for computing analytical Jacobians of discretized nonlinear
            integro-differential equations. J. Comp. Phys. 2009 for details.


        """
        _jac_nonlin_left = self._dx_matrix().dot(self._time_transform_matrix())
        _jac_nonlin_middle = self._space_transform_matrix().dot(
            np.diag(self.transform(to="field").state.ravel())
        )
        _jac_nonlin_right = self._inv_spacetime_transform_matrix()
        _jac_nonlin = _jac_nonlin_left.dot(_jac_nonlin_middle).dot(_jac_nonlin_right)
        return _jac_nonlin

    def _jacobian_parameter_derivatives_concat(self, jac_):
        """
        Compute and concatenate parameter partial derivative vectors to Jacobian matrix

        Parameters
        ----------
        jac_ : np.ndarray,
            Jacobian matrix containined partial derivatives with respect to spatiotemporal modes.

        Returns
        -------
        jac_ :
            Jacobian augmented with parameter partial derivatives as columns. Required to solve for changes to period,
            space period in optimization process.

        """
        # If period is not fixed, need to include dF/dt in jacobian matrix
        if not self.constraints["t"]:
            time_period_derivative = (-1.0 / self.t) * self.dt(array=True).reshape(
                -1, 1
            )
            jac_ = np.concatenate((jac_, time_period_derivative), axis=1)

        # If spatial period is not fixed, need to include dF/dx in jacobian matrix
        if not self.constraints["x"]:
            self_field = self.transform(to="field")
            spatial_period_derivative = (
                (-2.0 / self.x) * self.dx(order=2, array=True)
                + (-4.0 / self.x) * self.dx(order=4, array=True)
                + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
            )
            jac_ = np.concatenate(
                (jac_, spatial_period_derivative.reshape(-1, 1)), axis=1
            )

        return jac_

    def _dx_matrix(self, order=1, computation_basis="modes"):
        """
        The spatial derivative matrix operator for the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.
        computation_basis : str
            The basis in which to produce the operator. 'modes' or 'spatial_modes'.

        Returns
        ----------
        spacetime_dxn : ndarray
            The operator whose matrix-vector product with spatiotemporal
            Fourier modes is equal to the time derivative. *Only used in
            the construction of the Jacobian operator.

        """
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        if computation_basis == "spatial_modes":
            # else use time discretization size.
            spacetime_dxn = np.kron(np.eye(self.n), dxn_block(self.x, self.m, order))
        else:
            # When the dimensions of dxn_block are the same as the mode tensor, the slicing does nothing.
            spacetime_dxn = np.kron(
                np.eye(self.shapes()[2][0]),
                dxn_block(self.x, self.m, order)[
                    : self.shapes()[2][1], : self.shapes()[2][1]
                ],
            )

        return spacetime_dxn

    def _dt_matrix(self, order=1):
        """
        The time derivative matrix operator for the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.

        Returns
        ----------
        spacetime_dtn : ndarray
            The operator whose matrix-vector product with spatiotemporal
            Fourier modes is equal to the time derivative. *Only used in
            the construction of the Jacobian operator.

        """
        # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
        # Zeroth frequency was not included in frequency vector.
        dtn_matrix = block_diag([[0]], dtn_block(self.t, self.n, order))
        # Take kronecker product to account for the number of spatial modes.
        spacetime_dtn = np.kron(dtn_matrix, np.eye(self.shapes()[2][1]))
        return spacetime_dtn

    def _inv_spacetime_transform_matrix(self):
        """
        Inverse Space-time Fourier transform operator

        Returns
        -------
        ndarray :
            Matrix operator whose action maps a set of spatiotemporal modes into a physical field u(x,t)

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """
        return np.dot(
            self._inv_space_transform_matrix(), self._inv_time_transform_matrix()
        )

    def _spacetime_transform_matrix(self):
        """
        Space-time Fourier transform operator

        Returns
        -------
        np.ndarray :
            Matrix operator whose action maps a physical field u(x,t) into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """
        return np.dot(self._time_transform_matrix(), self._space_transform_matrix())

    def _time_transform(self, array=False, inplace=False):
        """
        Temporal Fourier transform

        Parameters
        ----------
        array : bool
            Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
            Orbit instances.

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """

        if inplace:
            # Take rfft, accounting for unitary normalization.
            self.state = rfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            self.state = np.concatenate(
                (self.state.real[:-1, :], self.state.imag[1:-1, :]), axis=0
            )
            self.state[1:, :] = np.sqrt(2) * self.state[1:, :]
            self.basis = "modes"
            if array:
                return self.state
            else:
                return self
        else:
            # Take rfft, accounting for unitary normalization.
            modes = rfft(self.state, workers=self.workers, norm="ortho", axis=0)
            modes = np.concatenate((modes.real[:-1, :], modes.imag[1:-1, :]), axis=0)
            modes[1:, :] = np.sqrt(2) * modes[1:, :]
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "modes"}
                )

    def _inv_time_transform(self, array=False, inplace=False):
        """
        Temporal Fourier transform

        Parameters
        ----------
        array : bool
            Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
            Orbit instances.

        Returns
        -------
        OrbitKS or ndarray
            OrbitKS whose state is in the temporal Fourier mode basis or the corresponding array.

        """
        if inplace:
            self.state = self.state.astype(complex)
            self.state[1 : -max([int(self.n // 2) - 1, 1]), :] += (
                1j * self.state[-max([int(self.n // 2) - 1, 1]) :, :]
            )
            self.state = self.state[: -max([int(self.n // 2) - 1, 1]) + 1, :]
            self.state[-1, :] = 0
            self.state[1:, :] *= 1.0 / np.sqrt(2)
            self.state = irfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            self.basis = "spatial_modes"
            if array:
                return self.state
            else:
                return self
        else:
            modes = self.state
            padding = np.zeros([1, modes.shape[1]])
            time_real = np.concatenate(
                (modes[: -max([int(self.n // 2) - 1, 1]), :], padding), axis=0
            )
            time_imaginary = np.concatenate(
                (padding, modes[-max([int(self.n // 2) - 1, 1]) :, :], padding), axis=0
            )
            complex_modes = time_real + 1j * time_imaginary
            complex_modes[1:, :] *= 1.0 / np.sqrt(2)
            space_modes = irfft(
                complex_modes, workers=self.workers, norm="ortho", axis=0
            )
            if array:
                return space_modes
            else:
                return self.__class__(
                    **{**vars(self), "state": space_modes, "basis": "spatial_modes"}
                )

    def _space_transform(self, array=False, inplace=False):
        """
        Spatial Fourier transform

        Parameters
        ----------
        array : bool
            Whether to return numpy array or not.

        Returns
        -------
        OrbitKS or np.ndarray:
            OrbitKS instance in the physical field basis or corresponding array.

        """
        if inplace:
            # Take rfft, accounting for unitary normalization.
            self.state = (
                np.sqrt(2)
                * rfft(
                    self.state,
                    workers=self.workers,
                    norm="ortho",
                    axis=1,
                    overwrite_x=True,
                )[:, 1:-1]
            )
            self.state = np.concatenate((self.state.real, self.state.imag), axis=1)
            self.basis = "spatial_modes"
            if array:
                return self.state
            else:
                return self
        else:
            # Take rfft, accounting for unitary normalization.
            spatial_modes = (
                np.sqrt(2)
                * rfft(self.state, workers=self.workers, norm="ortho", axis=1)[:, 1:-1]
            )
            spatial_modes = np.concatenate(
                (spatial_modes.real, spatial_modes.imag), axis=1
            )
            if array:
                return spatial_modes
            else:
                return self.__class__(
                    **{**vars(self), "state": spatial_modes, "basis": "spatial_modes"}
                )

    def _inv_space_transform(self, array=False, inplace=False):
        """
        Inverse spatial Fourier transform

        Parameters
        ----------
        array : bool
            Whether to return numpy array or not.

        Returns
        -------
        OrbitKS or np.ndarray:
            OrbitKS instance in the physical field basis or corresponding array.

        """
        if inplace:
            # Do the transform inplace; do not need to change other attributes other than basis; discretization
            # attribute is the shape in 'field' basis NOT current basis so it remains constant.
            self.state = self.state.astype(complex)
            self.state = (
                self.state[:, : -(int(self.m // 2) - 1)]
                + 1j * self.state[:, -(int(self.m // 2) - 1) :]
            )
            # Re-add the zeroth and Nyquist spatial frequency modes (zeros) and then transform back
            z = np.zeros([self.n, 1])
            self.state = (1.0 / np.sqrt(2)) * irfft(
                np.concatenate((z, self.state, z), axis=1),
                workers=self.workers,
                norm="ortho",
                axis=1,
                overwrite_x=True,
            )
            self.basis = "field"
            if array:
                return self.state
            else:
                return self
        else:
            # Make the modes complex valued again.
            complex_modes = (
                self.state[:, : -(int(self.m // 2) - 1)]
                + 1j * self.state[:, -(int(self.m // 2) - 1) :]
            )
            # Re-add the zeroth and Nyquist spatial frequency modes (zeros) and then transform back
            z = np.zeros([self.n, 1])
            field = (1.0 / np.sqrt(2)) * irfft(
                np.concatenate((z, complex_modes, z), axis=1),
                workers=self.workers,
                norm="ortho",
                axis=1,
            )
            if array:
                return field
            else:
                return self.__class__(
                    **{**vars(self), "state": field, "basis": "field"}
                )

    def _space_transform_matrix(self):
        """
        Spatial Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a physical field u(x,t) into a set of spatial Fourier modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """

        dft_mat = rfft(
            np.eye(self.m), workers=self.workers, norm="ortho", axis=0, overwrite_x=True
        )[1:-1, :]
        space_dft_mat = np.sqrt(2) * np.concatenate(
            (dft_mat.real, dft_mat.imag), axis=0
        )
        return np.kron(np.eye(self.n), space_dft_mat)

    def _time_transform_matrix(self):
        """
        Inverse Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatiotemporal modes into a set of spatial modes

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """
        dft_mat = rfft(
            np.eye(self.n), workers=self.workers, norm="ortho", axis=0, overwrite_x=True
        )
        time_dft_mat = np.concatenate(
            (dft_mat[:-1, :].real, dft_mat[1:-1, :].imag), axis=0
        )
        time_dft_mat[1:, :] = np.sqrt(2) * time_dft_mat[1:, :]
        return np.kron(time_dft_mat, np.eye(self.m - 2))

    def _inv_time_transform_matrix(self):
        """
        Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into a set of spatiotemporal modes.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """
        return self._time_transform_matrix().transpose()

    def _inv_space_transform_matrix(self):
        """
        Time Fourier transform operator

        Returns
        -------
        matrix :
            Matrix operator whose action maps a set of spatial modes into field basis.

        Notes
        -----
        Only used for the construction of the Jacobian matrix. Do not use this for the Fourier transform.

        """
        return self._space_transform_matrix().transpose()

    def _inv_spacetime_transform(self, array=False, inplace=False):
        """
        Inverse space-time Fourier transform

        Parameters
        ----------
        array : bool
            Whether to return numpy array or not.

        Returns
        -------
        OrbitKS or np.ndarray:
            OrbitKS instance in the physical field basis or corresponding array.

        """
        if array:
            return (
                self._inv_time_transform(inplace=inplace)
                ._inv_space_transform(inplace=inplace)
                .state
            )
        else:
            return self._inv_time_transform(inplace=inplace)._inv_space_transform(
                inplace=inplace
            )

    def _spacetime_transform(self, array=False, inplace=False):
        """
        Space-time Fourier transform

        Parameters
        ----------
        array : bool
            Whether to return numpy array or not.

        Returns
        -------
        OrbitKS or np.ndarray:
            OrbitKS instance in the physical field basis or corresponding array.

        """
        if array:
            return (
                self._space_transform(inplace=inplace)
                ._time_transform(inplace=inplace)
                .state
            )
        else:
            # Return transform of field
            return self._space_transform(inplace=inplace)._time_transform(
                inplace=inplace
            )


class RelativeOrbitKS(OrbitKS):
    def __init__(
        self,
        state=None,
        basis=None,
        parameters=None,
        discretization=None,
        constraints=None,
        frame="comoving",
        **kwargs,
    ):
        """
        Same as OrbitKS except for setting the 'frame' attribute.

        """
        self.frame = frame
        super().__init__(
            state=state,
            basis=basis,
            parameters=parameters,
            discretization=discretization,
            constraints=constraints,
            **kwargs,
        )

    def periodic_dimensions(self):
        """
        Bools indicating whether or not dimension is periodic for persistent homology calculations.

        Returns
        -------

        Notes
        -----
        Static for base class, however for relative periodic solutions this can be dependent on the frame/slice the
        state is in.

        """
        if self.frame == "comoving":
            return True, True
        else:
            return False, True

    def dt(self, order=1, array=False, **kwargs):
        """
        A time derivative of the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.
        array : bool
            Whether to return np.ndarray or Orbit instance

        Returns
        ----------
        orbit_dtn : OrbitKS or subclass instance
            The class instance whose state is the time derivative in the spatiotemporal mode basis.

        """
        if self.frame == "comoving":
            return super().dt(order, array=array, **kwargs)
        else:
            raise ValueError(
                "Attempting to compute time derivative of "
                + str(self)
                + " in physical reference frame."
            )

    def matvec(self, other, **kwargs):
        """
        Extension of parent class method

        Parameters
        ----------
        other : RelativeOrbitKS
            RelativeOrbitKS instance whose state represents the vector in the matrix-vector multiplication.

        Returns
        -------
        RelativeOrbitKS
            RelativeOrbitKS whose state and other parameters result from the matrix-vector product.

        Notes
        -----
        Equivalent to computation of (v_t + v_xx + v_xxxx + phi * v_x) + d_x (u .* v)
        The additional term phi * v_x is in the linear portion of the equation, meaning that
        we can calculate it and add it to the rest of the mapping a posteriori.

        """

        assert (self.basis == "modes") and (other.basis == "modes")
        matvec_orbit = super().matvec(other)
        # this is needed unless all parameters are fixed, but that isn't ever a realistic choice.
        self_dx = self.dx(array=True)
        if not self.constraints["t"]:
            matvec_orbit += other.t * (-1.0 / self.t) * (-self.s / self.t) * self_dx

        if not self.constraints["x"]:
            # Derivative of mapping with respect to T is the same as -1/T * u_t
            matvec_orbit += other.x * (-1.0 / self.x) * (-self.s / self.t) * self_dx

        if not self.constraints["s"]:
            # technically could do self_comoving / self.s but this can be numerically unstable when self.s is small
            matvec_orbit += other.s * (-1.0 / self.t) * self_dx

        return matvec_orbit

    def jacobian(self, **kwargs):
        """
        Jacobian matrix evaluated at the current state.

        kwargs :
            Unused, included to match signature in Orbit class.

        Returns
        -------
        jac_ : 2-d ndarray
            Jacobian matrix whose columns are derivatives with respect to all unconstrained state variables;
            including periods. Has dimensions dependent on number of spatiotemporal modes and free parameters,
            (self.shapes()[-1].size, self.shapes()[-1].size + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(self.constraints)

        Notes
        -----
        Original implementation was pretty, but very inefficient. This now computes the Jacobian matrix,
        minimizing the amount of allocated memory by overwriting and performing matrix-free implementations of
        Fourier transform matrix operations. Computes the Jacobian
        $J = D_t + D_xx + D_xxxx + F_t D_x F_x Diag(u) F_x^{-1} F_t^{-1}$ in the following steps:

        """
        assert self.basis == "modes"
        field_size, smode_size, mode_size = (np.prod(shp) for shp in self.shapes())
        # Begin with nonlinear term. Apply matrix operators in matrix-free fashion. begin with diag(u)
        J = np.diag(self.transform(to="field", array=True).ravel()).reshape(-1, self.m)
        # By creatively reshaping J, can apply FFTs to 3-d tensor.
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        ).transform(to="spatial_modes", array=True, inplace=True)
        J = J.reshape(field_size, smode_size).T.reshape(-1, self.m)

        # After transforming the columns, transpose and transform again to get F_x Diag(u) F_x^{-1}
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        )

        # Reshape back into a matrix, and then into another 3-d tensor after transposing back, to take derivative.
        J = J.transform(to="spatial_modes", array=True, inplace=True).reshape(
            smode_size, smode_size
        )
        J = J.T.reshape(-1, self.m - 2)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "basis": "spatial_modes",
                "discretization": (J.shape[0], self.m),
            }
        )
        J = J.dx(array=True, computation_basis="spatial_modes")

        # At this point J represents D_x F_x Diag(u) F_x^{-1}; reshape into 3-d tensor again and apply
        # time transforms
        J = J.reshape(-1, smode_size).T.reshape(self.n, self.m - 2, -1)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        # reshape, transpose and transform.
        J = (
            J.reshape((*self.shapes()[2], -1))
            .reshape(-1, smode_size)
            .T.reshape(self.n, self.m - 2, -1)
        )
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        J = J.reshape((*self.shapes()[2], -1)).reshape(J.shape[-1], J.shape[-1]).T

        # Produce the linear term; spatial derivatives are diagonal; time is more complicated due to SO(2) operator.
        e = np.ones(self.shapes()[2])
        dx2 = self.__class__(state=e, basis="modes", parameters=self.parameters).dx(
            order=2
        )
        J[np.diag_indices(J.shape[0])] += dx2.state.ravel() + dx2.state.ravel() ** 2
        e = e.ravel()
        # For time, get the correct frequency by taking the derivative of individual elements; swapping of real
        # and imaginary components handled by .dt()
        for i in range(J.shape[0]):
            e *= 0
            e[i] = 1.0
            J[:, i] += (
                self.__class__(
                    state=e.reshape(self.shape),
                    basis="modes",
                    parameters=self.parameters,
                )
                .dt()
                .state.ravel()
            )
            J[:, i] += (-self.s / self.t) * self.__class__(
                state=e.reshape(self.shape), basis="modes", parameters=self.parameters
            ).dx().state.ravel()

        J = self._jacobian_parameter_derivatives_concat(J)
        return J

    def change_reference_frame(self, frame):
        """
        Transform to (or from) the co-moving frame depending on the current reference frame

        Parameters
        ----------
        Returns
        -------
        RelativeOrbitKS :
            RelativeOrbitKS in transformed reference frame.

        Notes
        -----
        This operation occurs in spatial Fourier mode basis because I feel its more straightforward to understand
        a time parameterized shift of spatial modes; this is the consistent approach given how the shifts are
        calculated.

        Spatiotemporal modes are not designed to be used in the physical reference frame due to Gibbs' phenomenon
        due to discontinuity. Physical reference frame should really only be used to plot the relative periodic field.

        """
        # shift is ALWAYS stored as the shift amount from comoving to physical frame.
        if frame == "comoving":
            if self.frame == "physical":
                shift = -1.0 * self.s
            else:
                return self
        elif frame == "physical":
            if self.frame == "comoving":
                shift = self.s
            else:
                return self
        else:
            raise ValueError("Trying to change to unrecognizable reference frame.")

        spatial_modes = self.transform(to="spatial_modes").state
        time_vector = np.flipud(
            np.linspace(0, self.t, num=self.n, endpoint=True)
        ).reshape(-1, 1)
        translation_per_period = shift / self.t
        time_dependent_translations = translation_per_period * time_vector
        thetak = (
            time_dependent_translations
            * spatial_frequencies(self.x, self.m, 1)[:, : -int(self.m // 2 - 1)]
        )
        cosinek = np.cos(thetak)
        sinek = np.sin(thetak)
        real_modes = spatial_modes[:, : -(int(self.m // 2) - 1)]
        imag_modes = spatial_modes[:, -(int(self.m // 2) - 1) :]
        frame_rotated_spatial_modes_real = cosinek * real_modes + sinek * imag_modes
        frame_rotated_spatial_modes_imag = -sinek * real_modes + cosinek * imag_modes
        frame_rotated_spatial_modes = np.concatenate(
            (frame_rotated_spatial_modes_real, frame_rotated_spatial_modes_imag), axis=1
        )

        rotated_orbit = self.__class__(
            **{
                **vars(self),
                "state": frame_rotated_spatial_modes,
                "basis": "spatial_modes",
                "frame": frame,
            }
        )
        return rotated_orbit.transform(to=self.basis)

    def orbit_vector(self):
        """
        Vector which completely describes the orbit.

        """
        return np.concatenate(
            (
                self.state.reshape(-1, 1),
                np.array([[float(self.t)]]),
                np.array([[float(self.x)]]),
                np.array([[float(self.s)]]),
            ),
            axis=0,
        )

    def populate(self, attr="all", **kwargs):
        """
        Randomly initialize parameters which are currently zero.

        Parameters
        ----------

        attr : str
            Takes values 'all', 'parameters', or 'state'. Determines what is populated.

        kwargs :74
            p_ranges : dict
                keys are parameter_labels, values are uniform sampling intervals or iterables to sample from

        Returns
        -------
        Orbit :
            self with new values for parameters or state or both.

        """

        # Can only initialize spatial shift if both the shift and the parameters have been instantiated.
        if attr in ["all", "parameters"] and self.size > 0 and self.s == 0.0:
            shift = self.calculate_spatial_shift(**kwargs)
        else:
            shift = 0
        # This will first replace any None valued parameters (or if parameters itself is None)
        super().populate(attr=attr, **kwargs)
        # For chaining operations.
        parameters_with_shift = tuple(
            shift if label == "s" else val
            for label, val in zip(self.parameter_labels(), self.parameters)
        )
        setattr(self, "parameters", parameters_with_shift)
        return self

    def calculate_spatial_shift(self, **kwargs):
        """
        Calculate the phase difference between the spatial modes at t=0 and t=T

        Parameters
        ----------
        kwargs :
            n_modes : int
                Number of spatial modes to use in the phase calculation.

        Returns
        -------
        shift : float
            The best approximation for physical->comoving shift for relative periodic solutions.

        """
        spatial_modes = self.transform(to="spatial_modes").state
        m0 = spatial_modes.shape[1] // 2
        modes_included = np.min([kwargs.get("n_modes", m0), m0])
        if -m0 + modes_included == 0:
            space_imag_slice_end = None
        else:
            space_imag_slice_end = -m0 + modes_included
        # select the spatial modes at t=0 and t=T
        modes_0 = np.concatenate(
            (
                spatial_modes[-1, :modes_included],
                spatial_modes[-1, -m0:space_imag_slice_end],
            )
        ).ravel()
        modes_T = np.concatenate(
            (
                spatial_modes[0, :modes_included],
                spatial_modes[0, -m0:space_imag_slice_end],
            )
        ).ravel()
        m = modes_T.size // 2
        # This function is used very sparingly, extra imports kept in this scope only.
        # Warnings come from fsolve not converging; only want approximate guess as exact solution won't generally exist
        from scipy.optimize import fsolve

        # If they are close enough to the same point, then shift equals 0
        if np.linalg.norm(modes_0 - modes_T) <= 10 ** -6:
            shift = self.x / spatial_modes.shape[1]
        else:
            # Get guess shift from the angle between the vectors
            shift_guess = (self.x / (2 * pi)) * float(
                np.arccos(
                    (
                        np.dot(np.transpose(modes_T), modes_0)
                        / (np.linalg.norm(modes_T) * np.linalg.norm(modes_0))
                    )
                )
            )

            def fun_(shift_):
                # find shift which minimizes the differences at the boundaries.
                thetak = shift_ * ((2 * pi) / self.x) * np.arange(1, m + 1)
                cosinek = np.cos(thetak)
                sinek = np.sin(thetak)
                rotated_real_modes_T = np.multiply(cosinek, modes_T[:-m]) + np.multiply(
                    sinek, modes_T[-m:]
                )
                rotated_imag_modes_T = np.multiply(-sinek, modes_T[:-m]) + np.multiply(
                    cosinek, modes_T[-m:]
                )
                rotated_modes = np.concatenate(
                    (rotated_real_modes_T, rotated_imag_modes_T)
                )
                return np.linalg.norm(modes_0 - rotated_modes)

            # suppress fsolve's warnings that occur when it stalls; not expecting an exact answer anyway.
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            shift = fsolve(fun_, np.array(shift_guess))[0]
            warnings.resetwarnings()
            # because periodic boundary conditions take modulo; "overstretching" doesn't occur from physical limits.
            shift = np.sign(shift) * np.mod(np.abs(shift), self.x)

        if self.frame == "comoving":
            return -shift
        else:
            return shift

    def preprocess(self):
        """
        Check whether the orbit converged to an equilibrium or close-to-zero solution

        """
        orbit_with_inverted_shift = self.__class__(
            **{**vars(self), "parameters": (self.t, self.x, -self.s)}
        )
        cost_imported_S = self.cost()
        cost_negated_S = orbit_with_inverted_shift.cost()
        if cost_imported_S > cost_negated_S:
            orbit_ = orbit_with_inverted_shift
        else:
            orbit_ = self.copy()
        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        field_orbit = orbit_.transform(to="field")

        # See if the L_2 norm is beneath a threshold value, if so, replace with zeros.
        if field_orbit.norm() < 10 ** -5 or self.t == 0:
            return RelativeEquilibriumOrbitKS(
                state=np.zeros(self.discretization),
                basis="field",
                parameters=self.parameters,
            ).transform(to=self.basis)
        # Equilibrium is defined by having no temporal variation, i.e. time derivative is a uniformly zero.
        elif self.t == 0.0:
            # If there is sufficient evidence that solution is an equilibrium, change its class
            # store T just in case we want to refer to what the period was before conversion to EquilibriumOrbitKS
            return EquilibriumOrbitKS(
                state=field_orbit.state, basis="field", parameters=self.parameters
            ).transform(to=self.basis)
        elif field_orbit.dt().transform(to="field").norm() < 10 ** -5:
            # If there is sufficient evidence that solution is an equilibrium, change its class
            return RelativeEquilibriumOrbitKS(
                state=self.transform(to="modes").state,
                basis="modes",
                parameters=self.parameters,
            ).transform(to=self.basis)
        else:
            return orbit_

    def from_fundamental_domain(self):
        return self.change_reference_frame(frame="comoving")

    def to_fundamental_domain(self):
        return self.change_reference_frame(frame="physical")

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Default parameter ranges.

        Returns
        -------
        dict :
            Keys are parameter labels, values are intervals to sample from in parameter generation.

        Notes
        -----
        The shift parameter is included so that it can be iterated over; only s=0 allowed because this is
        determined by state, in the _populate_parameters method.

        """
        return {"t": (20, 200), "x": (20, 100), "s": (0, 0)}

    def _default_constraints(self):
        return {"t": False, "x": False, "s": False}

    def _eqn_linear_component(self, array=False):
        """
        Linear component of the KSE

        Parameters
        ----------
        array : bool
            Whether to return ndarray or not.

        Returns
        -------
        ndarray or class instance.

        """
        return (
            self.dt(array=array)
            + self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
            - (self.s / self.t) * self.dx(array=array)
        )

    def _rmatvec_parameters(self, self_field, other):
        other_modes = other.state.ravel()
        self_dx_modes = self.dx(array=True)
        if not self.constraints["t"]:
            # Derivative with respect to T term equal to DF/DT * v
            rmatvec_t = (-1.0 / self.t) * (
                self.dt(array=True) + (-self.s / self.t) * self_dx_modes
            ).ravel().dot(other_modes)
        else:
            rmatvec_t = 0

        if not self.constraints["x"]:
            # change in L, dx, equal to DF/DL * v
            rmatvec_x = (
                (
                    (-2.0 / self.x) * self.dx(order=2, array=True)
                    + (-4.0 / self.x) * self.dx(order=4, array=True)
                    + (-1.0 / self.x)
                    * (
                        self_field._nonlinear(self_field, array=True)
                        + (-self.s / self.t) * self_dx_modes
                    )
                )
                .ravel()
                .dot(other_modes)
            )

        else:
            rmatvec_x = 0

        if not self.constraints["s"]:
            rmatvec_s = (-1.0 / self.t) * self_dx_modes.ravel().dot(other_modes)
        else:
            rmatvec_s = 0.0

        return rmatvec_t, rmatvec_x, rmatvec_s

    def _rmatvec_linear_component(self, array=False):
        """
        Linear component of the adjoint Jacobian-vector product

        Parameters
        ----------
        array : bool
            Whether or not to return ndarray

        Returns
        -------
        Orbit or ndarray.

        """
        return (
            -1.0 * self.dt(array=array)
            + self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
            + (self.s / self.t) * self.dx(array=array)
        )

    def _pad(self, size, axis=0):
        """
        Checks if in comoving frame then pads. See OrbitKS for more details

        """
        assert (
            self.frame == "comoving"
        ), "Mode padding requires comoving frame; set padding=False if plotting"
        return super()._pad(size, axis=axis)

    def _truncate(self, size, axis=0):
        """
        Checks if in comoving frame then truncates. See OrbitKS for more details

        """
        assert (
            self.frame == "comoving"
        ), "Mode truncation requires comoving frame; set padding=False if plotting"
        return super()._truncate(size, axis=axis)

    def _jacobian_parameter_derivatives_concat(self, jac_):
        """
        Concatenate parameter partial derivatives to Jacobian matrix

        Parameters
        ----------
        jac_ : np.ndArray,
        (N-1) * (M-2) dimensional array resultant from taking the derivative of the spatioatemporal mapping
        with respect to Fourier modes.

        Returns
        -------
        Jacobian augmented with parameter partial derivatives as columns. Required to solve for changes to period,
        space period in optimization process. Makes the system rectangular; needs to be solved by least squares type
        methods.

        """
        # If period is not fixed, need to include dF/dt in jacobian matrix
        if not self.constraints["t"]:
            time_period_derivative = (-1.0 / self.t) * (
                self.dt(array=True) + (-self.s / self.t) * self.dx(array=True)
            ).reshape(-1, 1)
            jac_ = np.concatenate((jac_, time_period_derivative), axis=1)

        # If spatial period is not fixed, need to include dF/dx in jacobian matrix
        if not self.constraints["x"]:
            self_field = self.transform(to="field")
            spatial_period_derivative = (
                (-2.0 / self.x) * self.dx(order=2, array=True)
                + (-4.0 / self.x) * self.dx(order=4, array=True)
                + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
            )
            jac_ = np.concatenate(
                (jac_, spatial_period_derivative.reshape(-1, 1)), axis=1
            )

        if not self.constraints["s"]:
            spatial_shift_derivatives = (-1.0 / self.t) * self.dx(array=True)
            jac_ = np.concatenate(
                (jac_, spatial_shift_derivatives.reshape(-1, 1)), axis=1
            )

        return jac_

    def _jac_lin(self):
        """
        Extension of the OrbitKS method that includes the term for spatial translation symmetry.

        """
        return super()._jac_lin() + (-self.s / self.t) * self._dx_matrix()


class AntisymmetricOrbitKS(OrbitKS):
    def dx(self, **kwargs):
        """
        Spatial derivative of the current state.

        Parameters
        ----------

        kwargs :
            order :int
                The order of the derivative.
            array : bool
                Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
                Orbit instances.
            computation_basis : str
                The basis in which to compute the derivative.
            return_basis : str
                Which basis to return the ShiftReflectionOrbitKS in, if array=False.

        Returns
        ----------
        ShiftReflectionOrbitKS :
            Class instance whose spatiotemporal state represents the spatial derivative in the
            the basis of the original state.

        Notes
        -----
        See OrbitKS.dx() for more details

        """
        order = kwargs.get("order", 1)
        # can compute spatial derivative in spatial mode or spatiotemporal mode basis. spatial_modes basis is required
        # for orbits with discrete symmetry as they are orthogonal to the dx() direction.
        if order % 2:
            # odd ordered derivatives NEED to be in spatial mode basis
            return super().dx(**{**kwargs, "computation_basis": "spatial_modes"})
        else:
            # even ordered derivatives CAN be in the spatial mode basis or the spatiotemporal mode basis.
            return super().dx(**kwargs)

    def from_fundamental_domain(self, **kwargs):
        """
        Overwrite of parent method

        """
        field = self.transform(to="field")
        return self.__class__(
            **{
                **vars(self),
                "state": np.concatenate(
                    (field.reflection().state, field.state), axis=1
                ),
                "basis": "field",
                "parameters": (self.t, 2 * self.x, 0.0),
            }
        ).transform(to=self.basis)

    def to_fundamental_domain(self, half=0, **kwargs):
        """
        Overwrite of parent method

        """
        if half == 0:
            f_domain = self.transform(to="field").state[:, : -int(self.m // 2)]
        else:
            f_domain = self.transform(to="field").state[:, -int(self.m // 2) :]

        return self.__class__(
            **{
                **vars(self),
                "state": f_domain,
                "basis": "field",
                "parameters": (self.t, self.x / 2.0, 0.0),
            }
        ).transform(to=self.basis)

    def shapes(self):
        """
        State array shapes in different bases.

        """
        return (
            (self.n, self.m),
            (self.n, self.m - 2),
            (max([self.n - 1, 1]), (int(self.m // 2) - 1)),
        )

    def selection_rules(self):
        # Apply the pattern to (int(self.m//2) - 1) modes
        reflection_selection_rules_integer_flags = np.repeat(
            (np.arange(0, 2 * (self.n - 1)) % 2).ravel(), (int(self.m // 2) - 1)
        )
        # These indices are used for the transform as well as the transform matrices; therefore they are returned
        # in a format compatible with both; applying .nonzero() yields indices., resize(self.shapes()[2]) yields
        # tensor format.
        return reflection_selection_rules_integer_flags

    @staticmethod
    def _default_shape():
        """
        The shape of a generic state, see core.py for details

        """
        return 1, 32

    def _nonlinear(self, other, array=False):
        """
        nonlinear computation of the nonlinear term of the Kuramoto-Sivashinsky equation

        """
        # Elementwise product, both self and other should be in physical field basis.
        assert (self.basis == "field") and (other.basis == "field")
        # to get around the special behavior of discrete symmetries, will return spatial modes without this workaround.
        nl_orbit = 0.5 * (self * other).transform(to="spatial_modes").dx(
            computation_basis="spatial_modes", return_basis="modes"
        )
        if array:
            return nl_orbit.state
        else:
            return nl_orbit

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Default parameter ranges.

        Returns
        -------
        dict :
            Keys are parameter labels, values are intervals to sample from in parameter generation.

        Notes
        -----
        The shift parameter is included so that it can be iterated over; only s=0 allowed because this is
        determined by state, in the _populate_parameters method.
        L=38.5 based on Yueheng Lan and Cvitanovic investigations

        """

        return {"t": (20.0, 200.0), "x": (38.5, 100.0)}

    def _pad(self, size, axis=0):
        """
        Overwrite of parent method

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                # Split into real and imaginary components, pad separately.
                padding = (size - modes.n) // 2
                padding_tuple = ((padding, padding), (0, 0))
                padded_modes = np.concatenate(
                    (
                        modes.state[: -(modes.n // 2 - 1), :],
                        np.pad(modes.state[-(modes.n // 2 - 1) :, :], padding_tuple),
                    ),
                    axis=0,
                )
                padded_modes *= np.sqrt(size / modes.n)
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                padding_number = (size - modes.m) // 2
                padded_modes = np.sqrt(size / modes.m) * np.pad(
                    modes.state, ((0, 0), (0, padding_number))
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Overwrite of parent method

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                truncate_number = int(size // 2) - 1
                first_half = modes.state[: truncate_number + 1, :]
                second_half = modes.state[
                    -(modes.n // 2 - 1) : -(modes.n // 2 - 1) + truncate_number, :
                ]
                truncated_modes = np.sqrt(size / modes.n) * np.concatenate(
                    (first_half, second_half), axis=0
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                truncate_number = int(size // 2) - 1
                truncated_modes = (
                    np.sqrt(size / modes.m) * modes.state[:, :truncate_number]
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _parse_state(self, state, basis, **kwargs):
        if isinstance(state, np.ndarray):
            if len(state.shape) != 2:
                raise ValueError('"state" array must be two-dimensional')
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(0, 0)

        if self.size > 0:
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            elif basis == "modes":
                n, m = self.shape[0] + 1, 2 * self.shape[1] + 2
            elif basis == "field":
                n, m = self.shape
            elif basis == "spatial_modes":
                n, m = self.shape[0], self.shape[1] + 2
            else:
                raise ValueError(
                    'basis not recognized; must equal "field" or "spatial_modes", or "modes"'
                )
            if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
                warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
                warnings.warn(warn_str, RuntimeWarning)
            self.basis = basis
            self.discretization = n, m
        else:
            self.discretization = None
            self.basis = None

    def _time_transform_matrix(self):
        """

        Notes
        -----
        Dramatic simplification over old code; now just the full DFT matrix plus projection

        """
        return super()._time_transform_matrix()[self.selection_rules().nonzero()[0], :]

    def _jac_nonlin(self):
        """
        The nonlinear component of the Jacobian matrix of the Kuramoto-Sivashinsky equation

        Returns
        -------
        nonlinear_dx : matrix
            Matrix which represents the nonlinear component of the Jacobian.

        Notes
        -----
        See OrbitKS for more details.

        """

        _jac_nonlin_left = self._time_transform_matrix().dot(
            self._dx_matrix(computation_basis="spatial_modes")
        )
        _jac_nonlin_middle = self._space_transform_matrix().dot(
            np.diag(self.transform(to="field").state.ravel())
        )
        _jac_nonlin_right = self._inv_spacetime_transform_matrix()
        _jac_nonlin = _jac_nonlin_left.dot(_jac_nonlin_middle).dot(_jac_nonlin_right)

        return _jac_nonlin

    def _time_transform(self, array=False, inplace=False):
        """
        Spatial Fourier transform

        Parameters
        ----------

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        if inplace:
            # Take rfft, accounting for unitary normalization.
            self.state = rfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            self.state = np.concatenate(
                (
                    self.state.real[:-1, -(int(self.m // 2) - 1) :],
                    self.state.imag[1:-1, -(int(self.m // 2) - 1) :],
                ),
                axis=0,
            )
            self.state[1:, :] = np.sqrt(2) * self.state[1:, :]
            self.basis = "modes"
            if array:
                return self.state
            else:
                return self
        else:
            # Take rfft, accounting for unitary normalization.
            modes = rfft(self.state, workers=self.workers, norm="ortho", axis=0)
            modes = np.concatenate(
                (
                    modes.real[:-1, -(int(self.m // 2) - 1) :],
                    modes.imag[1:-1, -(int(self.m // 2) - 1) :],
                ),
                axis=0,
            )
            modes[1:, :] = np.sqrt(2) * modes[1:, :]
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "modes"}
                )

    def _inv_time_transform(self, array=False, inplace=False):
        """
        Spatial Fourier transform

        Parameters
        ----------

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        if inplace:
            # can take advantage of the array sizes by adding to current array and multiplying by zero instead
            # of concatenating.
            self.state = self.state.astype(complex)
            self.state[1 : -max([int(self.n // 2) - 1, 1]), :] += (
                1j * self.state[-max([int(self.n // 2) - 1, 1]) :, :]
            )
            self.state = self.state[: -max([int(self.n // 2) - 1, 1]) + 1, :]
            self.state[1:, :] /= np.sqrt(2)
            self.state[-1, :] = 0
            self.state = irfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            self.state = np.concatenate((0 * self.state, self.state), axis=1)
            self.basis = "spatial_modes"
            if array:
                return self.state
            else:
                return self
        else:
            # Take rfft, accounting for unitary normalization.
            modes = self.state.astype(complex)
            modes[1 : -max([int(self.n // 2) - 1, 1]), :] += (
                1j * modes[-max([int(self.n // 2) - 1, 1]) :, :]
            )
            modes = modes[: -max([int(self.n // 2) - 1, 1]) + 1, :]
            modes[1:, :] /= np.sqrt(2)
            modes[-1, :] = 0
            modes = irfft(modes, workers=self.workers, norm="ortho", axis=0)
            modes = np.concatenate((0 * modes, modes), axis=1)
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "spatial_modes"}
                )


class ShiftReflectionOrbitKS(OrbitKS):
    def dx(self, **kwargs):
        """
        Spatial derivative of the current state.

        Parameters
        ----------
        kwargs :
            order :int
                The order of the derivative.
            array : bool
                Whether or not to return a numpy array. Used for efficiency/avoiding construction of redundant
                Orbit instances.
            computation_basis : str
                The basis in which to compute the derivative.
            return_basis : str
                Which basis to return the ShiftReflectionOrbitKS in, if array=False.

        Returns
        ----------
        ShiftReflectionOrbitKS :
            Class instance whose spatiotemporal state represents the spatial derivative in the
            the basis of the original state.

        Notes
        -----
        See OrbitKS.dx() for more details

        """
        order = kwargs.get("order", 1)
        # can compute spatial derivative in spatial mode or spatiotemporal mode basis. spatial_modes basis is required
        # for orbits with discrete symmetry as they are orthogonal to the dx() direction.
        if order % 2:
            # odd ordered derivatives NEED to be in spatial mode basis
            return super().dx(**{**kwargs, "computation_basis": "spatial_modes"})
        else:
            # even ordered derivatives CAN be in the spatial mode basis or the spatiotemporal mode basis.
            return super().dx(**kwargs)

    def selection_rules(self):
        """
        Symmetry selection rules

        """
        # equivalent to indices 0 + j from thesis; time indices go like {0, j, j}
        i = np.arange(0, self.n // 2)[:, None]
        selection_tensor_pattern = np.block(
            [[i % 2, (i + 1) % 2], [i[1:] % 2, (i[1:] + 1) % 2]]
        )
        # Apply the pattern to (int(orbit_.m//2) - 1) modes
        shiftreflection_selection_rules_integer_flags = np.repeat(
            selection_tensor_pattern.ravel(), (int(self.m // 2) - 1)
        )
        # These indices are used for the transform as well as the transform matrices; therefore they are returned
        # in a format compatible with both; applying .nonzero() yields indices., resize(self.shapes()[2]) yields
        # tensor format.
        return shiftreflection_selection_rules_integer_flags

    def shapes(self):
        """
        State array shapes in different bases. See core.py for details.

        """
        return (
            (self.n, self.m),
            (self.n, self.m - 2),
            (max([self.n - 1, 1]), (int(self.m // 2) - 1)),
        )

    def to_fundamental_domain(self, half=0):
        """
        Overwrite of parent method

        """
        field = self.transform(to="field").state
        if half == 0:
            f_domain = field[-int(self.n // 2) :, :]
        else:
            f_domain = field[: -int(self.n // 2), :]
        return self.__class__(
            **{
                **vars(self),
                "state": f_domain,
                "basis": "field",
                "parameters": (self.t / 2.0, self.x, 0.0),
            }
        ).transform(to=self.basis)

    def from_fundamental_domain(self):
        """
        Reconstruct full field from discrete fundamental domain

        """
        field = np.concatenate((self.reflection().state, self.state), axis=0)
        return self.__class__(
            **{
                **vars(self),
                "state": field,
                "basis": "field",
                "parameters": (2 * self.t, self.x, 0.0),
            }
        ).transform(to=self.basis)

    def _nonlinear(self, other, array=False):
        """
        nonlinear computation of the nonlinear term of the Kuramoto-Sivashinsky equation

        """
        # Elementwise product, both self and other should be in physical field basis.
        assert (self.basis == "field") and (other.basis == "field")
        # to get around the special behavior of discrete symmetries, will return spatial modes without this workaround.
        nl_orbit = 0.5 * (self * other).transform(to="spatial_modes").dx(
            computation_basis="spatial_modes", return_basis="modes"
        )
        if array:
            return nl_orbit.state
        else:
            return nl_orbit

    def _pad(self, size, axis=0):
        """
        Overwrite of parent method

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                padding = (size - modes.n) // 2
                padding_tuple = ((padding, padding), (0, 0))
                padded_modes = np.concatenate(
                    (
                        modes.state[: -(modes.n // 2 - 1), :],
                        np.pad(modes.state[-(modes.n // 2 - 1) :, :], padding_tuple),
                    ),
                    axis=0,
                )
                padded_modes *= np.sqrt(size / modes.n)
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                padding_number = (size - modes.m) // 2
                padded_modes = np.sqrt(size / modes.m) * np.pad(
                    modes.state, ((0, 0), (0, padding_number))
                )

                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Overwrite of parent method

        """
        modes = self.transform(to="modes")
        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                truncate_number = int(size // 2) - 1
                first_half = modes.state[: truncate_number + 1, :]
                second_half = modes.state[
                    -(modes.n // 2 - 1) : -(modes.n // 2 - 1) + truncate_number, :
                ]
                truncated_modes = np.sqrt(size / modes.n) * np.concatenate(
                    (first_half, second_half), axis=0
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                truncate_number = int(size // 2) - 1
                truncated_modes = (
                    np.sqrt(size / modes.m) * modes.state[:, :truncate_number]
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": truncated_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _jac_nonlin(self):
        """
        The nonlinear component of the Jacobian matrix of the Kuramoto-Sivashinsky equation

        Returns
        -------
        nonlinear_dx : matrix
            Matrix which represents the nonlinear component of the Jacobian.

        Notes
        -----
        See OrbitKS for more details.

        """
        _jac_nonlin_left = self._time_transform_matrix().dot(
            self._dx_matrix(computation_basis="spatial_modes")
        )
        _jac_nonlin_middle = self._space_transform_matrix().dot(
            np.diag(self.transform(to="field").state.ravel())
        )
        _jac_nonlin_right = self._inv_spacetime_transform_matrix()
        _jac_nonlin = _jac_nonlin_left.dot(_jac_nonlin_middle).dot(_jac_nonlin_right)
        return _jac_nonlin

    def _parse_state(self, state, basis, **kwargs):
        if isinstance(state, np.ndarray):
            if len(state.shape) != 2:
                raise ValueError('"state" array must be two-dimensional')
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(0, 0)
        if self.size > 0:
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            elif basis == "modes":
                n, m = self.shape[0] + 1, 2 * self.shape[1] + 2
            elif basis == "field":
                n, m = self.shape
            elif basis == "spatial_modes":
                n, m = self.shape[0], self.shape[1] + 2
            else:
                raise ValueError(
                    'basis not recognized; must equal "field" or "spatial_modes", or "modes"'
                )
            if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
                warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
                warnings.warn(warn_str, RuntimeWarning)
            self.basis = basis
            self.discretization = n, m
        else:
            self.discretization = None
            self.basis = None

    def _time_transform(self, array=False, inplace=False):
        """
        Spatial Fourier transform

        Parameters
        ----------

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        if inplace:
            # Take rfft, accounting for orthogonal normalization.
            self.state = rfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            # Project onto shift-reflection subspace.
            self.state[::2, : -(int(self.m // 2) - 1)] = 0
            self.state[1::2, -(int(self.m // 2) - 1) :] = 0
            # Due to projection, can add the different components without mixing information, this allows
            # us to avoid a complex operation like shuffling.

            self.state = np.concatenate(
                (
                    (
                        self.state.real[:-1, : -(int(self.m // 2) - 1)]
                        + self.state.real[:-1, -(int(self.m // 2) - 1) :]
                    ),
                    (
                        self.state.imag[1:-1, : -(int(self.m // 2) - 1)]
                        + self.state.imag[1:-1, -(int(self.m // 2) - 1) :]
                    ),
                ),
                axis=0,
            )
            self.state[1:, :] = np.sqrt(2) * self.state[1:, :]
            self.basis = "modes"
            if array:
                return self.state
            else:
                return self
        else:
            # Take rfft, accounting for orthogonal normalization.
            modes = rfft(self.state, workers=self.workers, norm="ortho", axis=0)
            # Project onto shift-reflection subspace.
            modes[::2, : -(int(self.m // 2) - 1)] = 0
            modes[1::2, -(int(self.m // 2) - 1) :] = 0
            # Due to projection, can add the different components without mixing information, this allows
            # us to avoid a complex operation like shuffling.
            modes = np.concatenate(
                (
                    (
                        modes.real[:-1, : -(int(self.m // 2) - 1)]
                        + modes.real[:-1, -(int(self.m // 2) - 1) :]
                    ),
                    (
                        modes.imag[1:-1, : -(int(self.m // 2) - 1)]
                        + modes.imag[1:-1, -(int(self.m // 2) - 1) :]
                    ),
                ),
                axis=0,
            )
            modes[1:, :] = np.sqrt(2) * modes[1:, :]
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "modes"}
                )

    def _inv_time_transform(self, array=False, inplace=False):
        """
        Spatial Fourier transform

        Parameters
        ----------
        array : bool
            Whether to return np.ndarray or Orbit subclass instance

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is in the spatial Fourier mode basis.

        """
        if inplace:
            # can take advantage of the array sizes by adding to current array and multiplying by zero instead
            # of concatenating.
            self.state = self.state.astype(complex)
            self.state[1 : -max([int(self.n // 2) - 1, 1]), :] += (
                1j * self.state[-max([int(self.n // 2) - 1, 1]) :, :]
            )
            self.state = self.state[: -max([int(self.n // 2) - 1, 1]) + 1, :]
            self.state[1:, :] /= np.sqrt(2)
            self.state[-1, :] = 0
            self.state = np.concatenate((self.state, self.state), axis=1)
            self.state[::2, : -(int(self.m // 2) - 1)] = 0
            self.state[1::2, -(int(self.m // 2) - 1) :] = 0
            self.state = irfft(
                self.state, workers=self.workers, norm="ortho", axis=0, overwrite_x=True
            )
            self.basis = "spatial_modes"
            if array:
                return self.state
            else:
                return self
        else:
            modes = self.state.copy().astype(complex)
            modes[1 : -max([int(self.n // 2) - 1, 1]), :] += (
                1j * modes[-max([int(self.n // 2) - 1, 1]) :, :]
            )
            modes = modes[: -max([int(self.n // 2) - 1, 1]) + 1, :]
            modes[1:, :] /= np.sqrt(2)
            modes[-1, :] = 0
            modes = np.concatenate((modes, modes), axis=1)
            modes[::2, : -(int(self.m // 2) - 1)] = 0
            modes[1::2, -(int(self.m // 2) - 1) :] = 0
            modes = irfft(modes, workers=self.workers, norm="ortho", axis=0)
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "spatial_modes"}
                )

    def _time_transform_matrix(self):
        """

        Notes
        -----
        Dramatic simplification over old code; now just the full DFT matrix plus projection

        """
        return super()._time_transform_matrix()[self.selection_rules().nonzero()[0], :]

    def _inv_time_transform_matrix(self):
        """

        Notes
        -----
        Dramatic simplification over old code; now just transpose of forward dft matrix b.c. orthogonal

        """
        return self._time_transform_matrix().transpose()


class EquilibriumOrbitKS(AntisymmetricOrbitKS):
    """
    Class for temporal equilibria

    Notes
    -----
    For convenience, this subclass accepts any (even) value for the time discretization. Only a single time point
    is required however to fully represent the solution and therefore perform any computations. If the
    discretization size is greater than 1 then then different bases will have the following shapes: field (N, M).
    spatial modes = (N, m), spatiotemporal modes (1, m). In other words, discretizations
    of this type can still be used in the optimization codes but will be much more inefficient.
    The reason for this choice is because it is possible
    to start with a spatiotemporal orbit with no symmetry (i.e. OrbitKS) and still end up at an equilibrium
    solution. Therefore, I am accommodating transformations from other orbit types to EquilibriumOrbitKS. To
    make the computations more efficient all that is required is usage of the method
    self.optimize_for_calculations(), which converts N -> 1, making the shape of the state (1, M) in the
    physical field basis. Also can inherit more methods with this choice.
    More details are included in the thesis and in the documentation.
    While only the imaginary components of the spatial modes are non-zero, both real and imaginary components are
    kept to allow for odd order spatial derivatives, required for the nonlinear term. Other allowed operations such
    as rotation are preferably performed after converting to a different symmetry type such as AntisymmetricOrbitKS
    or OrbitKS.

    """

    @staticmethod
    def minimal_shape():
        """
        The smallest possible compatible discretization

        Returns
        -------
        tuple of int :
            The default array shape when dimensions are not specified.

        Notes
        -----
        Often symmetry constraints reduce the dimensionality; if too small this reduction may leave the state empty,
        used for aspect ratio correction and possibly other gluing applications.

        If parity is important, than that must be built into the definition.

        """
        return 1, 4

    def dt(self, order=1, array=False, **kwargs):
        """
        A time derivative of the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.

        array : bool
            Whether to return np.ndarray or Orbit

        Returns
        ----------
        EquilibriumOrbitKS or np.ndarray :
            Orbit or array representation of the time derivative of an equilibrium (i.e. zero).

        """
        if array:
            return np.zeros(self.state.shape)
        else:
            return self.__class__(
                **{**vars(self), "state": np.zeros(self.shape), "basis": "modes"}
            )

    def jacobian(self, **kwargs):
        """
        Jacobian matrix evaluated at the current state.

        kwargs :
            Unused, included to match signature in Orbit class.

        Returns
        -------
        jac_ : 2-d ndarray
            Jacobian matrix whose columns are derivatives with respect to all unconstrained state variables;
            including periods. Has dimensions dependent on number of spatiotemporal modes and free parameters,
            (self.shapes()[-1].size, self.shapes()[-1].size + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(self.constraints)

        Notes
        -----
        Original implementation was pretty, but very inefficient. This now computes the Jacobian matrix,
        minimizing the amount of allocated memory by overwriting and performing matrix-free implementations of
        Fourier transform matrix operations. Computes the Jacobian
        $J = D_t + D_xx + D_xxxx + F_t D_x F_x Diag(u) F_x^{-1} F_t^{-1}$ in the following steps:

        """
        assert self.basis == "modes"
        field_size, smode_size, mode_size = (np.prod(shp) for shp in self.shapes())
        # Begin with nonlinear term. Apply matrix operators in matrix-free fashion. begin with diag(u)
        J = np.diag(self.transform(to="field", array=True).ravel()).reshape(-1, self.m)
        # By creatively reshaping J, can apply FFTs to 3-d tensor.
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        ).transform(to="spatial_modes", array=True, inplace=True)
        J = J.reshape(field_size, smode_size).T.reshape(-1, self.m)

        # After transforming the columns, transpose and transform again to get F_x Diag(u) F_x^{-1}
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        )

        # Reshape back into a matrix, and then into another 3-d tensor after transposing back, to take derivative.
        J = J.transform(to="spatial_modes", array=True, inplace=True).reshape(
            smode_size, smode_size
        )
        J = J.T.reshape(-1, self.m - 2)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "basis": "spatial_modes",
                "discretization": (J.shape[0], self.m),
            }
        )
        J = J.dx(array=True, computation_basis="spatial_modes")

        # At this point J represents D_x F_x Diag(u) F_x^{-1}; reshape into 3-d tensor again and apply
        # time transforms
        J = J.reshape(-1, smode_size).T.reshape(self.n, self.m - 2, -1)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        # reshape, transpose and transform.
        J = (
            J.reshape((*self.shapes()[2], -1))
            .reshape(-1, smode_size)
            .T.reshape(self.n, self.m - 2, -1)
        )
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        J = J.reshape((*self.shapes()[2], -1)).reshape(J.shape[-1], J.shape[-1]).T

        # Produce the linear term; spatial derivatives are diagonal; time is more complicated due to SO(2) operator.
        e = np.ones(self.shapes()[2])
        dx2 = self.__class__(state=e, basis="modes", parameters=self.parameters).dx(
            order=2
        )
        J[np.diag_indices(J.shape[0])] += dx2.state.ravel() + dx2.state.ravel() ** 2
        J = self._jacobian_parameter_derivatives_concat(J)
        return J

    def orbit_vector(self):
        """
        Overwrite of parent method

        """
        return np.concatenate(
            (self.state.reshape(-1, 1), np.array([[float(self.x)]])), axis=0
        )

    def shapes(self):
        """
        State array shapes in different bases. See core.py for details.

        """
        return (self.n, self.m), (self.n, self.m - 2), (1, (int(self.m // 2) - 1))

    def precondition(self, **kwargs):
        """
        Precondition a vector with the inverse (aboslute value) of linear spatial terms

        Parameters
        ----------
        kwargs : dict
            pmult : tuple of tuples
                Values for the frequencies to use in the rescaling
            pexp : tuple of int
                The exponentiation factor used to rescale parameter corrections.

        Returns
        -------
        target : OrbitKS
            Return the OrbitKS instance, modified by preconditioning.

        Notes
        -----
        Often we want to precondition a state derived from a mapping or rmatvec (gradient descent step),
        with respect to ANOTHER orbit's (current state's) parameters.

        """
        pmult = kwargs.get("pmult", self.preconditioning_parameters())
        pexp = kwargs.get("pexp", (0, 4))

        p_multipliers = 1.0 / (
            np.abs(spatial_frequencies(*pmult[1], order=2)[:, : self.shapes()[2][1]])
            + spatial_frequencies(*pmult[1], order=4)[:, : self.shapes()[2][1]]
        )
        preconditioned_state = np.multiply(self.state, p_multipliers)

        # Precondition the change in T and L so that they do not dominate
        if not self.constraints["x"]:
            # self is the orbit being preconditioned, i.e. the correction orbit; by default this is dx = dx / L^4
            x = self.x * (pmult[1][0] ** -pexp[1])
        else:
            x = self.x

        return self.__class__(
            **{
                **vars(self),
                "state": preconditioned_state,
                "basis": "modes",
                "parameters": (0, x, 0),
            }
        )

    def preprocess(self):
        """
        Check whether the orbit converged to an equilibrium or close-to-zero solution

        """
        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        field_orbit = self.transform(to="field")

        # See if the L_2 norm is beneath a threshold value, if so, replace with zeros.
        if field_orbit.norm() < 10 ** -5:
            return self.__class__(
                **{
                    **vars(self),
                    "state": np.zeros(self.discretization),
                    "basis": "field",
                }
            ).transform(to=self.basis)
        else:
            return self

    def from_fundamental_domain(self, **kwargs):
        """
        Overwrite of parent method

        """
        field = self.transform(to="field")
        return self.__class__(
            **{
                **vars(self),
                "state": np.concatenate(
                    (field.reflection().state, field.state), axis=1
                ),
                "basis": "field",
                "parameters": (0.0, 2.0 * self.x, 0.0),
            }
        ).transform(to=self.basis)

    def to_fundamental_domain(self, half=0, **kwargs):
        """
        Overwrite of parent method

        """
        if half == 0:
            f_domain = self.transform(to="field").state[:, : -int(self.m // 2)]
        else:
            f_domain = self.transform(to="field").state[:, -int(self.m // 2) :]

        return self.__class__(
            **{
                **vars(self),
                "state": f_domain,
                "basis": "field",
                "parameters": (0.0, self.x / 2.0, 0.0),
            }
        )

    @classmethod
    def dimension_based_discretization(cls, dimensions, **kwargs):
        """
        Orbithunter conventions for discretization size.

        Parameters
        ----------
        dimensions : tuple
            tuple containing (t, L, S) i.e. OrbitKS().parameters

        kwargs :
            resolution : str
                Takes values 'coarse', 'normal', 'fine'. These options return one of three orbithunter conventions for
                the discretization size.
            N : int, default None
                Temporal discretization size, if provided
            M : int, default None
                Spatial discretization size, if provided
        Returns
        -------
        int, int
        The new spatiotemporal field discretization; number of time points (rows) and number of space points (columns)

        Notes
        -----
        This function should only ever be called by rediscretize, the returned values can always be accessed by
        the appropriate attributes of the rediscretized orbit.

        """
        # Change the default to N = 1 from N = None, this ensures that the temporal period (t=0) is never used.
        resolution = kwargs.get("resolution", "default")
        n, m = kwargs.get("resolution", (1, None))
        t, x = dimensions
        if m is None:
            if x == 0:
                m = cls._default_shape()[1]
            elif resolution == "coarse":
                m = np.max([2 ** (int(np.log2(x) - 1)), 8])
            elif resolution == "fine":
                m = np.max([2 ** (int(np.log2(x) + 2)), 16])
            elif resolution == "power":
                m = np.max([2 * (int(4 * x ** (1.0 / 2.0)) // 2), 16])
            else:
                m = np.max([2 ** (int(np.log2(x) + 0.5)), 16])
            m = max(m, cls.minimal_shape()[1])
        return n, m

    def _default_constraints(self):
        return {"x": False}

    def _eqn_linear_component(self, array=False):
        """
        Linear component of the KSE

        Parameters
        ----------
        array : bool
            Whether to return ndarray or not.

        Returns
        -------
        ndarray or class instance.

        """
        return self.dx(order=2, array=array) + self.dx(order=4, array=array)

    def _rmatvec_parameters(self, self_field, other):
        other_modes_in_vector_form = other.state.ravel()
        if not self.constraints["x"]:
            # change in L, dx, equal to DF/DL * v
            rmatvec_x = (
                (
                    (-2.0 / self.x) * self.dx(order=2, array=True)
                    + (-4.0 / self.x) * self.dx(order=4, array=True)
                    + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
                )
                .ravel()
                .dot(other_modes_in_vector_form)
            )
        else:
            rmatvec_x = 0

        return 0.0, rmatvec_x, 0.0

    def _rmatvec_linear_component(self, array=False):
        """
        Linear component of the adjoint Jacobian-vector product

        Parameters
        ----------
        array : bool
            Whether or not to return ndarray

        Returns
        -------
        Orbit or ndarray.

        """
        return self._eqn_linear_component(array=array)

    def _pad(self, size, axis=0):
        """
        Overwrite of parent method

        Notes
        -----
        If starting and ending in the spatiotemporal modes basis, this will only create an instance with a different
        value of time dimensionality attribute 'N'.

        """
        spatial_modes = self.transform(to="spatial_modes")
        if axis == 0:
            # Not technically zero-padding, just copying. Just can't be in temporal mode basis
            # because it is designed to only represent the zeroth modes.
            padded_spatial_modes = np.tile(spatial_modes.state[None, 0, :], (size, 1))
            return self.__class__(
                **{
                    **vars(self),
                    "state": padded_spatial_modes,
                    "basis": "spatial_modes",
                    "discretization": (size, self.m),
                }
            ).transform(to=self.basis)
        else:
            # Split into real and imaginary components, pad separately.
            padding = (size - self.m) // 2
            padding_tuple = ((0, 0), (padding, padding))
            padded_modes = np.concatenate(
                (
                    spatial_modes.state[:, : -(self.m // 2 - 1)],
                    np.pad(spatial_modes.state[:, -(self.m // 2 - 1) :], padding_tuple),
                ),
                axis=1,
            )
            padded_modes *= np.sqrt(size / self.m)
            return self.__class__(
                **{
                    **vars(self),
                    "state": padded_modes,
                    "basis": "spatial_modes",
                    "discretization": (self.n, size),
                }
            ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Overwrite of parent method

        """
        if axis == 0:
            spatial_modes = self.transform(to="spatial_modes")
            truncated_spatial_modes = spatial_modes.state[-size:, :]
            return self.__class__(
                **{
                    **vars(self),
                    "state": truncated_spatial_modes,
                    "basis": "spatial_modes",
                    "discretization": (size, self.m),
                }
            ).transform(to=self.basis)
        else:
            modes = self.transform(to="modes")
            truncate_number = int(size // 2) - 1
            truncated_modes = np.sqrt(size / modes.m) * modes.state[:, :truncate_number]
            # cannot distinguish between n != 1 and n == 1 when in modes basis; therefore, pass the shape to keep track
            return self.__class__(
                **{
                    **vars(self),
                    "state": truncated_modes,
                    "basis": "modes",
                    "discretization": (self.n, size),
                }
            ).transform(to=self.basis)

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Default parameter ranges.

        Returns
        -------
        dict :
            Keys are parameter labels, values are intervals to sample from in parameter generation.

        Notes
        -----
        Default spatial domain size; minimum L=2*pi based on fundamental orbit.

        """
        return {"x": (2 * pi, 100.0)}

    def _parse_state(self, state, basis, **kwargs):
        """
        Parse the provided state information of an equilibrium state.

        """
        if isinstance(state, np.ndarray):
            if len(state.shape) != 2:
                raise ValueError('"state" array must be two-dimensional')
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(0, 0)

        if self.size > 0:

            if len(self.shape) == 1:
                self.state = state[None, :]

            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            elif basis == "modes":
                # If passed as modes; which only contain the zeroth modes, the shape in the other bases
                # cannot be inferred from the NumPy array; it must either be provided via discretization keyword
                # or it will default to the minimal value.
                self.state = self.state[None, 0, :]
                n = kwargs.get("discretization", (None, None))[0] or 1
                m = 2 * self.shape[1] + 2
            elif basis == "field":
                n, m = self.shape
            elif basis == "spatial_modes":
                n, m = self.shape[0], self.shape[1] + 2
            else:
                raise ValueError(
                    'basis not recognized; must equal "field" or "spatial_modes", or "modes"'
                )
            if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
                warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
                warnings.warn(warn_str, RuntimeWarning)
            self.discretization = n, m
            self.basis = basis
        else:
            self.discretization = None
            self.basis = None

    def _jac_lin(self):
        """
        Extension of the OrbitKS method that includes the term for spatial translation symmetry

        """
        return self._dx_matrix(order=2) + self._dx_matrix(order=4)

    def _jacobian_parameter_derivatives_concat(self, jac_):
        """
        Concatenate parameter partial derivatives to Jacobian matrix

        Parameters
        ----------
        jac_ : np.ndArray,
        (N-1) * (M-2) dimensional array resultant from taking the derivative of the spatioatemporal mapping
        with respect to Fourier modes.

        Returns
        -------
        Jacobian augmented with parameter partial derivatives as columns. Required to solve for changes to period,
        space period in optimization process. Makes the system rectangular; needs to be solved by least squares type
        methods.

        """
        # If spatial period is not fixed, need to include dF/dx in jacobian matrix
        if not self.constraints["x"]:

            self_field = self.transform(to="field")
            spatial_period_derivative = (
                (-2.0 / self.x) * self.dx(order=2, array=True)
                + (-4.0 / self.x) * self.dx(order=4, array=True)
                + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
            )
            jac_ = np.concatenate(
                (jac_, spatial_period_derivative.reshape(-1, 1)), axis=1
            )

        return jac_

    def _inv_time_transform_matrix(self):
        """
        Overwrite of parent method

        Notes
        -----
        Originally this transform just selected the antisymmetric spatial modes (imaginary component),
        but in order to be consistent with all other time transforms, I will actually apply the normalization
        constant associated with a forward in time transformation. The reason for this is for comparison
        of states between different subclasses.

        """
        return np.tile(
            np.concatenate(
                (0 * np.eye((int(self.m // 2) - 1)), np.eye((int(self.m // 2) - 1))),
                axis=0,
            ),
            (self.n, 1),
        )

    def _time_transform_matrix(self):
        """
        Overwrite of parent method

        """

        time_dft_mat = np.tile(
            np.concatenate(
                (0 * np.eye((int(self.m // 2) - 1)), np.eye((int(self.m // 2) - 1))),
                axis=1,
            ),
            (1, self.n),
        )
        time_dft_mat[:, 2 * (int(self.m // 2) - 1) :] = 0
        return time_dft_mat

    def _time_transform(self, array=False, inplace=False):
        """
        Overwrite of parent method

        Notes
        -----
        Taking the RFFT, with orthogonal normalization, of a constant time series defined on N points is equivalent
        to multiplying the constant value by sqrt(N). This is because the transform sums over N repeats of the same
        value and then divides by 1/sqrt(N). i.e. (N * C)/sqrt(N) = sqrt(N) * C. Therefore we can save some time by
        just doing this without calling the rfft function.

        """
        # Select the nonzero (imaginary) components of modes and transform in time (w.r.t. axis=0).
        if inplace:
            # reshapes but also generalizes for Jacobian calculation
            self.state = self.state[np.newaxis, 0, -(int(self.m // 2) - 1) :]
            self.basis = "modes"
            if array:
                return self.state
            else:
                return self
        else:
            modes = self.state[np.newaxis, 0, -(int(self.m // 2) - 1) :]
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "modes"}
                )

    def _inv_time_transform(self, array=False, inplace=False):
        """
        Overwrite of parent method

        Notes
        -----
        Taking the IRFFT, with orthogonal normalization is equivalent to dividing by the normalization constant; because
        there would only be

        """
        if inplace:
            real = np.zeros(self.state.shape)
            imaginary = self.state
            self.state = np.tile(np.concatenate((real, imaginary), axis=1), (self.n, 1))
            self.basis = "spatial_modes"
            if array:
                return self.state
            else:
                return self
        else:
            real = np.zeros(self.state.shape)
            imaginary = self.state
            spatial_modes = np.tile(
                np.concatenate((real, imaginary), axis=1), (self.n, 1)
            )
            if array:
                return spatial_modes
            else:
                return self.__class__(
                    **{**vars(self), "state": spatial_modes, "basis": "spatial_modes"}
                )


class RelativeEquilibriumOrbitKS(RelativeOrbitKS):
    @staticmethod
    def minimal_shape():
        """
        The smallest possible compatible discretization

        Returns
        -------
        tuple of int :
            The default array shape when dimensions are not specified.

        Notes
        -----
        Often symmetry constraints reduce the dimensionality; if too small this reduction may leave the state empty,
        used for aspect ratio correction and possibly other gluing applications.

        If parity is important, than that must be built into the definition.

        """

        return 1, 4

    def dt(self, order=1, array=False, **kwargs):
        """
        A time derivative of the current state.

        Parameters
        ----------
        order :int
            The order of the derivative.

        array : bool
            Whether to return np.ndarray or Orbit

        Returns
        ----------
        RelativeEquilibriumOrbitKS :
            The class instance whose state is the time derivative in
            the spatiotemporal mode basis (i.e. zero in comoving frame).

        """
        if self.frame == "comoving":
            if array:
                return np.zeros(self.state.shape)
            else:
                return self.__class__(
                    **{**vars(self), "state": np.zeros(self.shape), "basis": "modes"}
                )
        else:
            raise ValueError(
                "Attempting to compute time derivative of "
                + str(self)
                + " in physical reference frame."
                + "If this is truly desired, convert to RelativeOrbitKS first."
            )

    def shapes(self):
        """
        State array shapes in different bases. See core.py for details.

        """
        return (self.n, self.m), (self.n, self.m - 2), (1, self.m - 2)

    def preprocess(self):
        """
        Check whether the orbit converged to an equilibrium or close-to-zero solution

        """
        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        orbit_with_inverted_shift = self.copy()
        orbit_with_inverted_shift.s = -self.s
        cost_imported_S = self.cost()
        cost_negated_S = orbit_with_inverted_shift.cost()
        if cost_imported_S > cost_negated_S:
            orbit_ = orbit_with_inverted_shift
        else:
            orbit_ = self.copy()

        # Take the L_2 norm of the field, if uniformly close to zero, the magnitude will be very small.
        field_orbit = orbit_.transform(to="field")
        zero_check = field_orbit.norm()
        if zero_check < 10 ** -5:
            return RelativeEquilibriumOrbitKS(
                **{
                    **vars(self),
                    "state": np.zeros(self.discretization),
                    "basis": "field",
                }
            ).transform(to=self.basis)
        else:
            return orbit_

    def to_fundamental_domain(self):
        return self.change_reference_frame(frame="physical")

    def from_fundamental_domain(self):
        """
        For compatibility purposes with plotting and other utilities

        """
        return self.change_reference_frame(frame="comoving")

    def jacobian(self, **kwargs):
        """
        Jacobian matrix evaluated at the current state.

        kwargs :
            Unused, included to match signature in Orbit class.

        Returns
        -------
        jac_ : 2-d ndarray
            Jacobian matrix whose columns are derivatives with respect to all unconstrained state variables;
            including periods. Has dimensions dependent on number of spatiotemporal modes and free parameters,
            (self.shapes()[-1].size, self.shapes()[-1].size + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(self.constraints)

        Notes
        -----
        Original implementation was pretty, but very inefficient. This now computes the Jacobian matrix,
        minimizing the amount of allocated memory by overwriting and performing matrix-free implementations of
        Fourier transform matrix operations. Computes the Jacobian
        $J = D_t + D_xx + D_xxxx + F_t D_x F_x Diag(u) F_x^{-1} F_t^{-1}$ in the following steps:

        """
        assert self.basis == "modes"
        field_size, smode_size, mode_size = (np.prod(shp) for shp in self.shapes())
        # Begin with nonlinear term. Apply matrix operators in matrix-free fashion. begin with diag(u)
        J = np.diag(self.transform(to="field", array=True).ravel()).reshape(-1, self.m)
        # By creatively reshaping J, can apply FFTs to 3-d tensor.
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        ).transform(to="spatial_modes", array=True, inplace=True)
        J = J.reshape(field_size, smode_size).T.reshape(-1, self.m)

        # After transforming the columns, transpose and transform again to get F_x Diag(u) F_x^{-1}
        J = self.__class__(
            **{**vars(self), "state": J, "basis": "field", "discretization": J.shape}
        )

        # Reshape back into a matrix, and then into another 3-d tensor after transposing back, to take derivative.
        J = J.transform(to="spatial_modes", array=True, inplace=True).reshape(
            smode_size, smode_size
        )
        J = J.T.reshape(-1, self.m - 2)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "basis": "spatial_modes",
                "discretization": (J.shape[0], self.m),
            }
        )
        J = J.dx(array=True, computation_basis="spatial_modes")

        # At this point J represents D_x F_x Diag(u) F_x^{-1}; reshape into 3-d tensor again and apply
        # time transforms
        J = J.reshape(-1, smode_size).T.reshape(self.n, self.m - 2, -1)
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        # reshape, transpose and transform.
        J = (
            J.reshape((*self.shapes()[2], -1))
            .reshape(-1, smode_size)
            .T.reshape(self.n, self.m - 2, -1)
        )
        J = self.__class__(
            **{
                **vars(self),
                "state": J,
                "discretization": (*self.discretization, J.shape[-1]),
                "basis": "spatial_modes",
            }
        ).transform(to="modes", array=True, inplace=True)
        J = J.reshape((*self.shapes()[2], -1)).reshape(J.shape[-1], J.shape[-1]).T

        # Produce the linear term; spatial derivatives are diagonal; time is more complicated due to SO(2) operator.
        e = np.ones(self.shapes()[2])
        dx2 = self.__class__(state=e, basis="modes", parameters=self.parameters).dx(
            order=2
        )
        J[np.diag_indices(J.shape[0])] += dx2.state.ravel() + dx2.state.ravel() ** 2
        e = e.ravel()
        # For time, get the correct frequency by taking the derivative of individual elements; swapping of real
        # and imaginary components handled by .dt()
        for i in range(J.shape[0]):
            e *= 0
            e[i] = 1.0
            J[:, i] += (-self.s / self.t) * self.__class__(
                state=e.reshape(self.shape), basis="modes", parameters=self.parameters
            ).dx().state.ravel()

        J = self._jacobian_parameter_derivatives_concat(J)
        return J

    @classmethod
    def dimension_based_discretization(cls, dimensions, **kwargs):
        """
        Subclassed method for equilibria.

        """
        kwargs.setdefault("discretization", (1, None))
        n, m = super().dimension_based_discretization(dimensions, **kwargs)
        return n, m

    def _pad(self, size, axis=0):
        """
        Overwrite of parent method

        Notes
        -----
        If starting and ending in the spatiotemporal modes basis, this will only create an instance with a different
        value of time dimensionality attribute 'N'.

        """
        assert (
            self.frame == "comoving"
        ), "Transform to comoving frame before padding modes"

        if np.mod(size, 2):
            raise ValueError(
                "New discretization size must be an even number, preferably a power of 2"
            )
        else:
            if axis == 0:
                # Not technically zero-padding, just copying. Just can't be in temporal mode basis
                # because it is designed to only represent the zeroth modes.
                spatial_modes = self.transform(to="spatial_modes")
                padded_spatial_modes = np.tile(
                    spatial_modes.state[None, -1, :], (size, 1)
                )
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_spatial_modes,
                        "basis": "spatial_modes",
                        "discretization": (size, self.m),
                    }
                ).transform(to=self.basis)
            else:
                modes = self.transform(to="modes")
                padding = (size - modes.m) // 2
                padding_tuple = ((0, 0), (padding, padding))
                padded_modes = np.concatenate(
                    (
                        modes.state[:, : -modes.m // 2 - 1],
                        np.pad(modes.state[:, -modes.m // 2 - 1 :], padding_tuple),
                    ),
                    axis=1,
                )
                padded_modes *= np.sqrt(size / modes.m)
                return self.__class__(
                    **{
                        **vars(self),
                        "state": padded_modes,
                        "basis": "modes",
                        "discretization": (self.n, size),
                    }
                ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Subclassed method to handle RelativeEquilibriumOrbitKS mode's shape.

        """
        assert (
            self.frame == "comoving"
        ), "Transform to comoving frame before truncating modes"
        if axis == 0:
            truncated_spatial_modes = self.transform(to="spatial_modes").state[
                -size:, :
            ]
            return self.__class__(
                **{
                    **vars(self),
                    "state": truncated_spatial_modes,
                    "basis": "spatial_modes",
                    "discretization": (size, self.m),
                }
            ).transform(to=self.basis)
        else:
            truncate_number = int(size // 2) - 1
            # Split into real and imaginary components, truncate separately.
            spatial_modes = self.transform(to="spatial_modes")
            first_half = spatial_modes.state[:, :truncate_number]
            second_half = spatial_modes.state[
                :,
                -spatial_modes.m // 2 - 1 : -spatial_modes.m // 2 - 1 + truncate_number,
            ]
            truncated_spatial_modes = np.sqrt(size / spatial_modes.m) * np.concatenate(
                (first_half, second_half), axis=1
            )
            return self.__class__(
                **{
                    **vars(self),
                    "state": truncated_spatial_modes,
                    "basis": "spatial_modes",
                    "discretization": (self.n, size),
                }
            ).transform(to=self.basis)

    def _eqn_linear_component(self, array=False):
        """
        Linear component of the KSE

        Parameters
        ----------
        array : bool
            Whether to return ndarray or not.

        Returns
        -------
        ndarray or class instance.

        """
        return (
            self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
            - (self.s / self.t) * self.dx(array=array)
        )

    def _rmatvec_linear_component(self, array=False):
        """
        Linear component of the adjoint Jacobian-vector product

        Parameters
        ----------
        array : bool
            Whether or not to return ndarray

        Returns
        -------
        Orbit or ndarray.

        """
        return (
            self.dx(order=2, array=array)
            + self.dx(order=4, array=array)
            + (self.s / self.t) * self.dx(array=array)
        )

    def _parse_state(self, state, basis, **kwargs):
        if isinstance(state, np.ndarray):
            if len(state.shape) != 2:
                raise ValueError('"state" array must be two-dimensional')
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(0, 0)

        if self.size > 0:
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            elif basis == "modes":
                self.state = self.state[None, 0, :]
                # For uniformity with _populate_state, use 'shape' instead of 'n' for kwarg
                n = kwargs.get("discretization", (None, None))[0] or 1
                m = self.shape[1] + 2
            elif basis == "field":
                n, m = self.shape
            elif basis == "spatial_modes":
                n, m = self.shape[0], self.shape[1] + 2
            else:
                raise ValueError(
                    'basis not recognized; must equal "field" or "spatial_modes", or "modes"'
                )
            # To allow for multiple time point fields and spatial modes, for plotting purposes.
            if n < self.minimal_shape()[0] or m < self.minimal_shape()[1]:
                warn_str = "\nminimum discretization requirements not met; methods may not work as intended."
                warnings.warn(warn_str, RuntimeWarning)
            self.discretization = n, m
            self.basis = basis
        else:
            self.discretization = None
            self.basis = None

    def _inv_time_transform_matrix(self):
        """
        Overwrite of parent method

        Notes
        -----
        Originally this transform just selected the antisymmetric spatial modes (imaginary component),
        but in order to be consistent with all other time transforms, I will actually apply the normalization
        constant associated with a forward in time transformation. The reason for this is for comparison
        of states between different subclasses.

        """
        return np.tile(np.eye(self.m - 2), (self.n, 1))

    def _time_transform_matrix(self):
        """
        Overwrite of parent method

        Notes
        -----
        Input state is [N, M-2] dimensional array which is to be sliced to return only the last row.
        N * (M-2) repeats of modes coming in, M-2 coming out, so M-2 rows.

        """

        dft_mat = np.tile(np.eye(self.m - 2), (1, self.n))
        dft_mat[:, self.m - 2 :] = 0
        return dft_mat

    def _time_transform(self, array=False, inplace=False):
        """
        Overwrite of parent method

        Notes
        -----
        Taking the RFFT, with orthogonal normalization, of a constant time series defined on N points is equivalent
        to multiplying the constant value by sqrt(N). This is because the transform sums over N repeats of the same
        value and then divides by 1/sqrt(N). i.e. (N * C)/sqrt(N) = sqrt(N) * C. Therefore we can save some time by
        just doing this without calling the rfft function.

        This always returns a single instant in time, for solving purposes; if N != 1 then computations will work
        but will be much less efficient.

        Originally wanted to include normalization but it just screws things up given how the modes are truncated.

        """
        # Select the nonzero (imaginary) components of modes and transform in time (w.r.t. axis=0).
        if inplace:
            self.state = self.state[np.newaxis, 0, :]
            self.basis = "modes"
            if array:
                return self.state
            else:
                return self
        else:
            modes = self.state[np.newaxis, 0, :]
            if array:
                return modes
            else:
                return self.__class__(
                    **{**vars(self), "state": modes, "basis": "modes"}
                )

    def _inv_time_transform(self, array=False, inplace=False):
        """
        Overwrite of parent method

        Notes
        -----
        Taking the IRFFT, with orthogonal normalization is equivalent to dividing by the normalization constant; because
        there would only be

        """
        spatial_modes = np.tile(self.state[None, 0, :], (self.n, 1))
        if array:
            return spatial_modes
        else:
            return self.__class__(
                **{**vars(self), "state": spatial_modes, "basis": "spatial_modes"}
            )

    def _jac_lin(self):
        """
        Extension of the OrbitKS method that includes the term for spatial translation symmetry.

        """
        return (
            self._dx_matrix(order=2)
            + self._dx_matrix(order=4)
            + (-self.s / self.t) * self._dx_matrix()
        )

    def _jacobian_parameter_derivatives_concat(self, jac_):
        """
        Concatenate parameter partial derivatives to Jacobian matrix

        Parameters
        ----------
        jac_ : np.ndArray,
            (N-1) * (M-2) dimensional array resultant from taking the derivative of the spatioatemporal mapping
            with respect to Fourier modes.

        Returns
        -------
        Jacobian augmented with parameter partial derivatives as columns. Required to solve for changes to period,
        space period in optimization process. Makes the system rectangular; needs to be solved by least squares type
        methods.

        """
        # If period is not fixed, need to include dF/dt in jacobian matrix
        if not self.constraints["t"]:
            time_period_derivative = (
                (-1.0 / self.t)
                * (-self.s / self.t)
                * self.dx(array=True).reshape(-1, 1)
            )
            jac_ = np.concatenate((jac_, time_period_derivative), axis=1)

        # If spatial period is not fixed, need to include dF/dx in jacobian matrix
        if not self.constraints["x"]:
            self_field = self.transform(to="field")
            spatial_period_derivative = (
                (-2.0 / self.x) * self.dx(order=2, array=True)
                + (-4.0 / self.x) * self.dx(order=4, array=True)
                + (-1.0 / self.x) * self_field._nonlinear(self_field, array=True)
            )
            jac_ = np.concatenate(
                (jac_, spatial_period_derivative.reshape(-1, 1)), axis=1
            )

        if not self.constraints["s"]:
            spatial_shift_derivatives = (-1.0 / self.t) * self.dx(array=True)
            jac_ = np.concatenate(
                (jac_, spatial_shift_derivatives.reshape(-1, 1)), axis=1
            )

        return jac_


def swap_modes(modes, axis=0):
    """
    Function which swaps real, imaginary components of arrays for SO(2) differentiation

    Parameters
    ----------
    modes : np.ndarray
        Array of Fourier modes,
    axis : int
        Whether to swap along time (0) or space(1) dimension
    Notes
    -----
        Both real and imaginary components are stored as real numbers. See class transforms for details.

    """
    if axis == 1:
        m = modes.shape[1] // 2
        t_dim = modes.shape[0]
        swapped_modes = np.concatenate(
            (modes[:, -m:].reshape(t_dim, -1), modes[:, :-m].reshape(t_dim, -1)), axis=1
        )
    else:
        n = (modes.shape[0] + 1) // 2 - 1
        # do not need the special case as .dt() shouldn't be used for either subclass mentioned above.
        swapped_modes = np.concatenate(
            (modes[None, 0, :], modes[-n:, :], modes[1:-n, :]), axis=0
        )
    return swapped_modes


@lru_cache()
def so2_generator(order):
    """
    Powers of the generator of the SO(2) Lie algebra

    Parameters
    ----------
    order : int
        Order of the derivative for which this function is called to produce frequencies for.

    Returns
    -------
    np.ndarray : 2x2 array
        Equal to generator to the n-th power.

    """
    return np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), np.mod(order, 4))


@lru_cache()
def so2_coefficients(order):
    """
    Non-zero elements of the Lie algebra generator to the order-th power

    Parameters
    ----------
    order : int
        Order of the derivative for which this function is called to produce frequencies for.

    Returns
    -------
    np.ndarray :
        (2,) ndarray of correct powers of -1 for differentiation

    """
    return np.sign(1j ** order).real, np.sign((-1j) ** order).real


@lru_cache()
def temporal_frequencies(t, n, order):
    """
    Matrix/rank 2 tensor of temporal mode frequencies

    Parameters
    ----------
    t : float
        Temporal period
    n : int
        Temporal discretization size
    order : int
        The order of the derivative/power of the frequencies desired.

    Returns
    ----------
    dtn_multipliers : np.ndarray
        Array of spatial frequencies in the same shape as modes

    Notes
    -----
    Creates and returns a rank 2 tensor whose elements are the properly ordered temporal frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes.

    """
    # Extra factor of -1 because of time ordering in array.
    # Extra factor of -1 because of time ordering in array.
    w = (-1 * (2 * pi * n / t) * rfftfreq(n)[1:-1]) ** order
    # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
    c1, c2 = np.sign(1j ** order).real, np.sign((-1j) ** order).real
    # The Nyquist frequency is never included, this is how time frequency modes are ordered.
    # Elementwise product of modes with time frequencies is the spectral derivative.
    return np.concatenate(([0], c1 * w, c2 * w))[:, np.newaxis]


@lru_cache()
def spatial_frequencies(x, m, order):
    """
    Matrix/rank 2 tensor of spatial mode frequencies

    Parameters
    ----------
    x : float
        Spatial period
    m : int
        Spatial discretization size
    order : int
        The order of the derivative/power of the frequencies desired.

    Returns
    ----------
    dxn_multipliers : np.ndarray
        Array of spatial frequencies in the same shape as modes

    Notes
    -----
    Creates and returns a rank 2 tensor whose elements are the properly ordered spatial frequencies,
    which is the same shape as the spatiotemporal Fourier mode state. The elementwise product
    with a set of spatiotemporal Fourier modes.

    """

    # Elementwise multiplication of modes with frequencies, this is the derivative.
    q = ((2 * pi * m / x) * rfftfreq(m)[1:-1]) ** order
    # Coefficients which depend on the order of the derivative, see SO(2) generator of rotations for reference.
    c1, c2 = np.sign(1j ** order).real, np.sign((-1j) ** order).real
    # spatial frequency array, reshaped for broadcasting.
    return np.concatenate((c1 * q, c2 * q))[None, :]


@lru_cache()
def dxn_block(x, m, order):
    """
    Block diagonal matrix of spatial frequencies

    Parameters
    ----------
    x : float
        spatial period
    m : int
        spatial discretization size.
    order : int
        Order of the desired derivative.

    Returns
    -------
    np.ndarray :
        Two dimensional block diagonal or diagonal array with SO(2) generator for multiple Fourier modes.
    Notes
    -----
    This is the SO(2) generator for multiple Fourier modes. Only used in explicit construction of matrices.

    """
    qkn = ((2 * pi * m / x) * rfftfreq(m)[None, 1:-1]) ** order
    return np.kron(so2_generator(order), np.diag(qkn.ravel()))


@lru_cache()
def dtn_block(t, n, order):
    """
    Block diagonal matrix of temporal frequencies

    Parameters
    ----------
    t : float
        Temporal period
    n : int
        Temporal discretization size.
    order : int
        Order of the desired derivative.

    Returns
    -------
    np.ndarray :
        Two dimensional block diagonal or diagonal array with SO(2) generator for multiple Fourier modes.
    Notes
    -----
    This is the SO(2) generator for multiple Fourier modes. Only used in explicit construction of matrices.

    """
    wjn = (-(2 * pi * n / t) * rfftfreq(n)[1:-1, None]) ** order
    return np.kron(so2_generator(order), np.diag(wjn.ravel()))
