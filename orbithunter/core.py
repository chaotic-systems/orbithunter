import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.resetwarnings()

__all__ = ['Orbit']

"""
The core class for all orbithunter calculations. The methods listed are the ones used in the other modules. If
full functionality isn't currently desired then I recommend only implementing the methods used in optimize.py,
saving data to disk, and plotting. Of course this is in addition to the dunder methods such as __init__. 

While not listed here explicitly, this package is fundamentally a spectral method based package. So in order
to write the functions rmatvec, matvec, spatiotemporal mapping, one will necessarily have to include
methods for differentiation, basis transformations, etc. I do not include them because different equations
have different dimensions and hence will have different transforms. All transforms should be wrapped by
the method .convert(). The spatiotemporal basis should be labeled as 'modes', the physical field should be
labeled as 'field'. There are no assumptions on what "field" and "modes" actually mean, so the physical state space
need not be an actual field. 

In order for all numerical methods to work, the mandatory methods compute the matrix-vector products rmatvec = J^T * x,
matvec = J * x, and must be able to construct the Jacobian = J. ***NOTE: the matrix vector products SHOULD NOT
explicitly construct the Jacobian matrix.***

"""


class Orbit:
    """ Base class for all equations


    Notes
    -----
    Methods listed here are required to have everything work.
    """

    def __init__(self, state=None, state_type='modes', **kwargs):
        if state is not None:
            self._parse_parameters(**kwargs)
            self._parse_state(state, state_type, **kwargs)
        else:
            self._parse_parameters(nonzero_parameters=True, **kwargs)
            self._random_initial_condition(**kwargs).convert(to=state_type, inplace=True)

    def __radd__(self, other):
        return None

    def __sub__(self, other):
        return None

    def __rsub__(self, other):
        return None

    def __mul__(self, num):
        return None

    def __rmul__(self, num):
        return None

    def __truediv__(self, num):
        return None

    def __floordiv__(self, num):
        return None

    def __pow__(self, power):
        return None

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __repr__(self):
        return None

    def __getattr__(self, attr):
        return None

    def convert(self, inplace=False, to=None, **kwargs):
        """ Method that handles all basis transformations.

        Parameters
        ----------
        inplace : bool
        Whether or not to return a new Orbit instance, or overwrite self.
        to : str
        The basis to transform into. i.e. 'field', 's_modes', 'modes', typically.
        kwargs :
        Included to allow identical signatures in subclasses.

        Returns
        -------

        """
        return None

    def spatiotemporal_mapping(self, *args, **kwargs):
        """ The Kuramoto-Sivashinsky equation evaluated at the current state.

        kwargs :
        preconditioning : bool
        Apply custom preconditioner, only used in numerical methods.

        Returns
        -------
        OrbitKS :
            OrbitKS whose state is the spatiotamporal fourier modes resulting from the calculation of the K-S equation:
            OrbitKS.state = u_t + u_xx + u_xxxx + 1/2 (u^2)_x
        :return:
        """
        return None

    def residual(self, apply_mapping=True):
        """ The value of the cost function

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 the squared L_2 norm of the spatiotemporal mapping,
            R = 1/2 ||F||^2. The current form generalizes to any equation.
        """
        if apply_mapping:
            v = self.convert(to='modes').spatiotemporal_mapping().state.ravel()
            return 0.5 * v.dot(v)
        else:
            u = self.state.ravel()
            return 0.5 * u.dot(u)

    def matvec(self, other, **kwargs):
        """ Matrix-vector product of a vector with the Jacobian of the current state.
        """
        return None

    def rmatvec(self, other, **kwargs):
        """ Matrix-vector product with the adjoint of the Jacobian with a state

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
            OrbitKS with values representative of the adjoint-vector product

        Notes
        -----
        The adjoint vector product in this case is defined as J^T * v,  where J is the jacobian matrix. Equivalent to
        evaluation of -v_t + v_xx + v_xxxx  - (u .* v_x). In regards to preconditioning (which is very useful
        for certain numerical methods, right preconditioning and left preconditioning switch meanings when the
        jacobian is transposed. i.e. Right preconditioning of the Jacobian can include preconditioning of the state
        parameters (which in this case are usually incremental corrections dT, dL, dS);
        this corresponds to LEFT preconditioning of the adjoint.

        """

        return None

    def state_vector(self):
        """ Vector representation of orbit. Includes state + parameters. """
        return None

    def from_numpy_array(self, state_array, **kwargs):
        """ Utility to convert from numpy array to orbithunter format for scipy wrappers.
        :param orbit:
        :param state_array:
        :param parameter_constraints:
        :return:

        Notes
        -----
        Takes a ndarray of the form of state_vector method and returns Orbit instance.
        """
        return None

    def increment(self, other, step_size=1):
        """ Incrementally add Orbit instances together

        Parameters
        ----------
        other : OrbitKS
            Represents the values to increment by.
        step_size : float
            Multiplicative factor which decides the step length of the correction.

        Returns
        -------
        Orbit
            New instance which results from adding an optimization correction to self.

        Notes
        -----
        This is used primarily in optimization methods, e.g. adding a gradient descent step using class instances
        instead of simply arrays.
        """
        return None

    def mode_padding(self, size, axis=0):
        """ Increase the size of the discretization via zero-padding collocation basis.

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

        Notes
        -----
        Function for increasing the discretization size in dimension corresponding to 'axis'.

        """

        return self

    def mode_truncation(self, size, axis=0):
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
        return self

    def jacobian(self, **kwargs):
        """ Jacobian matrix evaluated at the current state.
        Parameters
        ----------
        parameter_constraints : tuple
            Determines whether to include period and spatial period
            as variables.
        Returns
        -------
        jac_ : matrix ((N-1)*(M-2), (N-1)*(M-2) + n_params)
            Jacobian matrix of the KSe where n_params = 2 - sum(parameter_constraints)
        """
        return None

    def norm(self, order=None):
        """ Norm of spatiotemporal state via numpy.linalg.norm

        Example
        -------
        L_2 distance between two states
        >>> (self - other).norm()
        """
        return np.linalg.norm(self.state.ravel(), ord=order)

    @property
    def parameters(self):
        """ Pass all parameters as one collection instead of individually.


        Returns
        -------
        parameter_dict : dict
        A dictionary which includes all parameters relevant to all other methods.

        Notes
        -----
        The only thing that matters is that for a field with D dimensions, the first D values of the parameter
        dict should correspond to the respective dimension. For example, the Kuramoto-sivashinsky equation
        as (1+1) dimensional spacetime. These dimensions are labeled by T, L. Therefore the parameter dict's
        first two entries should be {'T': T, 'L': L , ...}. Time is always assumed to be the first axis (axis=0 in
        NumPy parlance).

        This acts as a bundling of different orbit attributes. Helps with keeping code succinct, passing
        values to functions with @lru_cache decorator, generalization of functions to multiple equations.
        """
        parameter_dict = {}
        return parameter_dict

    @classmethod
    def glue_parameters(cls, parameter_dict_with_zipped_values, axis=0):
        """ Class method for handling parameters in gluing

        Parameters
        ----------
        parameter_dict_with_zipped_values
        axis

        Returns
        -------

        Notes
        -----
        Only required if gluing module is to be used. In the gluing process, we must have a rule for how to combine
        the fields and how to approximate the dimensions of the newly glued field. This method approximates
        with simple summation and averaging. Should accomodate any dimension via zipping parameter dict values
        in the correct manner.

        """
        new_parameter_dict = {}
        return new_parameter_dict

    def plot(self, show=True, save=False, padding=True, fundamental_domain=True, **kwargs):
        """ Custom plotting method using matplotlib
        """
        return None

    def precondition(self, parameters, **kwargs):
        """

        Parameters
        ----------
        parameters : dict
        Dictionary containing all relevant orbit parameters.
        kwargs

        Returns
        -------

        Notes
        -----
        If no preconditioning is desired then pass preconditioning=False to numerical methods, or simply return
        self as is written here.

        """
        return self

    def preconditioner(self, parameters, **kwargs):
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

        Notes
        -----
        if nothing desired then return identity matrix whose main diagonal is the same size as state-vector
        (Could also need another identity that excludes parameter dimension, depending on whether right or left
        preconditioning is chosen).
        """
        # Preconditioner is the inverse of the absolute value of the linear spatial derivative operators.
        return np.eye(self.state_vector().size)

    def rescale(self, magnitude, inplace=False):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to rescale by.

        Notes
        -----
        This rescales the physical field such that the absolute value of the max/min takes on a new value
        of magnitude
        """
        return self

    @property
    def shape(self):
        """ Convenience to not have to type '.state.shape' all the time in notebooks"""
        return self.state.shape

    def to_h5(self, filename=None, directory='local', verbose=False):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str
            Name for the save file
        directory :
            Location to save at
        verbose : If true, prints save messages to std out
        """
        return None

    def parameter_dependent_filename(self, extension='.h5', decimals=3):
        """

        Parameters
        ----------
        extension : str
        The data format used in whichever saving method is being applied.
        decimals : int
        How many decimals to write in the filename
        Returns
        -------

        """
        save_filename = 'default'+extension
        return save_filename

    def verify_integrity(self):
        """ Check the status of a solution, whether or not it converged to the correct orbit type. """
        return None

    def _parse_state(self, state, state_type, **kwargs):
        """ Determine state shape parameters based on state array and the basis it is in.

        Parameters
        ----------
        state : ndarray
        Numpy array containing state information, can have any number of dimensions.
        state_type :
        The basis that the array 'state' is assumed to be in.
        kwargs

        Returns
        -------

        """
        self.state = state
        self.state_type = state_type
        return None

    def _parse_parameters(self, T=0., L=0., **kwargs):
        """ Determine the dimensionality and symmetry parameters.
        """
        return None

    def _random_initial_condition(self, **kwargs):
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
        These are the initial condition generators that I find the most useful. If a different method is
        desired, simply pass the array as 'state' variable to __init__.
        """
        return None

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
        return None

    def copy(self):
        """ Returns a shallow copy of an orbit instance.

        Returns
        -------

        """
        return None