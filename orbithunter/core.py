from json import dumps
import os
import h5py
import numpy as np

__all__ = ['Orbit', 'convert_class']

"""
The core class for all orbithunter calculations. The methods listed are the ones used in the other modules. If
full functionality isn't currently desired then I recommend only implementing the methods used in optimize.py,
saving data to disk, and plotting. Of course this is in addition to the dunder methods such as __init__. 

While not listed here explicitly, this package is fundamentally a (pseudo)spectral method based package; while it is not
technically required to use a spectral method, not doing so may result in awkwardly named attributes. For example,
the reshape method in spectral space essentially interpolates via zero-padding and truncation. Therefore, other
interpolation methods would be forced to use _pad and _truncate, unless they specifically overwrite the reshape
method itself. Another example, The spatiotemporal basis should be labeled as 'modes', the "physical" basis, 
'field'. There are no assumptions on what "field" and "modes" actually mean, so the physical state space
need not be an actual field.
 
In order to write the functions rmatvec, matvec, spatiotemporal mapping, one will necessarily have to include
methods for differentiation, basis transformations, etc. I do not include them because different equations
have different dimensions and hence will have different transforms. All transforms should be wrapped by
the method .transform(), such that transforming to another basis can be accessed by statements such as 
.transform(to='modes'). NOTE: if the orbit is in the same basis as that specified by 'to', the ORIGINAL orbit;
NOT a copy be returned. The reason why this is allowed is because transform then has the dual
functionality of ensuring an orbit is a certain basis for a calculation, while maintaining flexibility. There
are instances where it is required to be in the spatiotemporal basis, to avoid unanticipated transforms, however,
there are a number of operations which require the spatiotemporal basis, but often one wants to remain in the
physical basis. Therefore, to avoid verbosity, the user can specify the function self.method() instead of
self.transform(to='required_basis').method().transform(to='original_basis').  

In order for all numerical methods to work, the mandatory methods compute the matrix-vector products rmatvec = J^T * x,
matvec = J * x, and must be able to construct the Jacobian = J. ***NOTE: the matrix vector products SHOULD NOT
explicitly construct the Jacobian matrix. In the context of DAE's there is typically no need to write these
with finite difference approximations of time evolved Jacobians. (Of course if the DAEs are defined
in terms of finite differences, so should the Jacobian).***

"""


class Orbit:
    def __init__(self, state=None, basis='field', parameters=(0., 0., 0., 0.), **kwargs):
        """ Base/Template class for orbits

        Parameters
        ----------
        state : ndarray(dtype=float, ndim=4) or None
            If an array, it should contain the state values pertaining to the 'basis' label. E.g. for Navier-stokes
            this would be all spatiotemporal velocity field values,
        basis : str
            Which basis the array 'state' is currently in. Takes values
            'field', 's_modes', 'modes'.
        parameters : tuple
            Time period, spatial period, spatial shift (unused but kept for uniformity, in case of conversion between
            OrbitKS and RelativeOrbitKS).
        **kwargs :
            Extra arguments for _parse_parameters and random_state
                See the description of the aforementioned method.

        Notes
        -----
        Methods listed here are required to have everything work.

        """
        if state is not None:
            self._parse_parameters(parameters, nonzero_parameters=kwargs.pop('nonzero_parameters', False), **kwargs)
            self._parse_state(state, basis, **kwargs)
        else:
            # If the state is not passed, then it will be randomly generated. This will require referencing the
            # dimensions, expected to be nonzero to define the collocation grid. Therefore, it is required to
            # either provide
            self._parse_parameters(parameters, nonzero_parameters=kwargs.pop('nonzero_parameters', True), **kwargs)
            # Pass the newly generated parameter values, there are the originals if they were not 0's.
            self.random_state(self.parameters, **kwargs)

    def __add__(self, other):
        """ Addition of Orbit states

        Parameters
        ----------
        other : Orbit
        Should have same class as self. Should be in same basis as self.
        Notes
        -----
        Add two spatiotemporal states
        """
        return self.__class__(state=(self.state + other.state), basis=self.basis, parameters=self.parameters)

    def __radd__(self, other):
        """ Addition of Orbit states

        Parameters
        ----------
        other : Orbit
        Should have same class as self. Should be in same basis as self.
        Notes
        -----
        Adding two spatiotemporal velocity fields u(t, x) + v(t, x)

        Notes
        -----
        This is the same as __add__ by Python makes the distinction between where the operator is, i.e. x + vs. + x.
        """
        return self.__class__(state=(self.state + other.state), basis=self.basis, parameters=self.parameters)

    def __sub__(self, other):
        """ Subtraction of orbit states

        Parameters
        ----------
        other : Orbit
        Should have same class as self. Should be in same basis as self.
        Notes
        -----
        Subtraction of two spatiotemporal states self - other
        """
        return self.__class__(state=(self.state-other.state), basis=self.basis, parameters=self.parameters)

    def __rsub__(self, other):
        """ Subtraction of orbit states

        Parameters
        ----------
        other : Orbit
        Should have same class as self. Should be in same basis as self.
        Notes
        -----
        Subtraction of two spatiotemporal states other - self
        """
        return self.__class__(state=(other.state - self.state), basis=self.basis, parameters=self.parameters)

    def __mul__(self, num):
        """ Scalar multiplication of state values

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        """
        return self.__class__(state=np.multiply(num, self.state), basis=self.basis, parameters=self.parameters)

    def __rmul__(self, num):
        """ Scalar multiplication of state values

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        """
        return self.__class__(state=np.multiply(num, self.state), basis=self.basis, parameters=self.parameters)

    def __truediv__(self, num):
        """ Scalar division of state values

        Parameters
        ----------
        num : float
            Scalar value to divide by
        """
        return self.__class__(state=np.divide(self.state, num), basis=self.basis, parameters=self.parameters)

    def __floordiv__(self, num):
        """ Scalar multiplication

        Parameters
        ----------
        num : float
            Scalar value to division by.

        Notes
        -----
        Returns largest integer smaller or equal to the division of the inputs, of the state. This isn't useful
        but I'm including it because it's a fairly common binary operation and might be useful in some circumstances.
        """
        return self.__class__(state=np.floor_divide(self.state, num), basis=self.basis, parameters=self.parameters)

    def __pow__(self, power):
        """ Exponentiate a state

        Parameters
        ----------
        power : float
            Exponent
        """
        return self.__class__(state=self.state**power, basis=self.basis, parameters=self.parameters)

    def __str__(self):
        """ String name
        Returns
        -------
        str :
            Of the form 'Orbit'
        """
        return self.__class__.__name__

    def __repr__(self):
        # alias to save space
        dict_ = {'basis': self.basis,
                 'parameters': tuple(str(np.round(p, 4)) for p in self.parameters),
                 'shape': tuple(str(d) for d in self.shapes()[0])}
        # convert the dictionary to a string via json.dumps
        dictstr = dumps(dict_)
        return self.__class__.__name__ + '(' + dictstr + ')'

    def cost_function_gradient(self, dae, **kwargs):
        """ Derivative of 1/2 |F|^2

        Parameters
        ----------
        dae : Orbit
            Orbit instance whose state equals DAE evaluated with respect to current state, i.e. F(v)
        kwargs

        Returns
        -------
        gradient :
            Orbit instance whose state contains (dF/dv)^T * F ; (adjoint Jacobian * DAE)

        Notes
        -----
        Withing optimization routines, the DAE orbit is used for other calculations and hence should not be
        recalculated
        """
        preconditioning = kwargs.get('preconditioning', False)
        if preconditioning:
            gradient = (self.rmatvec(dae, **kwargs)
                        ).precondition(self.preconditioning_parameters, **kwargs)
        else:
            gradient = self.rmatvec(dae, **kwargs)
        return gradient

    def preconditioning_parameters(self, **kwargs):
        return self.parameters

    def reshape(self, *new_shape, **kwargs):
        """

        Parameters
        ----------
        new_shape : tuple of ints or None
        kwargs

        Returns
        -------

        """
        placeholder_orbit = self.transform(to='field').copy().transform(to='modes')

        if len(new_shape) == 1:
            # if passed as tuple, .reshape((a,b)), then need to unpack ((a, b)) into (a, b)
            new_shape = tuple(*new_shape)
        elif not new_shape:
            # if nothing passed, then new_shape == () which evaluates to false.
            # The default behavior for this will be to modify the current discretization
            # to a `parameter based discretization'. If this is not desired then simply do not call reshape.
            new_shape = self.parameter_based_discretization(self.parameters, **kwargs)

        if self.shapes()[0] == new_shape:
            # to avoid unintended overwrites, return a copy.
            return self.copy()
        else:
            for i, d in enumerate(new_shape):
                if d < self.shapes()[0][i]:
                    placeholder_orbit = placeholder_orbit._truncate(d, axis=i)
                elif d > self.shapes()[0][i]:
                    placeholder_orbit = placeholder_orbit._pad(d, axis=i)
                else:
                    pass
            return placeholder_orbit.transform(to=self.basis)

    def transform(self, to='field'):
        """ Method that handles all basis transformations.

        Parameters
        ----------
        to : str
        The basis to transform into. If already in said basis, returns self

        Returns
        -------

        """
        return self

    def dae(self, *args, **kwargs):
        """ The governing equations evaluated using the current state.

        Returns
        -------
        Orbit :
            Orbit instance whose state equals evaluation of governing equation.

        Notes
        -----
        If self.dae().state = 0. at every point (within some numerical tolerance), then 'self' constitutes
        a solution to the governing equation. Of course there is no equation for this class, so zeros are returned.
        """
        return self.__class__(state=np.zeros(self.shape), basis=self.basis, parameters=self.parameters)

    def residual(self, dae=True):
        """ The value of the cost function

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 the squared L_2 norm of the spatiotemporal mapping,
            R = 1/2 ||F||^2. The current form generalizes to any equation.
        """
        if dae:
            v = self.transform(to='modes').dae().state.ravel()
        else:
            v = self.state.ravel()

        return 0.5 * v.dot(v)

    def matvec(self, other, **kwargs):
        """ Matrix-vector product of a vector with the Jacobian of the current state.
        """
        return None

    def rmatvec(self, other, **kwargs):
        """ Matrix-vector product of a vector with the adjoint of the Jacobian of the current state.

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.
        Returns
        -------
        orbit_rmatvec :
            OrbitKS with values representative of the adjoint-vector product

        Notes
        -----
        The adjoint vector product in this case is defined as J^T * v,  where J is the jacobian matrix.
        """

        return None

    def state_vector(self):
        """ Vector representation of orbit"""
        return np.concatenate((self.state.ravel(), self.parameters), axis=0)

    def from_numpy_array(self, state_vector, **kwargs):
        """ Utility to convert from numpy array to orbithunter format for scipy wrappers.

        Parameters
        ----------
        state_vector : ndarray
            State vector containing state values (modes, typically) and parameters or optimization corrections thereof.

        kwargs :
            parameters : tuple
                If parameters from another Orbit instance are to overwrite the values within the state_vector
            parameter_constraints : dict
                constraint dictionary, keys are parameter_labels, values are bools
            OrbitKS or subclass kwargs : dict
                If special kwargs are required/desired for Orbit instantiation.
        Returns
        -------
        state_orbit : Orbit instance
            Orbit instance whose state and parameters are extracted from the input state_vector.

        Notes
        -----
        Important: If parameters are passed as a keyword argument, they are appended to the numpy array,
        'state_array', via concatenation. The common usage of this function is to take the output of SciPy
        optimization functions (numpy arrays) and store them in Orbit instances.

        This function assumes that the instance calling it is in the "spatiotemporal" basis; the basis in which
        the optimization occurs. This is why no additional specification for size and shape and basis is required.
        The spatiotemporal basis is allowed to be named anything, hence usage of self.basis.
        """
        params_list = list(kwargs.pop('parameters', state_vector.ravel()[self.size:].tolist()))
        parameters = tuple(params_list.pop(0) if not p and params_list else 0 for p in self.constraints.values())
        return self.__class__(state=np.reshape(state_vector.ravel()[:self.size], self.shape), basis=self.basis,
                              parameters=parameters, **kwargs)

    def increment(self, other, step_size=1, **kwargs):
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
        incremented_params = tuple(self_param + step_size * other_param for self_param, other_param
                                   in zip(self.parameters, other.parameters))
        return self.__class__(state=self.state+step_size*other.state, basis=self.basis,
                              parameters=incremented_params, **kwargs)

    def _pad(self, size, axis=0):
        """ Increase the size of the discretization along an axis.

        Parameters
        ----------
        size : int
            The new size of the discretization, must be an even integer
            larger than the current size of the discretization.
        axis : int
            Axis to pad along per numpy conventions.

        Returns
        -------
        OrbitKS :
            OrbitKS instance with larger discretization.

        Notes
        -----
        Function for increasing the discretization size in dimension corresponding to 'axis'.

        """

        return self

    def _truncate(self, size, axis=0):
        """ Decrease the size of the discretization along an axis

        Parameters
        -----------
        size : int
            The new size of the discretization, must be an even integer
            smaller than the current size of the discretization.
        axis : str
            Axis to truncate along per numpy conventions.

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

        Returns
        -------
        jac_ : matrix
        2-d numpy array equalling the Jacobian matrix of the governing equations evaluated at current state.
        """
        return np.zeros([self.size, self.state_vector().size])

    def norm(self, order=None):
        """ Norm of spatiotemporal state via numpy.linalg.norm
        """
        return np.linalg.norm(self.state.ravel(), ord=order)

    @property
    def shape(self):
        """ Current state's shape

        Notes
        -----
        Just a convenience to be able to write self.shape
        """
        return self.state.shape

    def shapes(self):
        """ The different array shapes based on discretization parameters.

        Returns
        -------
        tuple :
            tuple of three tuples corresponding to the shape of the 'state' array in the field, s_modes, modes bases.

        Notes
        -----
        This is a convenience function for operations which require the shape of the state array in a different basis.
        These shapes are defined by the transforms, essentially, but it is wasteful to transform simply for the shape.
        The tuple returned should be field basis is the 0th element, and the current basis
        is the last element. These may refer to the same element; only need indices 0 and -1 to be well defined for the
        general case.

        """
        return (self.state.shape,)

    @property
    def size(self):
        """ Current state's dimensionality

        Notes
        -----
        Just a convenience to be able to write self.size
        """
        return self.state.size

    @property
    def parameters(self):
        """ Parameters required to specify a solution

        Notes
        -----
        In this setting these will be dimensions of spatiotemporal tiles and any additional equation parameters.

        """
        return 0., 0., 0., 0.

    def dimensions(self):
        """ Continuous tile dimensions

        Returns
        -------
        tuple :
            Tuple of dimensions, typically this will take the form (T, L_x, L_y, L_z) for (3+1)-D spacetime
        """
        return 0., 0., 0., 0.

    @staticmethod
    def parameter_labels():
        """ Strings to use to label dimensions/periods
        """
        return 'T', 'Lx', 'Ly', 'Lz'

    @staticmethod
    def dimension_labels():
        """ Strings to use to label dimensions/periods
        """
        return 'T', 'Lx', 'Ly', 'Lz'

    @classmethod
    def glue_parameters(cls, tuple_of_zipped_dimensions, glue_shape=(1, 1, 1, 1)):
        """ Class method for handling parameters in gluing

        Parameters
        ----------
        tuple_of_zipped_dimensions : tuple of tuples

        glue_shape : tuple of ints
            The shape of the gluing being performed i.e. for a 2x2 orbit grid glue_shape would equal (2,2).

        Returns
        -------
        glued_parameters : tuple
            tuple of parameters same dimension and type as self.parameters

        """
        return tuple(glue_shape[i] * p[p > 0.].mean() for i, p in enumerate(np.array(ptuples) for ptuples
                                                                            in tuple_of_zipped_dimensions))

    def plot(self, show=True, save=False, padding=True, fundamental_domain=True, **kwargs):
        """ Signature for plotting method using matplotlib
        """
        return None

    def rescale(self, magnitude=3., method='absolute'):
        """ Scalar multiplication

        Notes
        -----
        This rescales the physical field such that the absolute value of the max/min takes on a new value
        of magnitude
        """
        field = self.transform(to='field').state
        if method == 'absolute':
            rescaled_field = ((magnitude * field) / np.max(np.abs(field.ravel())))
        elif method == 'power':
            rescaled_field = np.sign(field) * np.abs(field)**magnitude
        else:
            raise ValueError('Unrecognizable method.')
        return self.__class__(state=rescaled_field, basis='field',
                              parameters=self.parameters).transform(to=self.basis)

    def to_h5(self, filename=None, directory='local', verbose=False, include_residual=False):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str
            Name for the save file
        directory :
            Location to save at
        h5pymode : str
            Mode with which to open the file. Default is a, read/write if exists, create otherwise,
            other modes ['r', 'r+', 'a', 'w-'].
        verbose : If true, prints save messages to std out
        """
        # if filename is None, returns the latter.
        filename = filename or self.parameter_dependent_filename()

        save_path = os.path.abspath(os.path.join(directory, filename))
        if verbose:
            print('Saving data to {}'.format(save_path))

        with h5py.File(save_path, 'a') as f:
            # The velocity field.
            f.create_dataset("field", data=self.transform(to='field').state)
            # The parameters required to exactly specify an orbit.
            f.create_dataset('parameters', data=tuple(float(p) for p in self.parameters))
            if include_residual:
                # This is included as a conditional statement because it seems strange to make importing/exporting
                # dependent upon full implementation of the governing equations; i.e. perhaps the equations
                # are still in development and data is used to test their implementation. In that case you would
                # not be able to export data which is undesirable.
                f.create_dataset("residual", data=float(self.residual()))
        return None

    def parameter_dependent_filename(self, extension='.h5', decimals=3):
        if self.dimensions() is not None:
            dimensional_string = ''.join(['_'+''.join([self.dimension_labels()[i], str(d).split('.')[0],
                                                       'p', str(d).split('.')[1][:decimals]])
                                          for i, d in enumerate(self.dimensions()()) if d not in [0., 0]])
        else:
            dimensional_string = ''
        return ''.join([self.__class__.__name__, dimensional_string, extension])

    def verify_integrity(self):
        """ Check the status of a solution, whether or not it converged to the correct orbit type. """
        return self

    def _parse_state(self, state, basis, **kwargs):
        """ Determine state shape parameters based on state array and the basis it is in.

        Parameters
        ----------
        state : ndarray
            Numpy array containing state information, can have any number of dimensions.
        basis : str
            The basis that the array 'state' is assumed to be in.
        kwargs :
            Signature for subclasses.

        """
        # Initialize with the same amount of dimensions as labels; use labels because staticmethod.
        # The 'and-or' trick; if state is None then latter is used.
        self.state = state or np.zeros(len(self.dimension_labels()))
        self.basis = basis

    def _parse_parameters(self, parameters, **kwargs):
        """ Determine the dimensionality and symmetry parameters.
        """
        # default is not to be constrained in any dimension;
        self.constraints = kwargs.get('constraints', {dim_key: False for dim_key in self.dimension_labels()})
        return None

    def random_state(self, parameters, **kwargs):
        """ Initial a set of random spatiotemporal Fourier modes
        Parameters
        ----------

        Returns
        -------
        Orbit :
        -----

        """
        return None

    def to_fundamental_domain(self, **kwargs):
        """ Placeholder for symmetry subclassees"""
        return self

    def from_fundamental_domain(self, **kwargs):
        """ Placeholder for symmetry subclassees"""
        return self

    def copy(self):
        """ Returns a shallow copy of an orbit instance.

        Returns
        -------
        Orbit :
        """
        return None


def convert_class(orbit, class_generator, **kwargs):
    """ Utility for converting between different classes.

    Parameters
    ----------
    orbit : Instance of OrbitKS or any of the derived classes.
        The orbit instance to be converted
    new_type : str or class object (not an instance).
        The target class that orbit will be converted to.


    Notes
    -----
    To avoid conflicts with projections onto invariant subspaces, the orbit is always transformed into field
    prior to conversion; the default basis to return is the basis of the input.

    """
    return class_generator(state=orbit.transform(to='field').state, basis='field',
                           parameters=kwargs.pop('parameters', orbit.parameters), **kwargs).transform(to=orbit.basis)
