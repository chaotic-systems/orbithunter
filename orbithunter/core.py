from json import dumps
from itertools import zip_longest
import h5py
import numpy as np

__all__ = ['Orbit', 'convert_class']

"""
The core class for all orbithunter calculations. The methods listed are the ones used in the other modules. If
full functionality isn't currently desired then I recommend only implementing the methods used in optimize.py,
saving data to disk, and plotting. Of course this is in addition to the dunder methods such as __init__. 

While not listed here explicitly, this package is fundamentally a (pseudo)spectral method based package; while it is not
technically required to use a spectral method, not doing so may result in awkwardly named attributes. For example,
the resize method in spectral space essentially interpolates via zero-padding and truncation. Therefore, other
interpolation methods would be forced to use _pad and _truncate, unless they specifically overwrite the resize
method itself. Previous the labels on the bases were static but now everything is written such that they
can be accessed by an equation specific staticmethod .bases(). For the KSe this returns the tuple 
self.bases()--------->('field', 'spatial_modes', 'modes'), for example.
 
The implementation of this template class, Orbit, implements all numerical computations trivially; in other words,
the "associated equation" for this class is the trivial equation f=0, such that for any state the DAEs evaluate
to 0, the corresponding matrix vector products return zero, the Jacobian matrices are just appropriately sized
calls of np.zeros, etc. The idea is simply to implement the methods required for the numerical methods,
clipping, gluing, optimize, etc. Of course, in this case, all states are "solutions" and so any operations 
to the state doesn't *actually* do anything in terms of the "equation". The idea behind 
returning instances or arrays filled with zeros is to allow for debugging without having to complete everything
first; although one might argue *not* including the attributes/methods ensure that the team/person creating the module
does not forget anything. 

All transforms should be wrapped by
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

    def __init__(self, state=None, basis=None, parameters=None, **kwargs):
        """ Base/Template class for orbits

        Parameters
        ----------
        state : ndarray, default None
            If an array, it should contain the state values congruent with the 'basis' argument.
        basis : str, default None
            Which basis the array 'state' is currently in.
        parameters : tuple, default None
            Parameters required to uniquely define the Orbit.
        **kwargs :
            Extra arguments for _parse_parameters and random_state
                See the description of the aforementioned method.

        Notes
        -----
        If the state is not passed, then it will be randomly generated. This will require referencing the
        dimensions, expected to be nonzero to define the collocation grid. Therefore, if the user
        does not want randomly generated variables, it is required to either provide a isinstance(state, np.ndarray) or a set of
        non-zero dimensions.

        Random generation of state is (for KSe) performed in the spatiotemporal basis.

        """
        self._parse_parameters(parameters, **kwargs)
        self._parse_state(state, basis, **kwargs)

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
        return self.__class__(state=(self.state+other.state), basis=self.basis, parameters=self.parameters)

    def __radd__(self, other):
        """ Addition of Orbit states

        Parameters
        ----------
        other : Orbit

        """
        return self.__class__(state=(self.state+other.state), basis=self.basis, parameters=self.parameters)

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

    def __mul__(self, factor):
        """ Scalar multiplication of state values

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        """
        if issubclass(type(factor), Orbit):
            # product of 'states', which are themselves numpy arrays
            product = np.multiply(self.state, factor.state)
        elif isinstance(factor, np.ndarray):
            # product with numpy array, must have equivalent shape
            product = np.multiply(self.state, factor)
        else:
            # scalar multiplication
            product = factor * self.state
        return self.__class__(state=product, basis=self.basis, parameters=self.parameters)

    def __rmul__(self, factor):
        """ Scalar multiplication of state values

        Parameters
        ----------
        num : float
            Scalar value to multiply by.

        """
        if issubclass(type(factor), Orbit):
            # product of 'states', which are themselves numpy arrays
            product = np.multiply(self.state, factor.state)
        elif isinstance(factor, np.ndarray):
            # product with numpy array, must have equivalent shape
            product = np.multiply(self.state, factor)
        else:
            # scalar multiplication
            product = factor * self.state
        return self.__class__(state=product, basis=self.basis, parameters=self.parameters)

    def __truediv__(self, divisor):
        """ Scalar division of state values

        Parameters
        ----------
        num : float
            Scalar value to divide by
        """
        if issubclass(type(divisor), Orbit):
            # product of 'states', which are themselves numpy arrays
            quotient = np.divide(self.state, divisor.state)
        elif isinstance(divisor, np.ndarray):
            # product with numpy array, must have equivalent shape
            quotient = np.divide(self.state, divisor)
        else:
            # scalar multiplication
            quotient = self.state / divisor
        return self.__class__(state=quotient, basis=self.basis, parameters=self.parameters)

    def __floordiv__(self, divisor):
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
        if issubclass(type(divisor), Orbit):
            # product of 'states', which are themselves numpy arrays
            quotient = np.floor_divide(self.state, divisor.state)
        elif isinstance(divisor, np.ndarray):
            # product with numpy array, must have equivalent shape
            quotient = np.floor_divide(self.state, divisor)
        else:
            # scalar multiplication
            quotient = self.state // divisor
        return self.__class__(state=quotient, basis=self.basis, parameters=self.parameters)

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

        if self.parameters is not None:
            # parameters should be an iterable, but this allows for singletons.
            try:
                pretty_params = tuple(round(x, 3) if isinstance(x, float)
                                      else x for x in self.parameters)
            except TypeError:
                pretty_params = self.parameters
        else:
            pretty_params = None

        dict_ = {'shape': self.shape,
                 'basis': self.basis,
                 'parameters': pretty_params}
        # convert the dictionary to a string via json.dumps
        dictstr = dumps(dict_)
        return self.__class__.__name__ + '(' + dictstr + ')'

    def __getattr__(self, attr):
        # Called if self.attr is not found or getattr(orbit, 'attr') is called.
        try:
            attr = str(attr)
        except ValueError:
            print('Attribute is not of readable type')

        if attr in self.parameter_labels():
            # parameters must be cast as tuple, (p,) if singleton.
            if isinstance(self.parameters, tuple):
                # breaks down for non-unique labels; there is no reason for non-unique labels.
                return self.parameters[self.parameter_labels().index(attr)]
            else:
                return 0.
        elif attr in self.discretization_labels():
            # Note that this breaks down if discretization_labels are not unique.
            if isinstance(self.discretization, tuple):
                return self.discretization[self.discretization_labels().index(attr)]
            else:
                return 0.
        elif attr in ['state', 'basis']:
            return None
        elif attr in ['shape', 'size']:
            if not isinstance(self.state, np.ndarray):
                raise AttributeError('"shape" and "size" only defined for instances whose "state" is NumPy ndarray.')
        else:
            error_message = ' '.join([self.__class__.__name__, 'has no attribute\'{}\''.format(attr)])
            raise AttributeError(error_message)

    @staticmethod
    def bases():
        """ Labels of the different bases generated by different transforms.

        Notes
        -----
        Defaults to 'physical' and not None or empty string because it is used as a data group for writing .h5 files.
        """
        return ('physical',)

    @staticmethod
    def parameter_labels():
        """ Strings to use to label dimensions/periods. Generic 3+1 spacetime labels default.
        """
        return 't', 'x', 'y', 'z'

    @staticmethod
    def discretization_labels():
        """ Strings to use to label dimensions/periods. Generic 3+1 spacetime labels default.
        """
        return 'n', 'i', 'j', 'k'

    @classmethod
    def default_parameter_ranges(cls):
        return {p_label: (0, 1) for p_label in cls.parameter_labels()}

    @staticmethod
    def dimension_labels():
        """ Strings to use to label dimensions/periods; this is redundant for Orbit class.
        """
        return 't', 'x', 'y', 'z'

    @staticmethod
    def default_shape():
        """ The shape of a generic state, not based on any dimensions.

        Returns
        -------
        tuple of int :
            The default array shape when dimensions are not specified.
        """
        return 1, 1, 1, 1

    @staticmethod
    def minimal_shape():
        """ The smallest possible compatible discretization

        Returns
        -------
        tuple of int :
            The default array shape when dimensions are not specified.

        Notes
        -----
        Often symmetry constraints reduce the dimensionality; if too small this reduction may leave the state empty,
        used for aspect ratio correction and possibly other gluing applications.

        """
        return 1, 1, 1, 1

    def statemul(self, other):
        """ Elementwise multiplication of two Orbits states

        Parameters
        ----------
        other : Orbit instance
            Second component of the state multiplication

        Returns
        -------
        OrbitKS :
            The OrbitKS representing the product.

        Notes
        -----
        Only really makes sense when taking an elementwise product between Orbits defined on spatiotemporal
        domains of the same size.
        """
        if isinstance(other, np.ndarray):
            product = np.multiply(self.state, other)
        else:
            product = np.multiply(self.state, other.state)
        return self.__class__(state=product, basis=self.basis, parameters=self.parameters)

    def cost_function_gradient(self, eqn, **kwargs):
        """ Derivative of 1/2 |F|^2

        Parameters
        ----------
        eqn : Orbit
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
        return self.rmatvec(eqn, **kwargs)

    def resize(self, *new_discretization, **kwargs):
        """ Rediscretization method

        Parameters
        ----------
        new_discretization : int or tuple of ints
            New discretization size
        kwargs : dict
            keyword arguments for parameter_based_discretization.

        Returns
        -------
        Orbit :
            Orbit with newly discretized fields.

        Notes
        -----
        Technically this isn't "reshaping" in the NumPy sense, it is rediscretizing the current state; was
        previously called "reshape".
        """
        # Padding basis assumed to be in the spatiotemporal basis.
        placeholder_orbit = self.copy().transform(to=self.bases()[-1])

        # if nothing passed, then new_shape == () which evaluates to false.
        # The default behavior for this will be to modify the current discretization
        # to a `parameter based discretization'.
        new_shape = new_discretization or self.parameter_based_discretization(self.parameters, **kwargs)

        # unpacking unintended nested tuples i.e. ((a, b, ...)) -> (a, b, ...); leaves unnested tuples invariant.
        if isinstance(new_shape, tuple) and len(new_shape) == 1:
            new_shape = tuple(*new_shape)

        # If the current shape is discretization size (not current shape) differs from shape then resize
        if self.discretization != new_shape:
            # Although this is less efficient than doing every axis at once, it generalizes to cases where bases
            # are different for padding along different dimensions (i.e. transforms implicit in truncate and pad).
            for i, d in enumerate(new_shape):
                if d < self.discretization[i]:
                    placeholder_orbit = placeholder_orbit.truncate(d, axis=i)
                elif d > self.discretization[i]:
                    placeholder_orbit = placeholder_orbit.pad(d, axis=i)
                else:
                    pass

        return placeholder_orbit.transform(to=self.basis)

    def transform(self, to=None):
        """ Method that handles all basis transformations.

        Parameters
        ----------
        to : str
            The basis to transform into. If already in said basis, returns self. Default written here as '', but
            can of course be changed to suit the equations.

        Returns
        -------
        Orbit :
            either self or instance in new basis.
        """
        return self

    def eqn(self, *args, **kwargs):
        """ The governing equations evaluated using the current state.

        Returns
        -------
        Orbit :
            Orbit instance whose state equals evaluation of governing equation.

        Notes
        -----
        If self.eqn().state = 0. at every point (within some numerical tolerance), then 'self' constitutes
        a solution to the governing equation. Of course there is no equation for this class, so zeros are returned.
        The instance needs to be in spatiotemporal basis prior to computation; this avoids possible mistakes in the
        optimization process, which would result in a breakdown in performance from redundant transforms.
        """
        assert self.basis == self.bases()[-1], 'Convert to spatiotemporal basis before computing DAEs.'
        return self.__class__(state=np.zeros(self.shapes()[-1]), basis=self.bases()[-1], parameters=self.parameters)

    def residual(self, eqn=True):
        """ The value of the cost function

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 the squared L_2 norm of the spatiotemporal mapping,
            R = 1/2 ||F||^2. The current form generalizes to any equation.

        Notes
        -----
        In certain optimization methods, it is more efficient to have the DAEs stored, and then take their norm
        as opposed to re-evaluating the DAEs. The reason why .norm() isn't called instead is to allow for different
        residual functions other than, for instance, the L_2 norm of the DAEs; although in this case there is no
        difference.
        """
        if eqn:
            v = self.transform(to=self.bases()[-1]).eqn().state.ravel()
        else:
            v = self.state.ravel()

        return 0.5 * v.dot(v)

    def matvec(self, other, **kwargs):
        """ Matrix-vector product of a vector with the Jacobian of the current state.

        Notes
        -----
        Because the general Orbit template doesn't have an associated equation

        """
        return self.__class__(state=np.zeros(self.shape), basis=self.basis,
                              parameters=tuple([0] * len(self.parameter_labels())))

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

        return self.__class__(state=np.zeros(self.shape), basis=self.basis,
                              parameters=tuple([0] * len(self.parameter_labels())))

    def orbit_vector(self):
        """ Vector representation of orbit

        Returns
        -------
        ndarray :
            The state vector: the current state with parameters appended, returned as a (self.size + n_params , 1)
            dimensional array for scipy purposes.
        """
        return np.concatenate((self.state.ravel(), self.parameters), axis=0).reshape(-1, 1)

    def from_numpy_array(self, orbit_vector, **kwargs):
        """ Utility to convert from numpy array to orbithunter format for scipy wrappers.

        Parameters
        ----------
        orbit_vector : ndarray
            State vector containing state values (modes, typically) and parameters or optimization corrections thereof.

        kwargs :
            parameters : tuple
                If parameters from another Orbit instance are to overwrite the values within the orbit_vector
            parameter_constraints : dict
                constraint dictionary, keys are parameter_labels, values are bools
            OrbitKS or subclass kwargs : dict
                If special kwargs are required/desired for Orbit instantiation.
        Returns
        -------
        state_orbit : Orbit instance
            Orbit instance whose state and parameters are extracted from the input orbit_vector.

        Notes
        -----
        Important: If parameters are passed as a keyword argument, they are appended to the numpy array,
        'state_array', via concatenation. The common usage of this function is to take the output of SciPy
        optimization functions (numpy arrays) and store them in Orbit instances.

        This function assumes that the instance calling it is in the "spatiotemporal" basis; the basis in which
        the optimization occurs. This is why no additional specification for size and shape and basis is required.
        The spatiotemporal basis is allowed to be named anything, hence usage of self.basis.
        """
        # slice out the parameters; cast as list to gain access to pop
        params_list = list(kwargs.pop('parameters', orbit_vector.ravel()[self.size:].tolist()))
        # The usage of this function is to convert a vector of corrections to an orbit instance;
        # while default parameter values may be None, default corrections are 0.
        parameters = tuple(params_list.pop(0) if not constrained and params_list else 0
                           for constrained in self.constraints.values())
        return self.__class__(state=np.reshape(orbit_vector.ravel()[:self.size], self.shape), basis=self.basis,
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
        incremented_params = tuple(
                                   self_param + step_size * other_param
                                   if other_param is not None else self_param # assumed to be constrained if None
                                   for self_param, other_param in zip(self.parameters, other.parameters)
                                   )

        return self.__class__(state=self.state + step_size * other.state, basis=self.basis,
                              parameters=incremented_params, **kwargs)

    def pad(self, size, axis=0):
        """ Increase the size of the discretization along an axis.

        Parameters
        ----------
        size : int
            The new size of the discretization, restrictions typically imposed by equations.
        axis : int
            Axis to pad along per numpy conventions.

        Returns
        -------
        Orbit :
            Orbit instance whose state in the physical (self.bases()[0]) basis has a number of discretization
            points equal to 'size'

        Notes
        -----
            This function is typically an interpolation method, i.e. Fourier mode zero-padding.
        However, in the general case when we cannot assume the basis, the best we can do is pad the current basis,
        which is done in a symmetric fashion when possible.
            That is, if we have a 4x4 array, then calling this with size=6 and axis=0 would yield a 6x4 array, wherein
        the first and last rows are uniformly zero. I.e. a "border" of zeroes has been added. The choice to make
        this symmetric matters in the case of non-periodic boundary conditions.

        When writing this function for spectral interpolation methods BE SURE TO ACCOUNT FOR NORMALIZATION
        of your transforms. Also, in this instance the interpolation basis and the return basis are the same, as there
        is no way of specifying otherwise for the general Orbit class. For the KSe, the padding basis is 'modes'
        and the return basis is whatever the state was originally in. This is the preferred implementation.
        """
        padding_size = (size - self.shape[axis]) // 2
        if int(size) % 2:
            # If odd size then cannot distribute symmetrically, floor divide then add append extra zeros to beginning
            # of the dimension.
            padding_tuple = ((padding_size + 1, padding_size) if i == axis else (0, 0) for i in range(len(self.shape)))
        else:
            padding_tuple = ((padding_size, padding_size) if i == axis else (0, 0) for i in range(len(self.shape)))

        return self.__class__(state=np.pad(self.state, padding_tuple), basis=self.basis,
                              parameters=self.parameters).transform(to=self.basis)

    def truncate(self, size, axis=0):
        """ Decrease the size of the discretization along an axis

        Parameters
        -----------
        size : int
            The new size of the discretization, must be an even integer
            smaller than the current size of the discretization.
        axis : int
            Axis to truncate along per numpy conventions.

        Returns
        -------
        OrbitKS
            OrbitKS instance with smaller discretization.

        Notes
        -----
        The inverse of pad.
        """
        truncate_size = (self.shape[axis] - size) // 2
        if int(size) % 2:
            # If odd size then cannot distribute symmetrically, floor divide then add append extra zeros to beginning
            # of the dimension.
            truncate_slice = tuple(slice(truncate_size + 1, -truncate_size) if i == axis else slice(None)
                                   for i in range(len(self.shape)))
        else:
            truncate_slice = tuple(slice(truncate_size, -truncate_size) if i == axis else slice(None)
                                   for i in range(len(self.shape)))

        return self.__class__(state=self.state[truncate_slice], basis=self.basis,
                              parameters=self.parameters).transform(to=self.basis)

    def jacobian(self, **kwargs):
        """ Jacobian matrix evaluated at the current state.
        Parameters
        ----------

        Returns
        -------
        jac_ : matrix
        2-d numpy array equalling the Jacobian matrix of the governing equations evaluated at current state.
        """
        return np.zeros([self.size, self.orbit_vector().size])

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
            Contains shapes of state in all bases.

        Notes
        -----
        This is a convenience function for operations which require the shape of the state array in a different basis.
        These shapes are defined by the transforms, essentially, but it is wasteful to transform simply for the shape,
        and the amount of boilerplate code to constantly infer the shape justifies this method in most cases.
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

    def dimensions(self):
        """ Continuous tile dimensions

        Returns
        -------
        tuple :
            Tuple of dimensions, typically this will take the form (T, L_x, L_y, L_z) for (3+1)-D spacetime

        Notes
        -----
        Because this is usually a subset of self.parameters, it does not use the property decorator. This method
        is purposed for readability.
        """
        # collect dimensions
        dims = tuple(getattr(self, d_label) for d_label in self.dimension_labels())
        # check if all NoneType, if so, simply return NoneType; this allows future comparison
        # without having to know number of dimensions
        if len(dims) == dims.count(None):
            return None
        else:
            return dims

    @classmethod
    def parameter_based_discretization(cls, parameters, **kwargs):
        """ Follow orbithunter conventions for discretization size.

        Parameters
        ----------
        dimensions : tuple
            Tuple containing dimensions

        kwargs :
            resolution : str
            Takes values 'coarse', 'normal', 'fine', 'power'.
            These options return the according discretization sizes as described below.

        Returns
        -------
        N, M : tuple of ints
            The new spatiotemporal state discretization; number of time points
            (rows) and number of space points (columns)

        Notes
        -----
        This function should only ever be called by resize, the returned values can always be accessed by
        the appropriate attributes of the rediscretized orbit.
        """
        return cls.default_shape()

    @classmethod
    def glue_parameters(cls, dimension_tuples, glue_shape, non_zero=True):
        """ Class method for handling parameters in gluing

        Parameters
        ----------
        dimension_tuples : tuple of tuples

        glue_shape : tuple of ints
            The shape of the gluing being performed i.e. for a 2x2 orbit grid glue_shape would equal (2,2).

        Returns
        -------
        glued_parameters : tuple
            tuple of parameters same dimension and type as self.parameters

        Notes
        -----
        This returns an average of parameter tuples, used exclusively in the gluing method; wherein the new tile
        dimensions needs to be decided upon/inferred from the original tiles. As this average is only a very
        crude approximation, it can be worthwhile to also simply search the parameter space for a combination
        of dimensions which reduces the residual. The strategy produced by this method is simply a baseline.

        """
        if non_zero:
            return tuple(glue_shape[i] * p[p > 0.].mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                                in dimension_tuples))
        else:
            return tuple(glue_shape[i] * p.mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                        in dimension_tuples))

    def plot(self, show=True, save=False, padding=True, fundamental_domain=True, **kwargs):
        """ Signature for plotting method.
        """
        return None

    def rescale(self, magnitude, method='inf'):
        """ Scalar multiplication

        Notes
        -----
        This rescales the physical state such that the absolute value of the max/min takes on a new value
        of magnitude
        """
        state = self.transform(to=self.bases()[0]).state
        if method == 'inf':
            rescaled_state = magnitude * state / np.max(np.abs(state.ravel()))
        elif method == 'l1':
            rescaled_state = magnitude * state / np.linalg.norm(state, ord=1)
        elif method == 'l2':
            rescaled_state = magnitude * state / np.linalg.norm(state)
        elif method == 'power':
            rescaled_state = np.sign(state) * np.abs(state) ** magnitude
        else:
            raise ValueError('Unrecognizable method.')
        return self.__class__(state=rescaled_state, basis=self.bases()[0],
                              parameters=self.parameters).transform(to=self.basis)

    def to_h5(self, filename=None, orbit_name=None, h5py_mode='r+', verbose=False, include_residual=False):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str, default None
            filename to write/append to.
        orbit_name : str, default None
            Name of the orbit_name wherein to store the Orbit in the h5_file at location filename. Should be
            HDF5 group name, i.e. '/A/B/C/...'
        h5py_mode : str
            Mode with which to open the file. Default is a, read/write if exists, create otherwise,
            other modes ['r+', 'a', 'w-', 'w']. See h5py.File for details. 'r' not allowed, because this is a function
            to write to the file. Defaults to r+ to prevent overwrites.
        verbose : bool
            Whether or not to print save location and group
        include_residual : bool
            Whether or not to include residual as a dataset; requires DAE to be well-defined for current instance.
        """

        if verbose:
            print('Saving data to {} under group name'.format(filename, orbit_name))

        with h5py.File(filename, h5py_mode) as f:
            # Returns orbit_name if not None, else, filename method.
            orbit_group = f.require_group(orbit_name or self.filename(extension=''))
            # State may be empty, but can still save.
            orbit_group.create_dataset(self.bases()[0], data=self.transform(to=self.bases()[0]).state)
            # The parameters required to exactly specify an orbit.
            orbit_group.create_dataset('parameters', data=tuple(float(p) for p in self.parameters))
            if include_residual:
                # This is included as a conditional statement because it seems strange to make importing/exporting
                # dependent upon full implementation of the governing equations; i.e. perhaps the equations
                # are still in development and data is used to test their implementation. In that case you would
                # not be able to export data which is undesirable.
                try:
                    orbit_group.create_dataset('residual', data=float(self.residual()))
                except (ZeroDivisionError, ValueError):
                    print('Unable to compute residual for {}'.format(repr(self)))

    def filename(self, extension='.h5', decimals=3):
        """ Method for consistent/conventional filenaming. High dimensions will yield long filenames.

        Parameters
        ----------
        extension : str
            The extension to append to the filename
        decimals :
            The number of decimals to include in the str name of the orbit.

        Returns
        -------
        str :
            The conventional filename.
        """
        if self.dimensions() is not None:
            dimensional_string = ''.join(['_' + ''.join([self.dimension_labels()[i], str(d).split('.')[0],
                                                         'p', str(d).split('.')[1][:decimals]])
                                          for i, d in enumerate(self.dimensions()) if (d != 0) and (d is not None)])
        else:
            dimensional_string = ''
        return ''.join([self.__class__.__name__, dimensional_string, extension])

    def verify_integrity(self):
        """ Check the status of a solution, whether or not it converged to the correct orbit type. """
        return self

    def generate(self, attr='all', **kwargs):
        """ Initialize a random state.
        Parameters
        ----------
        attr : str
            Takes values 'state', 'parameters' or 'all'.


        Notes
        -----
        This should mimic the behavior of __init__ in the sense that
        the "solution" to not providing anything to __init__ is to set attr='all'. If you do not provide a state,
        then generate attr='state', if you do not provide parameters (and you do not want them to take the default
        values), then attr='parameters'.

        If extra values / variables not in the optimization parameters or state need to be declared then it is
        recommended to call super and then do extra operations after
        """
        #
        if attr in ['all', 'parameters']:
            self._generate_parameters(**kwargs)

        if attr in ['all', 'state']:
            self._generate_state(**kwargs)
        # For chaining operations.
        return self

    def to_fundamental_domain(self, **kwargs):
        """ Placeholder for symmetry subclassees"""
        return self

    def from_fundamental_domain(self, **kwargs):
        """ Placeholder for symmetry subclassees"""
        return self

    def copy(self):
        """ Return a deep copy of numpy array.

        Returns
        -------
        Orbit :
        """
        return self.__class__(state=self.state.copy(), parameters=self.parameters, basis=self.basis)

    def constrain(self, *labels):
        """

        Parameters
        ----------
        label

        Returns
        -------

        """
        # Maintain other constraints when constraining.
        constraints = {key: (True if key in tuple(*labels) else False)
                       for key, val in self.constraints.items()}
        setattr(self, 'constraints', constraints)

    def _parse_state(self, state, basis, **kwargs):
        """ Determine state shape parameters based on state array and the basis it is in.

        Parameters
        ----------
        state : ndarray
            Numpy array containing state information, can have any number of dimensions.
        basis : str
            The basis that the array 'state' is assumed to be in.

        Notes
        -----
        This is quite simple for the general case, but left as a method for overloading purposes.
        """
        # Initialize with the same amount of dimensions as labels; use labels because staticmethod.
        # The 'and-or' trick; if state is None then latter is used. give empty array the expected number of
        # dimensions, even though array with 0 size in dimensions will typically be flattened by NumPy anyway.
        if isinstance(state, np.ndarray):
            self.state = state
        else:
            self.state = np.array([], dtype=float).reshape(len(self.default_shape()) * [0])

        if self.size > 0:
            # This seems redundant but typically the discretization needs to be inferred from the state
            # and the basis; as the number of variables is apt to change when symmetries are taken into account.
            self.basis = basis
            self.discretization = self.state.shape
            if basis is None:
                raise ValueError('basis must be provided when state is provided')
        else:
            self.discretization = None
            self.basis = None

    def _parse_parameters(self, parameters, **kwargs):
        """ Parse and initialize the set of parameters

        Notes
        -----
        Parameters are required to be numerical in type. If there are categorical parameters then they
        should be assigned to a different attribute. The reason for this is that for numerical optimization,
        the orbit_vector; the concatenation of self.state and self.parameters is sent to the various algorithms.
        Cannot send categoricals to these algorithms.

        """
        # default is not to be constrained in any dimension;
        self.constraints = kwargs.get('constraints', {dim_key: False for dim_key in self.parameter_labels()})
        if parameters is None:
            self.parameters = parameters
        elif isinstance(parameters, tuple):
            # This ensures all parameters are filled, even if NoneType
            self.parameters = tuple(val for label, val in zip_longest(self.parameter_labels(), parameters,
                                                                      fillvalue=0))
        else:
            # A number of methods require parameters to be an iterable, hence the tuple requirement.
            raise TypeError('"parameters" is required to be a tuple or None. '
                            'singleton parameter "p" needs to be cast as tuple (p,).')

    def _generate_parameters(self, **kwargs):
        """ Randomly initialize parameters which are currently zero.

        Parameters
        ----------
        kwargs :
            p_ranges : dict
                keys are parameter_labels, values are uniform sampling intervals or iterables to sample from

        Returns
        -------

        """
        # helper function so comprehension can be used later on.
        def sample_from_generator(val, val_generator):
            if val is None:
                # for numerical parameter generators we're going to use uniform distribution to generate values
                # If the generator is "interval like" then use uniform distribution.
                if isinstance(val_generator, tuple) and len(val_generator) == 2:
                    pmin, pmax = val_generator
                    val = pmin + (pmax - pmin) * np.random.rand()
                # Everything else treated as distribution to sample from
                else:
                    val = np.random.choice(val_generator)
            return val

        # seeding takes a non-trivial amount of time, only set if explicitly provided.
        if isinstance(kwargs.get('seed', None), int):
            np.random.seed(kwargs.get('seed', None))

        # Can be useful to override default sample spaces to get specific cases.
        p_ranges = kwargs.get('parameter_ranges', self.default_parameter_ranges())
        # If *some* of the parameters were initialized, we want to save those values; iterate over the current
        # parameters if not None, else,
        parameter_iterable = self.parameters or len(self.parameter_labels()) * [0]
        parameters = tuple(sample_from_generator(val, p_ranges[label]) for label, val
                           in zip_longest(self.parameter_labels(), parameter_iterable, fillvalue=0))
        setattr(self, 'parameters', parameters)

    def _generate_state(self, **kwargs):
        """ Populate the 'state' attribute

        Parameters
        ----------
        kwargs

        Notes
        -----
        Must generate and set attributes 'state' and 'discretization'. The state is required to be a numpy
        array, the discretization is required to be its shape (tuple) in the basis specified by self.bases()[0].
        Discretization is coupled to the state, hence why it is generated here and not on its own.
        """
        # Just generate a random array; more intricate strategies should be written into subclasses.
        # Using standard normal distribution
        numpy_seed = kwargs.get('seed', None)
        if isinstance(numpy_seed, int):
            np.random.seed(numpy_seed)
        self.discretization = self.parameter_based_discretization(self.parameters, **kwargs)
        self.state = np.random.randn(*self.discretization)
        self.basis = self.bases()[0]


def convert_class(orbit, class_generator, **kwargs):
    """ Utility for converting between different classes.

    Parameters
    ----------
    orbit : Orbit instance
        The orbit instance to be converted
    class_generator : class generator
        The target class that orbit will be converted to.

    Notes
    -----
    To avoid conflicts with projections onto symmetry invariant subspaces, the orbit is always transformed into the
    physical basis prior to conversion; the instance is returned in the basis of the input, however.

    """
    return class_generator(state=orbit.transform(to=orbit.bases()[0]).state, basis=orbit.bases()[0],
                           parameters=kwargs.pop('parameters', orbit.parameters), **kwargs).transform(to=orbit.basis)
