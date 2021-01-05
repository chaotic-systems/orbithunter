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

    def __init__(self, state=None, basis='', parameters=(0., 0., 0., 0.), **kwargs):
        """ Base/Template class for orbits

        Parameters
        ----------
        state : ndarray, default None
            If an array, it should contain the state values congruent with the 'basis' argument.
        basis : str
            Which basis the array 'state' is currently in.
        parameters : tuple
            Parameters required to uniquely define the Orbit. Set here to be of length 4 because of 3+1 spacetime
            of the general case. Could be longer if symemtry related parameters exist.
        **kwargs :
            Extra arguments for _parse_parameters and random_state
                See the description of the aforementioned method.

        Notes
        -----
        If the state is not passed, then it will be randomly generated. This will require referencing the
        dimensions, expected to be nonzero to define the collocation grid. Therefore, if the user
        does not want randomly generated variables, it is required to either provide a state or a set of
        non-zero dimensions.

        Random generation of state is (for KSe) performed in the spatiotemporal basis.

        """
        # if state is not None:
        self._parse_parameters(parameters, **kwargs)
        self._parse_state(state, basis, **kwargs)
        # else:
        #     self._parse_parameters(parameters, nonzero_parameters=kwargs.pop('nonzero_parameters', True), **kwargs)
        #     # Generate a random initial state, whose shape is determined by parameter_based_discretization if
        #     # not provided.
        #     self.generate(**kwargs)

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
                 'parameters': tuple(str(np.round(p, 3)) for p in self.parameters),
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
        return self.rmatvec(dae, **kwargs)

    def resize(self, *new_shape, **kwargs):
        """ Rediscretization method

        Parameters
        ----------
        new_shape : int or tuple of ints
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
        new_shape = new_shape or self.parameter_based_discretization(self.dimensions(), **kwargs)

        # if passed as iterable, then unpack: i.e. ((a, b, ...)) -> (a, b, ...)
        if len(new_shape) == 1:
            new_shape = tuple(*new_shape)

        # If the current shape is different from the new shape, then resize, else, simply return the copy.
        if self.shapes()[0] != new_shape:
            # Although this is less efficient than doing every axis at once, it generalizes to cases where bases
            # are different for padding along different dimensions (i.e. transforms implicit in truncate and pad).
            for i, d in enumerate(new_shape):
                if d < self.shapes()[0][i]:
                    placeholder_orbit = placeholder_orbit.truncate(d, axis=i)
                elif d > self.shapes()[0][i]:
                    placeholder_orbit = placeholder_orbit.pad(d, axis=i)
                else:
                    pass

        return placeholder_orbit.transform(to=self.basis)

    def transform(self, to=''):
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
        The instance needs to be in spatiotemporal basis prior to computation; this avoids possible mistakes in the
        optimization process, which would result in a breakdown in performance from redundant transforms.
        """
        assert self.basis == self.bases()[-1], 'Convert to spatiotemporal basis before computing DAEs.'
        return self.__class__(state=np.zeros(self.shapes()[-1]), basis=self.bases()[-1], parameters=self.parameters)

    def residual(self, dae=True):
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
        if dae:
            v = self.transform(to=self.bases()[-1]).dae().state.ravel()
        else:
            v = self.state.ravel()

        return 0.5 * v.dot(v)

    def matvec(self, other, **kwargs):
        """ Matrix-vector product of a vector with the Jacobian of the current state.

        Notes
        -----
        Because the general Orbit template doesn't have an associated equation

        """
        return self.__class__(state=np.zeros(self.shape), basis=self.basis, parameters=self.parameters)

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

        return self.__class__(state=np.zeros(self.shape), basis=self.basis, parameters=self.parameters)

    def state_vector(self):
        """ Vector representation of orbit

        Returns
        -------
        ndarray :
            The state vector: the current state with parameters appended, returned as a (self.size + n_params , 1)
            dimensional array for scipy purposes.
        """
        return np.concatenate((self.state.ravel(), self.parameters), axis=0).reshape(-1, 1)

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

        Arguments for modifying the padding type are not included here. If extra functionality is desired then
        simply overload the method.
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
            tuple of three tuples corresponding to the shape of the 'state' array in the field, spatial_modes, modes bases.

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
        return tuple(getattr(self, p_label, 0.) for p_label in self.parameter_labels())

    def dimensions(self):
        """ Continuous tile dimensions

        Returns
        -------
        tuple :
            Tuple of dimensions, typically this will take the form (T, L_x, L_y, L_z) for (3+1)-D spacetime

        Notes
        -----
        Because this is usually a subset of self.parameters, it does not use the @property decorator. This method
        is purposed for readability.
        """
        return tuple(getattr(self, d_label, 0.) for d_label in self.dimension_labels())

    @staticmethod
    def bases():
        """ Labels of the different bases generated by different transforms. """
        return ''

    @staticmethod
    def parameter_generation_ranges():
        """ Ranges to uniformly sample from for parameter generation """
        return

    @staticmethod
    def parameter_labels():
        """ Strings to use to label dimensions/periods. Generic 3+1 spacetime labels default.
        """
        return 't', 'x', 'y', 'z'

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

    @classmethod
    def parameter_based_discretization(cls, dimensions, **kwargs):
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
            The new spatiotemporal field discretization; number of time points
            (rows) and number of space points (columns)

        Notes
        -----
        This function should only ever be called by resize, the returned values can always be accessed by
        the appropriate attributes of the rediscretized orbit.
        """
        return cls.default_shape()

    @classmethod
    def glue_parameters(cls, dimension_tuples, glue_shape=(1, 1, 1, 1), nonzero_mean=True):
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
        if nonzero_mean:
            return tuple(glue_shape[i] * p[p > 0.].mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                                in dimension_tuples))
        else:
            return tuple(glue_shape[i] * p.mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                        in dimension_tuples))

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
        field = self.transform(to=self.bases()[0]).state
        if method == 'absolute':
            rescaled_field = ((magnitude * field) / np.max(np.abs(field.ravel())))
        elif method == 'power':
            rescaled_field = np.sign(field) * np.abs(field) ** magnitude
        else:
            raise ValueError('Unrecognizable method.')
        return self.__class__(state=rescaled_field, basis=self.bases()[0],
                              parameters=self.parameters).transform(to=self.basis)

    def to_h5(self, filename=None, h5_group=None, h5py_mode='a', verbose=False, include_residual=False):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str, default None
            filename to write/append to.
        h5_group : str, default None
            Name of the h5_group wherein to store the Orbit in the h5_file at location filename. Should be
            HDF5 group name, i.e. '/A/B/C/...'
        h5py_mode : str
            Mode with which to open the file. Default is a, read/write if exists, create otherwise,
            other modes ['r+', 'a', 'w-', 'w']. See h5py.File for details. 'r' not allowed, because this is a function
            to write to the file.
        verbose : bool
            Whether or not to print save location and group
        include_residual : bool
            Whether or not to include residual as a dataset; requires DAE to be well-defined for current instance.
        """

        if verbose:
            print('Saving data to {} under group name'.format(filename, h5_group))

        with h5py.File(filename, h5py_mode) as f:
            # Returns h5_group if not None, else, filename method.
            orbit_group = f.require_group(h5_group or self.filename())
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
                except:
                    print('Unable to compute residual for {}'.format(repr(self)))
        return None

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
                                          for i, d in enumerate(self.dimensions()) if d not in [0., 0]])
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

        Notes
        -----
        This is quite simple for the general case, but left as a method for overloading purposes.
        """
        # Initialize with the same amount of dimensions as labels; use labels because staticmethod.
        # The 'and-or' trick; if state is None then latter is used. give empty array the expected number of
        # dimensions, even though array with 0 size in dimensions will typically be flattened by NumPy anyway.
        self.state = state or np.array([], dtype=float).reshape(len(self.default_shape()) * [0])
        self.basis = basis

    def _parse_parameters(self, parameters, **kwargs):
        """ Determine the dimensionality and symmetry parameters.
        """
        # default is not to be constrained in any dimension;
        self.constraints = kwargs.get('constraints', {dim_key: False for dim_key in self.dimension_labels()})
        # This will pad parameters with zeros if it is an iterable of smaller length than labels.
        for label, val in zip_longest(self.parameter_labels(), self.parameters, fillvalue=0.):
            setattr(self, label, val)

    def _generate_parameters(self, **kwargs):
        """ Randomly initialize parameters which are currently zero.

        Parameters
        ----------
        kwargs :
            p_ranges : dict
                keys are parameter_labels, values are uniform sampling intervals.

        Returns
        -------

        """
        p_ranges = kwargs.get('p_ranges', {p_label: (0, 1) for p_label in self.parameter_labels()})
        # seeding takes a non-trivial amount of time, only run if provided.
        if type(kwargs.get('seed', None)) is int:
            np.random.seed(kwargs.get('seed', None))

        # This will pad parameters with zeros if it is an iterable of smaller length than labels.
        for label, val in zip_longest(self.parameter_labels(), self.parameters, fillvalue=0.):
            if val == 0.:
                #
                pmin, pmax = p_ranges[label]
                val = pmin + (pmax - pmin) * np.random.rand()
                setattr(self, label, val)

    def _generate_state(self, **kwargs):
        # Just generate a random array; more intricate strategies should be written into subclasses.
        # Using standard normal distribution
        if type(kwargs.get('seed', None)) is int:
            np.random.seed(kwargs.get('seed', None))
        self.state = np.random.randn(self.parameter_based_discretization())
        self.basis = self.bases()[0]


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
        """
        #
        if attr in ['all', 'parameters']:
            self._generate_parameters(**kwargs)

        if attr in ['all', 'state']:
            # If generating a random st
            self._generate_state(**kwargs)

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
        return self.__class__(state=self.state.copy(), parameters=self.parameters, basis=self.basis)


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
    To avoid conflicts with projections onto symmetry invariant subspaces, the orbit is always transformed into the
    physical basis prior to conversion; the instance is returned in the basis of the input, however.

    """
    return class_generator(state=orbit.transform(to=orbit.bases()[0]).state, basis=orbit.bases()[0],
                           parameters=kwargs.pop('parameters', orbit.parameters), **kwargs).transform(to=orbit.basis)
