from json import dumps
import h5py
import numpy as np
import itertools
import warnings

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

The way that binary operations and dunder methods involving the state are designed is to maintain all attributes of a 
state except for the state itself. 

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
                
        """
        self._parse_parameters(parameters, **kwargs)
        self._parse_state(state, basis, **kwargs)

    def __add__(self, other):
        """ Addition of Orbit state and other numerical quantity.

        Parameters
        ----------
        other : Orbit, ndarray, float, int
        """
        if issubclass(type(other), Orbit):
            result = self.state + other.state
        else:
            result = self.state + other

        # construct a new instance with same attributes except for state, which is updated by merging dicts
        return self.__class__(**{**vars(self), 'state': result})

    def __radd__(self, other):
        """ Addition of Orbit state and other numerical quantity.

        Parameters
        ----------
        other : Orbit, ndarray, float, int
        """
        if issubclass(type(other), Orbit):
            result = other.state + self.state
        else:
            result = other + self.state

        return self.__class__(**{**vars(self), 'state': result})

    def __sub__(self, other):
        """ Subtraction of other numerical quantity from Orbit state.

        Parameters
        ----------
        other : Orbit, ndarray, float, int
        """
        if issubclass(type(other), Orbit):
            result = self.state - other.state
        else:
            result = self.state - other
        return self.__class__(**{**vars(self), 'state': result})

    def __rsub__(self, other):
        """ Subtraction of Orbit state from other numeric quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            result = other.state - self.state
        else:
            result = other - self.state
        return self.__class__(**{**vars(self), 'state': result})

    def __mul__(self, other):
        """ Multiplication of Orbit state and other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Notes
        -----
        If user defined classes are not careful with shapes then accidental outer products can happen (i.e.
        array of shape (x,) * array of shape (x, 1) = array of shape (x, x)
        """
        if issubclass(type(other), Orbit):
            result = np.multiply(self.state, other.state)
        else:
            result = np.multiply(self.state, other)

        return self.__class__(**{**vars(self), 'state': result})

    def __rmul__(self, other):
        """ Multiplication of Orbit state and other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Notes
        -----
        If user defined classes are not careful with shapes then accidental outer products can happen (i.e.
        array of shape (x,) * array of shape (x, 1) = array of shape (x, x)
        """
        if issubclass(type(other), Orbit):
            result = np.multiply(self.state, other.state)
        else:
            result = np.multiply(self.state, other)
        return self.__class__(**{**vars(self), 'state': result})

    def __truediv__(self, other):
        """ Division of Orbit state by other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Notes
        -----
        If user defined classes are not careful with shapes then accidental outer products can happen (i.e.
        array of shape (x,) / array of shape (x, 1) = array of shape (x, x)
        """
        if issubclass(type(other), Orbit):
            result = np.divide(self.state, other.state)
        else:
            result = np.divide(self.state, other)
        return self.__class__(**{**vars(self), 'state': result})

    def __floordiv__(self, other):
        """ Floor division of Orbit state by other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Notes
        -----
        If user defined classes are not careful with shapes then accidental outer products can happen (i.e.
        array of shape (x,) // array of shape (x, 1) = array of shape (x, x)
        """
        if issubclass(type(other), Orbit):
            result = np.floor_divide(self.state, other.state)
        else:
            result = np.floor_divide(self.state, other)
        return self.__class__(**{**vars(self), 'state': result})

    def __pow__(self, other):
        """ Exponentiation of Orbit state.

        Parameters
        ----------
        other : Orbit, ndarray, float, int
        """
        if issubclass(type(other), Orbit):
            result = self.state**other.state
        else:
            result = self.state**other
        return self.__class__(**{**vars(self), 'state': result})

    def __str__(self):
        """ String name

        Returns
        -------
        str :
            Of the form 'Orbit'
        """
        return self.__class__.__name__

    def __repr__(self):
        """ More descriptive than __str__ using beautified parameters."""
        if self.parameters is not None:
            # parameters should be an iterable
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
        """ Retrieve parameters, discretization variables with their labels instead of slicing 'parameters'

        Notes
        -----
        This is simply to avoid defining attributes or properties for each variable name; keep the namespace clean.
        """
        try:
            attr = str(attr)
        except ValueError:
            print('Attribute is not of readable type')

        if attr in self.parameter_labels():
            # parameters must be cast as tuple, (p,) if singleton.
            return self.parameters[self.parameter_labels().index(attr)]
        elif attr in self.discretization_labels():
            # discretization must be tuple, (d,) if singleton.
            return self.discretization[self.discretization_labels().index(attr)]
        else:
            error_message = ' '.join([self.__class__.__name__, 'has no attribute\'{}\''.format(attr)])
            raise AttributeError(error_message)

    @staticmethod
    def bases():
        """ Labels of the different bases that 'state' attribute can be in.

        Notes
        -----
        Defaults to 'physical' and not None or empty string because it is used as a data group for writing .h5 files.
        """
        return ('physical',)

    @staticmethod
    def ndims():
        """ Number of expected dimensions of state array

        Notes
        -----
        Auxiliary usage is to use inherit default labels; that is, so the labels defined as staticmethods do not
        have to be repeated for derived classes.
        """
        return 4

    @staticmethod
    def parameter_labels():
        """ Strings to use to label dimensions. Generic 3+1 spacetime labels default.

        Notes
        -----
        It might seem idiotic to have both a parameter labels staticmethod and parameters as a tuple; why not just make
        a dict? Because I wanted an immutable type for parameters that could easily be converted into a numpy array.
        """
        return 't', 'x', 'y', 'z'

    @staticmethod
    def dimension_labels():
        """ Strings to use to label dimensions/periods; this is redundant for Orbit class.
        """
        return 't', 'x', 'y', 'z'

    @staticmethod
    def discretization_labels():
        """ Strings to use to label discretization variables. Generic 3+1 spacetime labels default.
        """
        return 'n', 'i', 'j', 'k'

    @classmethod
    def default_parameter_ranges(cls):
        """ Intervals (continuous) or iterables (discrete) used to generate parameters.
        Notes
        -----
        tuples or length two are *always* interpreted to be continuous intervals. If you have a discrete variable
        with two options, simply use a list instead of tuple.
        """
        return {p_label: (0, 1) for p_label in cls.parameter_labels()}

    @staticmethod
    def default_shape():
        """ The default array shape when dimensions are not specified.
        """
        return 2, 2, 2, 2

    @staticmethod
    def minimal_shape():
        """ The smallest possible compatible discretization for the given class.

        Returns
        -------
        tuple of int :
            The minimal shape that the shape can take and still have numerical operations (transforms mainly)
            be compatible

        Notes
        -----
        Often symmetry constraints reduce the dimensionality; if too small this reduction may leave the state empty,
        this is used for aspect ratio correction and possibly other gluing applications.
        """
        return 1, 1, 1, 1

    @staticmethod
    def minimal_shape_increments():
        """ The smallest valid increment to change the discretization by.

        Returns
        -------
        tuple of int :
            The smallest valid increments to changes in discretization size; presumably to retain all functionality.

        Notes
        -----
        Used in aspect ratio correction and "discretization continuation". For example, the KSe code requires
        even valued field discretizations; therefore the minimum increments for the KSE are 2's.
        """
        return 1, 1, 1, 1

    @classmethod
    def default_constraints(cls):
        return {k: False for k in cls.parameter_labels()}

    @property
    def shape(self):
        """ Current state's shape

        Notes
        -----
        Just a convenience to be able to write self.shape instead of self.state.shape
        """
        return self.state.shape

    @property
    def size(self):
        """ Current state's total dimensionality

        Notes
        -----
        Just a convenience to be able to write self.sizeinstead of self.state.size
        """
        return self.state.size

    @classmethod
    def dimension_based_discretization(cls, parameters, **kwargs):
        """ Follow orbithunter conventions for discretization size.

        Parameters
        ----------
        parameters : tuple
            Values from which the discretization may be inferred.

        kwargs :
            Various flags for defining discretization; can be highly dependent on equation and so
            is left as vague as possible.

        Returns
        -------
        tuple :
            A tuple of ints
        """
        return cls.default_shape()

    @classmethod
    def glue_dimensions(cls, dimension_tuples, glue_shape, exclude_zero_dimensions=True):
        """ Class method for handling parameters in gluing

        Parameters
        ----------
        dimension_tuples : tuple of tuples

        glue_shape : tuple of ints
            The shape of the gluing being performed i.e. for a 2x2 orbit grid glue_shape would equal (2,2).
        exclude_zero_dimensions : bool
            If True, then the calculation of average dimensions excludes 0's.

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
        if exclude_zero_dimensions:
            # Take the average of non-zero parameter values
            return tuple(glue_shape[i] * p[p > 0.].mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                                in dimension_tuples))
        else:
            return tuple(glue_shape[i] * p.mean() for i, p in enumerate(np.array(ptuple) for ptuple
                                                                        in dimension_tuples))

    def dimensions(self):
        """ Continuous tile dimensions

        Returns
        -------
        tuple :
            Tuple of dimensions, typically this will take the form (t, x, y, z) for (3+1)-D spacetime

        Notes
        -----
        Because this is usually a subset of self.parameters, it does not use the property decorator. This method
        is purposed for readability.
        """
        return tuple(getattr(self, d_label) for d_label in self.dimension_labels())

    def shapes(self):
        """ Set of shapes based on discretization parameters and basis.

        Returns
        -------
        tuple :
            Contains shapes of state in all bases, ordered with respect to self.bases() ordering.

        Notes
        -----
        This function is used for operations which require the shape of the state array in a different basis.
        These shapes are defined by the transforms, essentially, but it is wasteful to transform simply for the shape,
        and the amount of boilerplate code to constantly infer the shape justifies this method in most cases.
        """
        return (self.state.shape,)

    def cost_function_gradient(self, eqn, **kwargs):
        """ Gradient of scalar cost functional

        Parameters
        ----------
        eqn : Orbit
            Orbit instance whose state is an evaluation of the governing equations
        kwargs : dict
            extra arguments for rmatvec method.

        Returns
        -------
        gradient :
            Orbit instance whose state contains (dF/dv)^T * F ; (adjoint Jacobian * eqn)

        Notes
        -----
        Withing optimization routines, the eqn orbit is used for other calculations and hence should not be
        recalculated; this is why eqn is passed rather than calculated.

        Default cost functional is 1/2 ||eqn||^2.
        """
        return self.rmatvec(eqn, **kwargs)

    def resize(self, *new_discretization, **kwargs):
        """ Rediscretize the current state

        Parameters
        ----------
        new_discretization : int or tuple of ints
            New discretization size
        kwargs : dict
            keyword arguments for dimension_based_discretization.

        Returns
        -------
        placeholder_orbit :
            Orbit with new discretization; the new shape always refers to the shape in the self.bases()[0] basis.
            Always returned in originating basis.

        Notes
        -----
        # These cases covered by unpacking tuples of length 1.
        If passed as single int x, then new_discretization=(x,): len==1 but type(*new_shape)==int
        If passed as tuple with length one (a,), then new_discretization=((a,),)
        If passed as tuple with length n, then new_discretization=((x,y,...,z),)
        If len >= 2 then could be multiple ints (x,y) or multiple tuples ((a,b), (c,d))
        In other words, they are all tuples, but type checking and unpacking has to be done carefully due to contents.

        All tuples of the form ((x,y,...,z),) are assumed to be redundant representations of (x,y,...,z) and hence is
        unpacked as so.
        """
        # Padding basis assumed to be in the spatiotemporal basis.
        placeholder_orbit = self.copy().transform(to=self.bases()[-1])

        # if nothing passed, then new_shape == () which evaluates to false.
        # The default behavior for this will be to modify the current discretization
        # to a `parameter based discretization'.
        new_shape = new_discretization or self.dimension_based_discretization(self.dimensions(), **kwargs)

        # unpacking unintended nested tuples i.e. ((a, b, ...)) -> (a, b, ...); leaves unnested tuples invariant.
        # New shape must be tuple; i.e. iterable and have __len__
        if len(new_shape) == 1 and isinstance(*new_shape, tuple):
            new_shape = tuple(*new_shape)

        # If the current shape is discretization size (not current shape) differs from shape then resize
        if self.discretization != new_shape:
            # Although this is less efficient than doing every axis at once, it generalizes to cases where bases
            # are different for padding along different dimensions (i.e. transforms implicit in truncate and pad).
            # Changed from iterating over new shape and comparing with old, to iterating over old and comparing
            # with new; this prevents accidentally
            for i, d in enumerate(self.discretization):
                if new_shape[i] < d:
                    placeholder_orbit = placeholder_orbit.truncate(new_shape[i], axis=i)
                elif new_shape[i] > d:
                    placeholder_orbit = placeholder_orbit.pad(new_shape[i], axis=i)
                else:
                    pass

        return placeholder_orbit.transform(to=self.basis)

    def transform(self, to=None):
        """ Method that handles all basis transformations. Undefined for Orbit class.

        Parameters
        ----------
        to : str
            The basis to transform into. If already in said basis, returns self

        Returns
        -------
        Orbit :
            either self or instance in new basis. Returning self and not copying may have unintended consequences
            but typically it would not matter as orbithunter avoids overwrites.
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
        If self.eqn().state = 0. at every point (within some numerical tolerance), then the state constitutes
        a solution to the governing equation. Of course there is no equation for this class, so zeros are returned.
        The instance needs to be in 'spatiotemporal' basis prior to computation; this avoids possible mistakes in the
        optimization process, which would result in a breakdown in performance from redundant transforms.

        Additionally, the equations and state are defined such that state + parameters are required to compute
        the governing equations. Often it is the case that  there will not be an associated component of the equations
        for the parameters themselves. Therefore, because the parameters in the continuous case define the
        spatiotemporal domain (tile), it makes sense for these values to be assigned to the "eqn" orbit. That is,
        the evaluation of the governing equations yields a state defined on the same domain.
        """
        assert self.basis == self.bases()[-1], 'Convert to spatiotemporal basis before computing governing equations.'
        return self.__class__(**{**vars(self), 'state': np.zeros(self.shapes()[-1])})

    def residual(self, eqn=True):
        """ Cost function evaluated at current state.

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 the squared L_2 norm of the spatiotemporal mapping,
            R = 1/2 ||F||^2. The current form generalizes to any equation.

        Notes
        -----
        In certain optimization methods, it is more efficient to have the equations stored, and then take their norm
        as opposed to re-evaluating the equations. The reason why .norm() isn't called instead is to allow for different
        residual functions other than, for instance, the L_2 norm of the DAEs; although in this case there is no
        difference.
        """
        if eqn:
            v = self.transform(to=self.bases()[-1]).eqn().state.ravel()
        else:
            v = self.state.ravel()
        return 0.5 * v.dot(v)

    def matvec(self, other, **kwargs):
        """ Matrix-vector product of Jacobian evaluated at instance state, times orbit_vector of other instance.

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_matvec :
            Orbit with values representative of the matrix-vector product.

        Notes
        -----
        This method represents the matrix-vector product of the Jacobian matrix with an orbit vector of dimension
        self.size+len(self.parameters). Typically for these systems, the Jacobian has dimensions
        [self.size, self.size + len(self.parameters)]. Because there are no associated components for the parameters
        (i.e. the last elements of the orbit vector), it is often convenient to simply pass the current state's
        parameters to the new instance; this philosophy mimics the eqn() method. Because the general Orbit template
        doesn't have an associated equation, return an array of zeros.
        """
        # Instance with all attributes except state and parameters
        return self.__class__(**{**vars(self), 'state': np.zeros(self.shapes()[-1])})

    def rmatvec(self, other, **kwargs):
        """ Matrix-vector product of adjoint Jacobian evaluated at instance state, times state of other instance.

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_rmatvec :
            Orbit with values representative of the adjoint-vector product; this returns a vector of the
            same dimension as self.orbit_vector; i.e. parameter values are returned.
        """
        return self.__class__(**{**vars(self), 'state': np.zeros(self.shape),
                                 'parameters': tuple([0] * len(self.parameter_labels()))})

    def orbit_vector(self):
        """ Vector representation of Orbit instance.

        Returns
        -------
        ndarray :
            The state vector: the current state with parameters appended, returned as a (self.size + n_params , 1)
            axis of dimension 1 for scipy purposes.

        Notes
        -----
        Parameters only need to be bundled into a tuple. There is no requirement that elements must be scalars.
        Any type which can be cast as a numpy array and has numeric values is allowed.
        """
        # By raveling and concatenating ensure that the array is 1-d for the second concatenation; i.e. flatten first.
        parameter_array = np.concatenate(tuple(np.array(p).ravel() for p in self.parameters))
        return np.concatenate((self.state.ravel(), parameter_array), axis=0).reshape(-1, 1)

    def from_numpy_array(self, orbit_vector, **kwargs):
        """ Utility to convert from numpy array (orbit_vector) to Orbit instance for scipy wrappers.

        Parameters
        ----------
        orbit_vector : ndarray
            Vector with (spatiotemporal basis) state values and parameters.

        kwargs :
            parameters : tuple
                If parameters from another Orbit instance are to overwrite the values within the orbit_vector
            parameter_constraints : dict
                constraint dictionary, keys are parameter_labels, values are bools
            Orbit or subclass kwargs : dict
                If special kwargs are required/desired for Orbit instantiation.
        Returns
        -------
        state_orbit : Orbit instance
            Orbit instance whose state and parameters are extracted from the input orbit_vector.

        Notes
        -----
        This function is mainly to retrieve output from (scipy) optimization methods and convert it back into Orbit
        instances. Because constrained parameters are not included in the optimization process, this is not
        as simple as merely slicing the parameters from the array.
        """
        # orbit_vector is defined to be concatenation of state and parameters;
        # slice out the parameters; cast as list to gain access to pop
        param_list = list(kwargs.pop('parameters', orbit_vector.ravel()[self.size:]))

        # The issue with parsing the parameters is that we do not know which list element corresponds to
        # which parameter unless the constraints are checked. Parameter keys which are not in the constraints dict
        # are assumed to be constrained.
        parameters = tuple(param_list.pop(0) if not self.constraints.get(each_label, True) else 0
                           for each_label in self.parameter_labels())
        return self.__class__(**{**vars(self), 'state': np.reshape(orbit_vector.ravel()[:self.size], self.shape),
                                 'parameters': parameters, **kwargs})

    def increment(self, other, step_size=1, **kwargs):
        """ Incrementally add Orbit instances together

        Parameters
        ----------
        other : Orbit
            Represents the values to increment by.
        step_size : float
            Multiplicative other which decides the step length of the correction.

        Returns
        -------
        Orbit :
            New instance incremented by other instance's values. Typically self is the current iterate and
            other is the optimization correction.

        Notes
        -----
        This is used primarily in optimization methods, e.g. adding a gradient descent step using class instances
        instead of simply arrays.
        """
        incremented_params = tuple(self_param + step_size * other_param # assumed to be constrained if 0.
                                   for self_param, other_param in zip(self.parameters, other.parameters))
        return self.__class__(**{**vars(self), **kwargs, 'state': self.state + step_size * other.state,
                                 'parameters': incremented_params})

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
            padding_tuple = tuple((padding_size + 1, padding_size) if i == axis else (0, 0)
                                  for i in range(len(self.shape)))
        else:
            padding_tuple = tuple((padding_size, padding_size) if i == axis else (0, 0)
                                  for i in range(len(self.shape)))

        return self.__class__(**{**vars(self), 'state': np.pad(self.state, padding_tuple)}).transform(to=self.basis)

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
        Orbit
            Orbit instance with smaller discretization.

        Notes
        -----
        The inverse of pad. Default behavior is to simply truncate in current basis in symmetric fashion along
        axis of numpy array specific by 'axis'.
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
        return self.__class__(**{**vars(self), 'state': self.state[truncate_slice]}).transform(to=self.basis)

    def jacobian(self, **kwargs):
        """ Jacobian matrix evaluated at the current state.

        Parameters
        ----------
        kwargs :
            Included in signature for derived classes; no usage here.
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

    def plot(self, show=True, save=False, padding=True, fundamental_domain=True, **kwargs):
        """ Signature for plotting method.
        """
        return None

    def rescale(self, magnitude, method='inf'):
        """ Rescaling of the state in the 'physical' basis per strategy denoted by 'method'
        """
        state = self.transform(to=self.bases()[0]).state
        if method == 'inf':
            # rescale by infinity norm
            rescaled_state = magnitude * state / np.max(np.abs(state.ravel()))
        elif method == 'L1':
            # rescale by L1 norm
            rescaled_state = magnitude * state / np.linalg.norm(state, ord=1)
        elif method == 'L2':
            # rescale by L2
            rescaled_state = magnitude * state / np.linalg.norm(state)
        elif method == 'LP':
            # rescale by L_p norm
            rescaled_state = np.sign(state) * np.abs(state) ** magnitude
        else:
            raise ValueError('Unrecognizable method.')
        return self.__class__(**{**vars(self), 'state': rescaled_state,
                                 'basis': self.bases()[0]}).transform(to=self.basis)

    def to_h5(self, filename=None, dataname=None, h5mode='a', verbose=False, include_residual=False, **kwargs):
        """ Export current state information to HDF5 file

        Parameters
        ----------
        filename : str, default None
            filename to write/append to.
        dataname : str, default 'state'
            Name of the h5_group wherein to store the Orbit in the h5_file at location filename. Should be
            HDF5 group name, i.e. '/A/B/C/...'
        h5mode : str
            Mode with which to open the file. Default is a, read/write if exists, create otherwise,
            other modes ['r+', 'a', 'w-', 'w']. See h5py.File for details. 'r' not allowed, because this is a function
            to write to the file. Defaults to r+ to prevent overwrites.
        verbose : bool
            Whether or not to print save location and group
        include_residual : bool
            Whether or not to include residual as metadata; requires equation to be well-defined for current instance.

        Notes
        -----
        The function could be much cleaner if the onus of responsibility for naming files, groups, datasets was
        put entirely upon the user. To avoid having write gratuitous amounts of code to name orbits, there
        are default options for the filename, groupname and dataname. groupname always acts as a prefix to dataname,
        it defaults to being an empty string. groupname is useful when there is a category of orbits (i.e. a family)
        that have different parameter values.
        """

        with h5py.File(filename or self.filename(extension='.h5'), mode=h5mode) as file:
            # When dataset==None then find the first string of the form orbit_# that is not in the
            # currently opened file. 'orbit' is the first value attempted.
            i = 0
            dataname = dataname or str(i)
            # Combine the group and dataset strings, accounting for possible missing/extra/inconsistent numbers of '/'
            groupname = kwargs.get('groupname', '')
            group_and_dataset = '/'.join(groupname.split('/') + dataname.split('/'))
            while group_and_dataset in file:
                # append _# so that multiple versions with the same name (typically determined by parameters, but
                # could have different values due to decimal truncation).
                try:
                    dataname = str(int(dataname)+1)
                    group_and_dataset = '/'.join(groupname.split('/') + dataname.split('/'))
                except ValueError:
                    group_and_dataset = '/'.join(groupname.split('/')
                                                 + ''.join([dataname.rstrip('/'), '_', str(i)]).split('/'))
                    i += 1

            if verbose:
                print('Writing dataset "{}" to file {}'.format(group_and_dataset, filename))
            # orbitset = file.require_dataset(group_and_dataset, data=self.state, dtype='float64', shape=self.shape)
            # If group already exists, which it typically does, create_group would fail; use require_group instead
            # create_dataset will always overwrite the h5py.Dataset
            orbitset = file.create_dataset(group_and_dataset, data=self.state)
            # Get the attributes that aren't being saved as a dataset. Combine with class name.
            # This is kept as general to allow for others' classes to have arbitrary attributes beyond
            # the defaults: parameters, discretization, shape, basis,
            orbitattributes = {**{k: v for k, v in vars(self).items() if k != 'state'},
                               'class': self.__class__.__name__}
            for key, val in orbitattributes.items():
                # If h5py encounters a dtype which it does not know how to encode (dict, for example), skip it.
                try:
                    orbitset.attrs[key] = val
                except TypeError:
                    continue

            if include_residual:
                # This is included as a conditional statement because it seems strange to make importing/exporting
                # dependent upon full implementation of the governing equations
                try:
                    orbitset.attrs['residual'] = self.residual()
                except (ZeroDivisionError, ValueError):
                    print('Unable to compute residual for {}'.format(repr(self)))

    def filename(self, extension='.h5', decimals=3, cls_name=True):
        """ Method for consistent/conventional filenaming

        Parameters
        ----------
        extension : str
            The extension to append to the filename
        decimals :
            The number of decimals to include in the str name of the orbit.
        cls_name :
            Whether or not to include str(self) in the filename.
        Returns
        -------
        str :
            The conventional filename.

        Notes
        -----
        Many dimensions will yield long filenames.

        Examples
        --------
        For an orbit with t=10, x=5.321 this would yield Orbit_t10p000_x5p321
        """
        if self.dimensions() is not None:
            # Of the form
            dimensional_string = ''.join(['_' + ''.join([self.dimension_labels()[i],
                                                         f'{d:.{decimals}f}'.replace('.', 'p')])
                                          for i, d in enumerate(self.dimensions()) if (d != 0) and (d is not None)])
        else:
            dimensional_string = ''

        if not cls_name and dimensional_string:
            return ''.join([dimensional_string, extension]).lstrip('_')
        else:
            return ''.join([self.__class__.__name__, dimensional_string, extension]).lstrip('_')

    def preprocess(self):
        """ Check the status of a solution, whether or not it converged to the correct orbit type. """
        return self

    def generate(self, attr='all', **kwargs):
        """ Initialize random parameters or state or both.

        Parameters
        ----------
        attr : str
            Takes values 'state', 'parameters' or 'all'.

        Notes
        -----
        Produces a random state and or parameters depending on 'attr' value.
        """
        # It is possible to only generate a subset of the parameters, hence conditional statements are in the method
        if attr in ['all', 'parameters']:
            self._generate_parameters(**kwargs)

        # While there likely isn't a partial initialization of a state
        if attr in ['all', 'state']:
            if self.size == 0. or kwargs.get('overwrite', False):
                self._generate_state(**kwargs)
            else:
                warn_str = '\noverwriting a non-empty state requires overwrite=True. '
                warnings.warn(warn_str, RuntimeWarning)
        # For chaining operations, return self instead of None
        return self

    def to_fundamental_domain(self, **kwargs):
        """ Placeholder/signature for possible symmetry subclasses. """
        return self

    def from_fundamental_domain(self, **kwargs):
        """ Placeholder/signature for possible symmetry subclasses. """
        return self

    def copy(self):
        """ Return an instance with deep copy of numpy array. """
        return self.__class__(**{k: v.copy() if hasattr(v, 'copy') else v for k, v in vars(self).items()})

    def constrain(self, *labels):
        """ Set self constraints based on labels provided.

        Parameters
        ----------
        labels : str or tuple of str

        Returns
        -------
        None

        Notes
        -----
        Does not maintain other constraints when constraining unless they are provided as labels. This
        is to avoid the requirement for an "unconstrain" method. If more constraints are to be maintained then
        just include them in labels.

        By the way that this any other methods are defined, parameters not in the default constraints dict will
        always be assumed to be constrained. This is important for optimization methods. However, having all parameters
        can also be necessary for both calculations and conversion processes. In other words, this is a strict
        enforcement of parameter constraints, hard coded into the default_constraints method. Of course, you can
        have a subclass without a stricter constraints dict.
        """
        if isinstance(labels, str):
            labels = (labels,)
        elif not isinstance(labels, tuple):
            raise TypeError('constraint labels must be str or tuple of str')

        # iterating over constraints items means that constant variables can never be unconstrained by accident.
        constraints = {key: (True if key in tuple(*labels) else False) for key, val in self.constraints.items()}
        setattr(self, 'constraints', constraints)

    def _parse_state(self, state, basis, **kwargs):
        """ Determine state and state shape parameters based on state array and the basis it is in.

        Parameters
        ----------
        state : ndarray
            Numpy array containing state information, can have any number of dimensions.
        basis : str
            The basis that the array 'state' is assumed to be in.
        """
        if isinstance(state, np.ndarray):
            self.state = state
        elif state is None:
            self.state = np.array([], dtype=float).reshape(len(self.default_shape()) * [0])
        else:
            raise ValueError('"state" attribute may only be provided as NumPy array or None.')

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
        Parameters are required to be of numerical type and are typically expected to be scalar values.
        If there are categorical parameters then they should be assigned to a different attribute.
        The reason for this is that in numerical optimization, the orbit_vector;
        the concatenation of self.state and self.parameters is sentf to the various algorithms.
        Cannot send categoricals to these algorithms.

        Default is not to be constrained in any dimension; account for when constraints come from a class with
        fewer parameters by iterated over current labels, then assigning value from passed 'constraints'
        dict. When an expected constraint is not included, the associated parameter is assumed to be constrained.

        # The subclass essentially defines what is and is not a constant; conversion between classes (and passing
        # constraints in the process) can mistakenly unconstrain these constants. Therefore, if the key is not
        # in the default constraints dict then it is assumed to be constant, even when being told otherwise.
        # This is not an issue because this is one of the fundamental requirements built into the subclasses
        # by the user themselves.
        """
        # Get the constraints, making sure to not mistakenly unconstrain constants.
        self.constraints = {k: kwargs.get('constraints', self.default_constraints()).get(k, True)
                            if k in self.default_constraints().keys() else True for k in self.parameter_labels()}
        if parameters is None:
            # None is a valid choice of parameters; it essentially means "generate all parameters" upon generation.
            self.parameters = parameters
        elif isinstance(parameters, tuple):
            # This does not check each tuple element; they can be whatever the user desires, technically.
            # This ensures all parameters are filled.
            if len(self.parameter_labels()) < len(parameters):
                # If more parameters than labels then we do not know what to call them by; truncate by using zip.
                self.parameters = tuple(val for label, val in zip(self.parameter_labels(), parameters))
            else:
                # if more labels than parameters, simply fill with the default missing value, 0.
                self.parameters = tuple(val for label, val in itertools.zip_longest(self.parameter_labels(), parameters,
                                                                          fillvalue=0))
        else:
            # A number of methods require parameters to be an iterable, hence the tuple requirement.
            raise TypeError('"parameters" is required to be a tuple or NoneType. '
                            'singleton parameters need to be cast as tuple (val,).')

    def _generate_parameters(self, **kwargs):
        """ Randomly initialize parameters which are currently zero.

        Parameters
        ----------
        kwargs :
            p_ranges : dict
                keys are parameter_labels, values are uniform sampling intervals or iterables to sample from

        """
        # helper function so comprehension can be used later on; each orbit type typically has a default
        # range of good parameters; however, it is often the case that using a user-defined range is desired.
        def sample_from_generator(val, val_generator, overwrite=False):
            if val == 0 or overwrite or val is None:
                # for numerical parameter generators we're going to use uniform distribution to generate values
                # If the generator is "interval like" then use uniform distribution.
                if isinstance(val_generator, tuple) and len(val_generator) == 2:
                    pmin, pmax = val_generator
                    val = pmin + (pmax - pmin) * np.random.rand()
                # Everything else treated as distribution to sample from
                else:
                    try:
                        val = np.random.choice(val_generator)
                    except (ValueError, TypeError):
                        val = 0

            return val

        # seeding takes a non-trivial amount of time, only set if explicitly provided.
        if isinstance(kwargs.get('seed', None), int):
            np.random.seed(kwargs.get('seed', None))

        # Can be useful to override default sample spaces to get specific cases.
        p_ranges = kwargs.get('parameter_ranges', self.default_parameter_ranges())
        # If *some* of the parameters were initialized, we want to save those values; iterate over the current
        # parameters if not None, else
        parameter_iterable = self.parameters or len(self.parameter_labels()) * [0]
        if len(self.parameter_labels()) < len(parameter_iterable):
            # If more values than labels, then truncate and discard the additional values
            parameters = tuple(sample_from_generator(val, p_ranges.get(label, (0, 0)),
                                                     overwrite=kwargs.get('overwrite', False))
                               for label, val in zip(self.parameter_labels(), parameter_iterable))
        else:
            # If more labels than parameters, fill the missing parameters with default
            parameters = tuple(sample_from_generator(val, p_ranges.get(label, (0, 0)),
                                                     overwrite=kwargs.get('overwrite', False))
                               for label, val in itertools.zip_longest(self.parameter_labels(), parameter_iterable, fillvalue=0))
        setattr(self, 'parameters', parameters)

    def _generate_state(self, **kwargs):
        """ Populate the 'state' attribute

        Parameters
        ----------
        kwargs

        Notes
        -----
        Must generate and set attributes 'state', 'discretization' and 'basis'. The state is required to be a numpy
        array, the discretization is required to be its shape (tuple) in the basis specified by self.bases()[0].
        Discretization is coupled to the state and its specific basis, hence why it is generated here.

        Historically, for the KSe, the strategy to define a specific state was to provide a keyword argument 'spectrum'
        which controlled a spectrum modulation strategy. This is not included in the base signature, because it is
        terminology specific to spectral methods.
        """
        # Just generate a random array; more intricate strategies should be written into subclasses.
        # Using standard normal distribution for values.
        numpy_seed = kwargs.get('seed', None)
        if isinstance(numpy_seed, int):
            np.random.seed(numpy_seed)
        if self.size == 0. or kwargs.get('overwrite', False):
            # Presumed to be in physical basis unless specified otherwise.
            self.discretization = self.dimension_based_discretization(self.parameters, **kwargs)
            self.state = np.random.randn(*self.discretization)
            self.basis = kwargs.get('basis', None) or self.bases()[0]



def convert_class(orbit_, class_generator, **kwargs):
    """ Utility for converting between different classes.

    Parameters
    ----------
    orbit_ : Orbit instance
        The orbit instance to be converted
    class_generator : class generator
        The target class that orbit will be converted to.

    Notes
    -----
    To avoid conflicts with projections onto symmetry invariant subspaces, the orbit is always transformed into the
    physical basis prior to conversion; the instance is returned in the basis of the input, however.

    # Include any and all attributes that might be relevant to the new orbit and those which transfer over from
    # old orbit via usage of vars(orbit_) and kwargs. If for some reason an attribute should not be passed,
    then providing attr=None in the function call is how to handle it.
    """
    # Note any keyword arguments will overwrite the values in vars(orbit_) or state or basis
    return class_generator(**{**vars(orbit_), 'state': orbit_.transform(to=orbit_.bases()[0]).state,
                              'basis': orbit_.bases()[0], **kwargs}).transform(to=orbit_.basis)
