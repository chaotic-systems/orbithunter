from json import dumps
from itertools import zip_longest
import h5py
import numpy as np


__all__ = ["Orbit", "convert_class"]


class Orbit:
    """
    Base class for orbits.

    Parameters
    ----------
    state : ndarray, default None
        If an array, it should contain the state values congruent with the 'basis' argument.
    basis : str, default None
        Which basis the array state is currently in. Must be str if type(state) is np.ndarray
    parameters : tuple, default None
        Parameters required to uniquely define the Orbit.
    discretization : tuple, default None
        The shape of the state array in configuration space, i.e. the 'physical' basis.
    constraints : dict, default None
        Dictionary whose keys are parameter labels, and values are bool. If True, then corresponding parameter
        will be treated as a constant during optimization routines.
    kwargs :
        Extra arguments for _parse_parameters and _parse_state (future/subclass usage only).

    Notes
    -----
    NumPy broadcasting will occur as one would expect if relevant numerical quantities are compatible type and shape.

    The common usage case of constructing instances has the user specifying the state, basis and parameter values.
    For advanced numerical operations, however, it is required to know which parameters are cosntrained, stored in the
    `constraints` attribute, and what the shape of the state array is, prior to any transformations, this is stored in
    the `discretization` attribute. Because of this, :meth:`Orbit.__init__` parses the input and sets the constraints
    using defaults specified by private method `Orbit._default_constraints()`; and also parses the state array for
    its shape in the 'physical' basis. Take note that the discretization attribute is NOT the current shape of the
    state. There are many cases where the current state's shape is not sufficient information; to avoid parsing of this
    type in all future calculations, it is done upon creation of the instance. This parsing takes time; it can and
    should be avoided. To avoid parsing, all primary attributes that would otherwise be parsed need to be passed upon
    creation. For :class:`Orbit` this consists of: state, basis, parameters, discretization, and constraints.
    It is assumed that the information is coherent if it is all being passed by the user; i.e. the discretization does
    in fact correspond to the state array passed.

    Examples
    --------

    Orbit instances can be created in a multitude of ways. Typically it is acceptable to think of Orbits as a
    bundle of a state array and parameters.

    Create an empty Orbit instance.

    >>> orb = Orbit()

    Create an empty Orbit instance, and then use built-in or user prescribed methods to fill its values

    >>> orb = Orbit()
    >>> orb.populate(seed=0) # By default all attributes are populated; seed for reproducibility
    >>> print(repr(orb))
    Orbit({"shape": [2, 2, 2, 2], "basis": "physical", "parameters": [0.549, 0.715, 0.603, 0.545]})

    The attributes can also be specified using values 'all' (default), 'state' and 'parameters' for keyword 'attr'.

    >>> u = Orbit()
    >>> u.populate(attr='parameters', seed=0)
    >>> print(repr(orb))
    Orbit({"shape": [0, 0, 0, 0], "basis": null, "parameters": [0.549, 0.715, 0.603, 0.545]})

    Create and Orbit by providing state and parameter information

    >>> example_state = np.ones([2, 2, 2, 2])
    >>> example_parameters = (16, 16, 16, 16)
    >>> u = Orbit(state=example_state, basis='physical', parameters=example_parameters)

    "Fast" initialization occurs when the five main attributes: 'state', 'basis', 'parameters', 'constraints'
    and 'discretization are provided. If 'state' and 'basis' are provided by 'discretization' is not, then
    'discretization' is parsed from the state array. If 'parameters' is provided but 'constraints' are not, then
    the default constraints are used. These two parsing/argument checking routines occur independently.

    >>> example_state = np.ones([2, 2, 2, 2])
    >>> example_parameters = (16, 16, 16, 16)
    >>> example_constraints = {'t': True, 'x': False, 'y': False, 'z': False}
    >>> example_discretization = example_state.shape
    >>> u = Orbit(state=example_state, basis='physical', parameters=example_parameters,
    ...           constraints=example_constraints, discretization=example_discretization)

    Using Jupyter Lab's cell magic %%timeit to demonstrate. No parsing is much faster on a relative basis.

    >>> %%timeit
    >>> u = Orbit(state=example_state, basis='physical', parameters=example_parameters,
    ...           constraints=example_constraints, discretization=example_discretization)
    1.44 µs ± 97.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    >>> %%timeit
    >>> u = Orbit(state=example_state, basis='physical', parameters=example_parameters)
    10.9 µs ± 36.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    The latter is the more common usage, but when writing functions it is recommended to pass all attributes
    using dictionary unpacking; dictionary merging of the form `{**dict1, **dict2}` can be used to overwrite values of
    `dict1` with new values in `dict2` (those corresponding to the same keys, that is, the operation is otherwise
    a union).

    >>> def example_func_(orbit_instance):
    >>>     new_state_same_shape = ...
    >>>     return orbit_instance.__class__(**{**vars(orbit_instance), 'state': new_state_same_shape})

    """

    def __init__(
        self,
        state=None,
        basis=None,
        parameters=None,
        discretization=None,
        constraints=None,
        **kwargs,
    ):

        if type(None) in [type(state), type(basis), type(discretization)]:
            self._parse_state(state, basis, **kwargs)
        else:
            self.state = state
            self.basis = basis
            self.discretization = discretization

        if type(None) in [type(parameters), type(constraints)]:
            self._parse_parameters(parameters, **kwargs)
        else:
            self.parameters = parameters
            self.constraints = constraints

        # Unused unless implemented for your class, like :meth:`orbithunter.ks.OrbitKS.transform`
        self._workers = kwargs.get("_workers", 1)

    def __add__(self, other):
        """
        Addition of Orbit state and other numerical quantity.

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the sum of current state and `other`

        """
        if issubclass(type(other), Orbit):
            result = self.state + other.state
        else:
            result = self.state + other
        return self.__class__(**{**vars(self), "state": result})

    def __radd__(self, other):
        """
        Addition of Orbit state and other numerical quantity.

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the sum of current state and `other`

        """
        if issubclass(type(other), Orbit):
            result = other.state + self.state
        else:
            result = other + self.state
        return self.__class__(**{**vars(self), "state": result})

    def __sub__(self, other):
        """
        Subtraction of other numerical quantity from Orbit state.

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the subtraction of `other` from current state

        """
        if issubclass(type(other), Orbit):
            result = self.state - other.state
        else:
            result = self.state - other
        return self.__class__(**{**vars(self), "state": result})

    def __rsub__(self, other):
        """
        Subtraction of Orbit state from other numeric quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the subtraction of current state from `other`

        """
        if issubclass(type(other), Orbit):
            result = other.state - self.state
        else:
            result = other - self.state
        return self.__class__(**{**vars(self), "state": result})

    def __mul__(self, other):
        """
        Multiplication of Orbit state and other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the product of current state and `other`


        """
        if issubclass(type(other), Orbit):
            result = np.multiply(self.state, other.state)
        else:
            result = np.multiply(self.state, other)

        return self.__class__(**{**vars(self), "state": result})

    def __rmul__(self, other):
        """
        Multiplication of Orbit state and other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the product of current state and `other`

        """
        if issubclass(type(other), Orbit):
            result = np.multiply(self.state, other.state)
        else:
            result = np.multiply(self.state, other)
        return self.__class__(**{**vars(self), "state": result})

    def __truediv__(self, other):
        """
        Division of Orbit state by other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the division of self.state by other.

        """
        if issubclass(type(other), Orbit):
            result = np.divide(self.state, other.state)
        else:
            result = np.divide(self.state, other)
        return self.__class__(**{**vars(self), "state": result})

    def __floordiv__(self, other):
        """
        Floor division of Orbit state by other numerical quantity

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the floor division of self.state by other.

        Notes
        -----
        If user defined classes are not careful with shapes then accidental outer products can happen (i.e.
        array of shape (x,) // array of shape (x, 1) = array of shape (x, x)

        """
        if issubclass(type(other), Orbit):
            result = np.floor_divide(self.state, other.state)
        else:
            result = np.floor_divide(self.state, other)
        return self.__class__(**{**vars(self), "state": result})

    def __pow__(self, other):
        """
        Exponentiation of Orbit state.

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the exponentiation of self.state by other.

        """
        if issubclass(type(other), Orbit):
            result = self.state ** other.state
        else:
            result = self.state ** other
        return self.__class__(**{**vars(self), "state": result})

    def __mod__(self, other):
        """
        Modulo of Orbit state.

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the self.state modulo other.

        """
        if issubclass(type(other), Orbit):
            result = self.state % other.state
        else:
            result = self.state % other
        return self.__class__(**{**vars(self), "state": result})

    def __iadd__(self, other):
        """
        Inplace addition of Orbit state with `other`

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        Returns
        -------
        Orbit:
            State is the division of self.state by other.

        """
        if issubclass(type(other), Orbit):
            self.state += other.state
        else:
            self.state += other
        return self

    def __isub__(self, other):
        """
        Inplace subtraction of `other` from Orbit state

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state -= other.state
        else:
            self.state -= other
        return self

    def __imul__(self, other):
        """
        Inplace multiplication of Orbit state with `other`

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state *= other.state
        else:
            self.state *= other
        return self

    def __ipow__(self, other):
        """
        Inplace exponentiation of Orbit state by `other`

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state **= other.state
        else:
            self.state **= other
        return self

    def __itruediv__(self, other):
        """
        Inplace division of Orbit state by `other`

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state /= other.state
        else:
            self.state /= other
        return self

    def __ifloordiv__(self, other):
        """
        Inplace floor division of Orbit state by `other`

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state //= other.state
        else:
            self.state //= other
        return self

    def __imod__(self, other):
        """
        In place modulo of Orbit state

        Parameters
        ----------
        other : Orbit, ndarray, float, int

        """
        if issubclass(type(other), Orbit):
            self.state %= other.state
        else:
            self.state %= other
        return self

    def __str__(self):
        """
        String name

        Returns
        -------
        str :
            Of the form 'Orbit'

        """
        return self.__class__.__name__

    def __repr__(self):
        """
        More descriptive representation than __str__ with beautified parameters.

        """
        if self.parameters is not None:
            # parameters should be an iterable
            try:
                pretty_params = tuple(
                    round(x, 3) if isinstance(x, float) else x for x in self.parameters
                )
            except TypeError:
                pretty_params = self.parameters
        else:
            pretty_params = None

        dict_ = {"shape": self.shape, "basis": self.basis, "parameters": pretty_params}
        # convert the dictionary to a string via json.dumps
        dictstr = dumps(dict_)
        return self.__class__.__name__ + "(" + dictstr + ")"

    def __getattr__(self, attr):
        """
        Allows parameters, discretization variables to be retrieved by label directly

        Notes
        -----
        This is setup to allow easy reference of parameters without adding to the Orbit namespace. It
        also increases readability in mathematical statements

        Examples
        --------

        For an Orbit type with parameter labels `( "t", "x", "y", "z")` the spacetime volume could be written

        >>> Orbit.t * Orbit.x * Orbit.y * Orbit.z

        Instead of

        >>> Orbit.parameters[0] *  Orbit.parameters[1] *  Orbit.parameters[2] *  Orbit.parameters[3]

        Although to be fair I would probably just use `np.prod` in this case. This also goes for discretization parameters.

        """
        # hasattr usage mainly
        try:
            attr = str(attr)
        except ValueError("Attribute is not of readable type"):
            ...

        label_index = None
        label_error_type = None
        # I opt for nested try-except statements to be more precise in catching exceptions.
        try:
            # If parameters are NoneType then this raises TypeError; if fewer discretization parameters
            # than expected,  raises IndexError
            if attr in self.parameter_labels():
                label_index = self.parameter_labels().index(attr)
                label_error_type = "parameters"
                return self.parameters[label_index]
            elif attr in self.discretization_labels():
                label_index = self.discretization_labels().index(attr)
                label_error_type = "discretization"
                return self.discretization[label_index]
            else:
                # Trying to access an attribute that does not exist.
                raise AttributeError(f"{str(self)} has no attribute {attr}")

        except TypeError as te:
            raise AttributeError(
                f"Cannot retrieve '{attr}' when {label_error_type} attribute is NoneType"
            ) from te

        except IndexError as ie:
            errstr = " ".join(
                [
                    f"'{attr}' parameter expected at index {label_index} but {str(self)} only has",
                    f"{len(self.parameters)} parameters. This occurs when an unparsed instance receives",
                    f"fewer parameters than the user intended",
                ]
            )
            raise AttributeError(errstr) from ie

    def __getitem__(self, key):
        """
        Slicing of Orbit state and corresponding dimensions

        Parameters
        ----------
        key : slice, tuple, np.ndarray
            Any type compatible with np.ndarray.__getitem__

        Raises
        ------


        Returns
        -------
        Orbit :
            New instance whose state equals self.state[key].

        Notes
        -----
        Because each axis typically represents a continuous dimension, the slicing operation needs to reflect the
        reduction in the dimension of the domain (tile) of the Orbit. Use OrbitKS as an example.
        This has 't', 'x' as its dimensions time and 1-d space, respectively. If a single row (i.e. single time 't') was
        sliced then one would expect self.t == 0. after slicing. Taking this type of effect into account becomes quite
        complicated considering `key` can be combinations of int, Ellipsis, slice, np.ndarray, tuple, np.newaxis, None.
        Therefore, in order to ensure that the dimensionality reduction is well defined, only basic indexing which
        preserves the number of array dimensions should be used. The dimension of any axis whose size is reduced to 1
        is equated to setting dimension = 0, as it has no longer has a notion of length in that dimension

        If the dimension is discrete, then the size along that new axis is simply set as the new dimension; discrete
        dimensions may take a value of 1 but 0 is still disallowed.

        Examples
        --------

        If we initialize and Orbit instance with a state of shape=(8, 8, 8, 8) and parameters=(10, 10, 10, 10)
        (assumed to all be dimensional lengths in this case)

        >>> orbit_instance = Orbit(state=np.ones([8, 8, 8, 8], basis='physical', parameters=(16, 16, 16, 16))

        Then any sliced instance whose state has shape (2, 2, 2, 2) will have dimensions equal to 4.

        >>> sliced_instance = orbit_instance[:2, 2:4, -2:, (0, 1)]
        >>> print(sliced_instance.parameters)
        (4, 4, 4, 4)

        To get a sliced state which maintains `len(self.shape) == self.ndim`, a slice can be used instead of an integer

        >>> sliced_instance = orbit_instance[:2, :2, :2, :1]
        >>> print(sliced_instance.shape)
        (2, 2, 2, 1)

        >>> sliced_instance = orbit_instance[:2, :2, :2, 0]
        ValueError: <bound method Orbit.__getitem__ of Orbit({"shape": [32, 32], "basis": "physical",
        "parameters": [16, 16, 16, 16]})> does not currently support arguments which squeeze, flatten, or otherwise
        reduce the dimension of the state array corresponding to type <class 'orbithunter.core.Orbit'>

        """
        try:
            state_slice = self.state[key]
            # If NumPy advanced indexing flattened the state array, or a single integer index was provided,
            # squeezing the ndarray in the process, need to account for the new dimensions very carefully.
            if len(state_slice.shape) != len(self.shape):
                raise ValueError(
                    "".join(
                        [
                            f"{self.__getitem__} does not currently support arguments ",
                            f"which squeeze, flatten, or otherwise reduce the dimension ",
                            f"of the state array corresponding to type {type(self)}",
                        ]
                    )
                )

            if self.parameters is not None:
                # parameters are passed as tuple, not dict but this is the easiest manner with which to update;
                # need to make sure the new dimensions are in the correct positions with respect to the parameter
                # labels.
                new_dimensions = [
                    dim * (newsize / oldsize)
                    if newsize > 1 and continuous
                    else 0.0
                    if continuous
                    else newsize
                    # If any axes are flattened by the slicing then
                    for dim, newsize, oldsize, continuous in zip(
                        self.dimensions(),
                        state_slice.shape,
                        self.shape,
                        self.continuous_dimensions(),
                    )
                ]
                param_dict = dict(zip(list(self.parameter_labels()), self.parameters))
                dim_dict = dict(zip(list(self.dimension_labels()), new_dimensions))
                param_dict = {**param_dict, **dim_dict}
                parameters = tuple(param_dict[key] for key in self.parameter_labels())
                # Discretization == None forces parsing of the state's shape using the current basis.
                return self.__class__(
                    **{
                        **vars(self),
                        "state": state_slice,
                        "parameters": parameters,
                        "discretization": None,
                    }
                )
            else:
                return self.__class__(
                    **{**vars(self), "state": state_slice, "discretization": None}
                )
        except IndexError as ie:
            if self.size == 0.0:
                raise ValueError(
                    "attempting to slice an Orbit whose state is empty."
                ) from ie
            else:
                raise ValueError()

    @staticmethod
    def bases_labels():
        """
        Labels of the different bases that 'state' attribute can be in.

        Returns
        -------
        tuple :
            tuple of strings which label the possible bases that the state can be in.

        Notes
        -----
        Defaults to single 'physical' basis.

        """
        return ("physical",)

    @staticmethod
    def parameter_labels():
        """
        Strings to use to label dimensions. Generic 3+1 spacetime labels default.

        Returns
        -------
        tuple :
            tuple of strings which label the parameters.

        Notes
        -----
        It might seem idiotic to have both a parameter labels staticmethod and parameters as a tuple; why not just make
        a dict? Because I wanted an immutable type for parameters that could easily be converted into a numpy array.

        """
        return "t", "x", "y", "z"

    @staticmethod
    def dimension_labels():
        """
        Strings to use to label dimensions/periods; typically a subset of parameter_labels.

        Returns
        -------
        tuple :
            tuple of strings which label the dimensions of the current equation

        """
        return "t", "x", "y", "z"

    @staticmethod
    def discretization_labels():
        """
        Strings to use to label discretization variables. Generic 3+1 spacetime labels default.

        Returns
        -------
        tuple :
            tuple of strings which label the discretization variables of the current state.

        """
        return "n", "i", "j", "k"

    @staticmethod
    def _default_shape():
        """
        The default array shape when dimensions are not specified.

        Returns
        -------
        tuple :
            tuple of int for the shape of the discretization when otherwise not determined.

        """
        return 2, 2, 2, 2

    @staticmethod
    def minimal_shape():
        """
        The smallest possible discretization that can be used without methods breaking down.

        Returns
        -------
        tuple of int :
            The minimal shape that the shape can take and still have numerical operations (transforms mainly)
            be compatible

        Notes
        -----
        Often symmetry constraints reduce the dimensionality; if too small this reduction may leave the state empty,
        this method is called in aspect ratio correction and possibly other gluing applications.

        """
        return 1, 1, 1, 1

    @staticmethod
    def minimal_shape_increments():
        """
        The smallest valid increment to change the discretization by.

        Returns
        -------
        tuple of int :
            The smallest valid increments to changes in discretization size to retain all functionality.

        Notes
        -----
        Used in aspect ratio correction and "discretization continuation". For example, the KSe code requires
        even valued field discretizations; therefore the minimum increments for the KSE are 2's.

        """
        return 1, 1, 1, 1

    def periodic_dimensions(self):
        """
        Bools indicating whether or not dimension is periodic for persistent homology calculations.

        Returns
        -------
        tuple of bool :
            Tuple containing flags indicating whether or not a state's dimensions are to be treated as periodic or not.

        Notes
        -----
        Static for base class, however for relative periodic solutions this can be dependent on the frame/slice the
        state is in, and so is left

        """
        return True, True, True, True

    @staticmethod
    def continuous_dimensions():
        """
        Bools indicating whether an array's axes represent continuous dimensions or not.

        Returns
        -------
        tuple of bool :
            Tuple containing flags indicating whether or not a state's dimensions are to be treated as periodic or not.

        Notes
        -----
        Discrete lattice systems would have false for all dimensions; this is necessary for Orbit.__getitem__ mostly.

        """
        return True, True, True, True

    @property
    def shape(self):
        """
        Current state array's shape

        Notes
        -----
        Just a convenience to be able to write self.shape instead of self.state.shape

        """
        return self.state.shape

    @property
    def size(self):
        """
        Current state array's dimensionality

        Notes
        -----
        Just a convenience to be able to write self.size instead of self.state.size

        """
        return self.state.size

    @property
    def ndim(self):
        """
        Current state array's number of dimensions

        Notes
        -----
        Just a convenience to be able to write self.size instead of self.state.ndim

        """
        return self.state.ndim

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = value

    def abs(self):
        """
    Orbit instance with absolute value of state.
        """
        return self.__class__(**{**vars(self), "state": np.abs(self.state)})

    def dot(self, other):
        """
        Return the L_2 inner product of two orbits

        Returns
        -------
        float :
            The value of self * other via L_2 inner product.

        """
        return float(np.dot(self.state.ravel(), other.state.ravel()))

    @classmethod
    def dimension_based_discretization(cls, dimensions, **kwargs):
        """
        Follow orbithunter conventions for discretization size.

        Parameters
        ----------
        dimensions : tuple
            Values from which the discretization may be inferred/calculated.

        kwargs :
            Various flags for defining discretization; can be highly dependent on equation and so
            is left as vague as possible.

        Returns
        -------
        tuple :
            A tuple of ints

        """
        return cls._default_shape()

    @classmethod
    def glue_dimensions(
        cls, dimension_tuples, glue_shape, include_zero_dimensions=True
    ):
        """
        Strategy for combining tile dimensions in gluing; default is arithmetic averaging.

        Parameters
        ----------
        dimension_tuples : tuple of tuples
            A tuple with a number of elements equal to the number of dimensions; each of these dimension-tuples contains
            a number of values equal to the number of orbits in the prospective gluing.
        glue_shape : tuple of ints
            The shape of the gluing being performed i.e. for a 2x2 orbit grid glue_shape would equal (2,2).
        include_zero_dimensions : bool
            If True, then the calculation of average dimensions includes 0's in the averages; else
            they are treated like they do not exist.

        Returns
        -------
        glued_parameters : tuple
            tuple of parameters same dimension and type as self.parameters

        Notes
        -----
        This returns an average of parameter tuples, used exclusively in the gluing method; wherein the new tile
        dimensions needs to be decided upon/inferred from the original tiles. As this average is only a very
        crude approximation, it can be worthwhile to also simply search the parameter space for a combination
        of dimensions which reduces the cost. The strategy produced by this method is simply a baseline; there
        may be way better ways but it is likely highly dependent on equation.

        """
        try:
            if include_zero_dimensions:
                return tuple(
                    glue_shape[i] * p.mean()
                    for i, p in enumerate(
                        np.array(ptuple) for ptuple in dimension_tuples
                    )
                )
            else:
                # Take the average of non-zero parameter values
                return tuple(
                    glue_shape[i] * p[p > 0.0].mean()
                    for i, p in enumerate(
                        np.array(ptuple) for ptuple in dimension_tuples
                    )
                )
        except IndexError as ie:
            raise ValueError(
                f"Gluing shapd must have as many elements as {cls} has dimensions"
            ) from ie

    def dimensions(self):
        """
        Dimensions of the spatiotemporal tile (configuration space).

        Returns
        -------
        tuple :
            Tuple of dimensions, typically this will take the form (t, x, y, z) for (3+1)-D spacetime

        Notes
        -----
        Because this is usually a subset of self.parameters, it does not use the property decorator. This method
        is purposed for readability and other reasons where only dimensions are required.

        """
        return tuple(getattr(self, d_label) for d_label in self.dimension_labels())

    def shapes(self):
        """
        The possible shapes of the current state based on discretization and basis.

        Returns
        -------
        tuple :
            Contains shapes of state in all bases, ordered with respect to self.bases_labels() ordering.

        Notes
        -----
        This function is used for operations which require the shape of the state array in a different basis.
        These shapes are defined by the transforms, essentially, but it is wasteful to transform simply for the shape,
        and the amount of boilerplate code to constantly infer the shape justifies this method in most cases.

        """
        return (self.state.shape,)

    def cost(self, eqn=True):
        """
        Cost function evaluated at current state.

        Returns
        -------
        float :
            The value of the cost function, equal to 1/2 of the squared $L_2$ norm of the spatiotemporal mapping,
            $R = 1/2 F^2$ by default. The current form generalizes to any equation.

        Notes
        -----
        In certain optimization methods, storing evaluations of the governing equation in instances can cut down on the
        number of function calls.

        """
        if eqn:
            v = self.transform(to=self.bases_labels()[-1]).eqn().state.ravel()
        else:
            v = self.state.ravel()
        return 0.5 * v.dot(v)

    def costgrad(self, *args, **kwargs):
        """
        Matrix-vector product corresponding to gradient of scalar cost functional $1/2 F^2$

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
        recalculated; this is why eqn is passed rather than recalculated.

        Default cost functional is $1/2 F^2$.

        """
        if args:
            return self.rmatvec(*args, **kwargs)
        else:
            return self.rmatvec(self.eqn(), **kwargs)

    def costhess(self, **kwargs):
        """
        Matrix-vector product with the Hessian of the cost function.

        Parameters
        ----------
        other : Orbit
            Orbit instance whose state is an evaluation of the governing equations
        kwargs : dict
            extra arguments for rmatvec method.

        Returns
        -------
        hess : np.ndarray
            Hessian matrix of Orbit.cost()

        Notes
        -----

        The hessian is the combination of jacobian-transpose multiplied with jacobian plus a second term, equal
        to the dot product of the hessian of the governing equations (tensor) dotted with the equations, resulting in a
        2-d matrix, $J^TJ + F (d^2F)$. This method has not been implemented for any equation,
        but the recipe is given below. While there are tensor product functions I think the easiest way to
        compute this is by broadcasting and dot product.

        """
        J = self.jacobian(**kwargs)
        return (
            J.T.dot(J)
            + np.tensordot(
                self.eqn(**kwargs).state.ravel(), self.hess(**kwargs), axes=1
            ).squeeze()
        )

    def costhessp(self, other, **kwargs):
        """
        Matrix-vector product with the Hessian of the cost function.

        Parameters
        ----------
        other : Orbit
            Orbit instance whose state is an evaluation of the governing equations
        kwargs : dict
            extra arguments for rmatvec method.

        Returns
        -------
        hess : np.ndarray
            Hessian matrix of Orbit.cost()

        Notes
        -----

        The hessian is the combination of jacobian-transpose multiplied with jacobian plus a second term, equal
        to the dot product of the hessian of the governing equations (tensor) dotted with the equations, resulting in a
        2-d matrix. This method has not been implemented for any equation,
        but the recipe is given below. While there are tensor product functions I think the easiest way to
        compute this is by broadcasting and dot product.

        """
        # rmatvec of anything produces components for entire orbit_vector, meaning that simply adding the
        # two results is insufficient; need to parameter-wise add as well; can do this via orbit_vector or increment.
        JTJ = self.rmatvec(self.matvec(other, **kwargs), **kwargs)

        # equal to (J^T J * v) + (F  * d^2F * v)
        # (d^2F) has dimensions (self.eqn().size, self.cdof().size, self.cdof().size)
        return JTJ.increment(self.hessp(self.eqn(**kwargs), other, **kwargs))

    def resize(self, *new_discretization, **kwargs):
        """
        Rediscretize the current state typically via zero padding or interpolation.

        Parameters
        ----------
        new_discretization : int or tuple of ints
            New discretization size
        kwargs : dict
            keyword arguments for dimension_based_discretization.

        Returns
        -------
        placeholder_orbit :
            Orbit with new discretization; the new shape always refers to the shape in the self.bases_labels()[0] basis.
            Always returned in originating basis.

        Notes
        -----
        Description of how different number/types of args are handled:
        If passed as single int x, then new_discretization=(x,): len==1 but type(new_shape)==int
        If passed as tuple with length one (a,), then new_discretization=((a,),)
        If passed as tuple with length n, then new_discretization=((x,y,...,z),)
        If len >= 2 then could be multiple ints (x,y) or multiple tuples ((a,b), (c,d))
        In other words, they are all tuples, but type checking and unpacking has to be done carefully due to contents.
        All tuples of the form ((x,y,...,z),) are assumed to be redundant representations of (x,y,...,z)

        """
        # Padding basis assumed to be in the spatiotemporal basis.
        placeholder_orbit = self.copy().transform(to=self.bases_labels()[-1])
        new_shape = new_discretization or self.dimension_based_discretization(
            self.dimensions(), **kwargs
        )
        if len(new_shape) == 1 and isinstance(*new_shape, tuple):
            new_shape = tuple(*new_shape)

        # If the current shape is discretization size (not current shape) differs from shape then resize
        if self.discretization != new_shape:
            # Although this is less efficient than doing every axis at once, it generalizes to cases where bases
            # are different for padding along different dimensions (i.e. transforms implicit in truncate and pad).
            for ax, (old, new, min_size) in enumerate(
                zip(self.discretization, new_shape, self.minimal_shape())
            ):
                if new < min_size:
                    errstr = (
                        "minimum discretization requirements not met during resize."
                    )
                    raise ValueError(errstr)
                if new < old:
                    placeholder_orbit = placeholder_orbit._truncate(new, axis=ax)
                elif new > old:
                    placeholder_orbit = placeholder_orbit._pad(new, axis=ax)
                else:
                    pass

        return placeholder_orbit.transform(to=self.basis)

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
        If provided tuples of ints, shift and axis need to be the same length as to be coherent.

        """
        return self.__class__(
            **{**vars(self), "state": np.roll(self.state, shift, axis=axis)}
        )

    def cell_shift(self, n_cell, axis=0):
        """
        Rotate by period/n_cell in either axis.

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
        return self.roll(
            np.sign(n_cell)
            * np.array(self.discretization)[np.array(axis)]
            // np.abs(n_cell),
            axis=axis,
        )

    def concat(self, other, axis=0):
        """
        Join two Orbits together by concatenating their state arrays and adding/averaging their parameters.

        Parameters
        ----------
        other : Orbit
            Orbit to be joined with.
        axis : int
            The axis along which to concatenate the state arrays.

        Returns
        -------
        Orbit :
            An instance whose stats have been concatenated, and whose parameters have been summed along that dimension.

        Notes
        -----
        To cause the least amount of distortion, the grid spacings of each state should be approximately equivalent
        (if discretizations of continuous dimension). Not allowing mapping from fundamental domains because for
        discrete symmetries those will often call concat; avoid all possible recursion errors this way.

        .. warning::
           `self.concat(other)` is only equivalent to `other.concat(self)` if `other` and `self` both have parameters
           defined, and the parameters not corresponding to dimensions are the same (i.e. categorical parameters).
           When this is not the case, the parameters of the instance calling the method are used: `self.concat(other)`
           will use `self`'s categoricals.

        .. warning::
           This function does not check the bases which each Orbit state is in.

        """
        concatenated_state = np.concatenate((self.state, other.state), axis=axis)
        if self.parameters is not None and other.parameters is not None:
            tuple_of_zipped_dimensions = tuple(
                zip(*(o.dimensions() for o in (self, other)))
            )
            # "glue shape"
            glue_shape = tuple(2 if i == axis else 1 for i in range(self.ndim))
            new_dimensions = self.glue_dimensions(
                tuple_of_zipped_dimensions, glue_shape, include_zero_dimensions=True
            )

        elif self.parameters is not None:
            new_dimensions = [
                dim * (newsize / oldsize)
                if newsize > 1 and continuous
                else 0.0
                if continuous
                else newsize
                # If any axes are flattened by the slicing then
                for dim, newsize, oldsize, continuous in zip(
                    self.dimensions(),
                    concatenated_state.shape,
                    self.shape,
                    self.continuous_dimensions(),
                )
            ]
        elif other.parameters is not None:
            new_dimensions = [
                dim * (newsize / oldsize)
                if newsize > 1 and continuous
                else 0.0
                if continuous
                else newsize
                # If any axes are flattened by the slicing then
                for dim, newsize, oldsize, continuous in zip(
                    self.dimensions(),
                    concatenated_state.shape,
                    self.shape,
                    self.continuous_dimensions(),
                )
            ]

        else:
            new_dimensions = None

        if new_dimensions is not None:
            # This handles the order of the parameters as given by the staticmethod :meth:`Orbit.parameter_labels`
            param_dict = dict(zip(list(self.parameter_labels()), self.parameters))
            dim_dict = dict(zip(list(self.dimension_labels()), new_dimensions))
            param_dict = {**param_dict, **dim_dict}
            parameters = tuple(param_dict[key] for key in self.parameter_labels())
            # setting discretization to None forces state shape parsing to occur.
            concat_orbit = self.__class__(
                **{
                    **vars(self),
                    "state": concatenated_state,
                    "parameters": parameters,
                    "discretization": None,
                }
            )
        else:
            # If no parameters were in either orbit, then parse then
            concat_orbit = self.__class__(
                **{**vars(self), "state": concatenated_state, "discretization": None}
            )

        return concat_orbit

    def reflection(self, axis=0, signed=True):
        """
        Reflect the velocity field about the spatial midpoint

        Parameters
        ----------

        axis : int, tuple
            The NumPy axes to reflect over; reflection occurs with respect to the 'middle' of each dimension.
        signed : bool
            Multiply by -1 after reflection.

        Returns
        -------
        OrbitKS :
            OrbitKS has been reflected over axis (may be tuple of axes).

        Notes
        -----
        In certain numerical implementations a call to np.roll may be required when discretization has even number of
        points.

        """

        if signed:
            reflected_field = -1 * np.flip(self.state, axis=axis)
        else:
            reflected_field = np.flip(self.state, axis=axis)

        return self.__class__(**{**vars(self), "state": reflected_field})

    def transform(self, to=None, **kwargs):
        """
        Method that handles all basis transformations. Undefined/trivial for Orbits with only one basis.

        Parameters
        ----------
        to : str
            The basis to transform into. If already in said basis, returns self (not a copy!)

        kwargs : dict
            Allows for keyword arguments to be passed for various scientific computing packages' transform methods.
            The first to come to mind is SciPy's FFT's 'workers' keyword, which enables parallelization.

        Returns
        -------
        Orbit :
            either self or instance in new basis. Returning self and not copying may have unintended consequences
            but typically it would not matter as orbithunter operations typically require copying numpy arrays.

        Notes
        -----
        Has no purpose for classes with singular basis. Convention is to return self (not a copy) if 'to' keyword
        argument equals self.basis.

        """
        return self

    def eqn(self, *args, **kwargs):
        """
        Return an instance whose state is an evaluation of the governing equations.

        Returns
        -------
        Orbit :
            Orbit instance whose state equals evaluation of governing equation.

        Notes
        -----
        If self.eqn().state = 0. at every point (within some numerical tolerance), then the state constitutes
        a solution to the governing equation. The instance needs to be in 'spatiotemporal' basis prior to computation;
        this avoids possible mistakes in the optimization process, which would result in a breakdown
        in performance from redundant transforms. If you do not like this choice then overwrite this method.

        Additionally, the equations and state are defined such that state + parameters are required to compute
        the governing equations. Often it is the case that  there will not be an associated component of the equations
        for the parameters themselves. Therefore, because the parameters in the continuous case define the
        spatiotemporal domain (tile), it makes sense for these values to be assigned to the "eqn" orbit. That is,
        the evaluation of the governing equations yields a state defined on the same domain.

        .. warning::
           If equation has components for any parameters:
           However, if the equations of motion have components for the parameters; this will in correctly overwrite
           the components stored in parameters and this will need an overwrite.

        """
        assert (
            self.basis == self.bases_labels()[-1]
        ), "Convert to spatiotemporal basis before computing governing equations."
        return self.__class__(**{**vars(self), "state": np.zeros(self.shapes()[-1])})

    def matvec(self, other, **kwargs):
        """
        Matrix-vector product of Jacobian and orbit_vector from other instance.

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
        [self.size, self.size + len(self.parameters) - len(self.constants)]. Because there are no associated components for the parameters
        (i.e. the last elements of the orbit vector), it is often convenient to simply pass the current state's
        parameters to the new instance; this philosophy mimics the eqn() method. Because the general Orbit template
        doesn't have an associated equation, return an array of zeros for its state.

        Assumed to be in 'spatiotemporal' basis. Equals J * dx
        """
        # Instance with all attributes except state and parameters
        return self.__class__(**{**vars(self), "state": np.zeros(self.shapes()[-1])})

    def rmatvec(self, other, **kwargs):
        """
        Matrix-vector product of adjoint Jacobian and state from `other` instance.

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_rmatvec :

            Orbit with values representative of the adjoint-vector product.

        Notes
        -----
        This method represents the matrix-vector product of the adjoint Jacobian matrix with an orbit state of dimension
        self.size. Typically for these systems, the adjoint Jacobian has dimensions
        [self.size + len(self.parameters), self.size]; this returns a vector of the same dimension as self.orbit_vector;
        i.e. this *does* produce components corresponding to the parameters.

        Creation and usage of _rmatvec_parameters motivated by generalizability/readability of the code.

        """
        params = self._rmatvec_parameters(other)
        return self.__class__(
            **{**vars(self), "state": np.zeros(self.shape), "parameters": params,}
        )

    def _rmatvec_parameters(self, other):
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

        Notes
        -----
        Would generally have elements equal to dot product (dF/dP)^T * other. This method isn't required but
        I personally found it cleaner to do the parameter components separately then return them in rmatvec method.
        """
        return tuple(
            0.0 if not constr else 0.0
            for p, constr in zip(self.parameters, self.constraints.values())
        )

    def hess(self, **kwargs):
        """
        Matrix of second derivatives of the governing equations.

        Returns
        -------
        np.ndarray :
            The tensor of second derivatives of the governing equations. Will typically have dimensions equal
            to (self.orbit_vector.size, self.orbit_vector.size, self.size).

        Notes
        -----

        The components of the tensor $H_{kj}$ is the vector of derivatives with respect to $x_k, x_j$
        of equation component $F_i$. The governing equations will typically have vector valued output;
        in general the matrix of second derivatives of each component is a tensor.

        What is returned is the rank 3 tensor with correct shape for a vector valued equation F(x) where x
        has N degrees of freedom but F only has d dimensions.

        """
        cdof = self.cdof().size
        return np.zeros([self.eqn().size, cdof, cdof])

    def hessp(self, left_other, right_other, **kwargs):
        """
        Tensor product u * H * v where H is the matrix of second derivatives of governing equations.

        Parameters
        ----------
        left_other : Orbit
            Orbit whose state is to be multiplied with the Hessian on the LEFT
        right_other : Orbit
            Orbit whose orbit_vector is to be multiplied with the Hessian on the RIGHT
        kwargs : dict
            Any keywords relevant for the tensor-free evaluation of the Hessian product.

        Returns
        -------
        Orbit :
            Orbit whose state and parameters are the product of :meth:`~orbithunter.core.Orbit.orbit_vector` with
            matrix of second derivatives.

        Notes
        -----
        Requires an equation to define, just like rmatvec and matvec; return zeros because no equation here.
        Tensor multiplication can be written u * d^2 F * v, dimensions of each component are: u=(N,)
        d^2F = (N, N+d-c, N+d-c), v = (N+d-c, 1); therefore it returns a vector of dimension (N+d-c, 1)
        this orbit_vector is then returned as an Orbit instance. N=number of state variables, p=parameters, c=constants

        """
        u = left_other.state.ravel()
        v = right_other.cdof()
        # This makes it look simple but the expression np.zeros(u.size) is acting as a placeholder.
        return self.from_numpy_array(np.tensordot(u, np.zeros(u.size), axes=1).dot(v))

    def cdof(self):
        """
        Computational degrees of freedom; equivalent to orbit_vector sans constant parameters.

        Returns
        -------
        ndarray :
            Vector with the current state with parameters appended, returned as a (self.size + n_params - n_constants, 1)
            axis of dimension 1 for scipy purposes.

        Notes
        -----
        Instead of having to decide whether or not to pass parameters or subset thereof, this method simply acts as
        a wrapper for applying constraints to the current parameters.

        """
        # By raveling and concatenating ensure that the array is 1-d for the second concatenation; i.e. flatten first.
        parameters = self.parameters
        if parameters:
            variables = tuple(
                p.ravel() if isinstance(p, np.ndarray) else p
                for p, const in zip(parameters, self.constraints.values())
                if not const
            )
        else:
            variables = ()
        return np.concatenate((self.state.ravel(), variables), axis=0).reshape(-1, 1)

    def constants(self):
        """
        Constant parameters; convenience function for optimization methods.

        Returns
        -------
        tuple :
            Parameters that are either always constant or have been constrained but still required for optimization.

        """
        parameters = self.parameters
        if parameters:
            constants = tuple(
                p.ravel() if isinstance(p, np.ndarray) else p
                for p, const in zip(parameters, self.constraints.values())
                if const
            )
        else:
            constants = ()
        return constants

    def orbit_vector(self):
        """
        Vector representation of Orbit instance; constants all variables required to define the Orbit instance.

        Returns
        -------
        ndarray :
            The state vector: the current state with parameters appended, returned as a (self.size + n_params , 1)
            axis of dimension 1 for scipy purposes.

        Notes
        -----
        Parameters only need to be bundled into a tuple. There is no requirement that elements must be scalars.
        Any type which can be cast as a numpy array and has numeric values is allowed, technically, but scalars are
        recommended. If non-scalar parameters are used, user will need to overwrite the Orbit.from_numpy_array() method.

        """
        # By raveling and concatenating ensure that the array is 1-d for the second concatenation; i.e. flatten first.

        parameters = self.parameters
        if parameters:
            variables = tuple(
                p.ravel() if isinstance(p, np.ndarray) else p for p in self.parameters
            )
        else:
            variables = ()
        return np.concatenate((self.state.ravel(), variables), axis=0).reshape(-1, 1)

    def from_numpy_array(self, cdof, *args, **kwargs):
        """
        Utility to convert from numpy array (orbit_vector) to Orbit instance for scipy wrappers.

        Parameters
        ----------
        cdof : ndarray
            Vector with (spatiotemporal basis) state values and parameters.

        kwargs :
            parameters : tuple
                If parameters from another Orbit instance are provided, overwrite the values within the orbit_vector
            parameter_constraints : dict
                constraint dictionary, keys are parameter_labels, values are bools
            subclass kwargs : dict
                If special kwargs are required/desired for Orbit instantiation.
        Returns
        -------
        Orbit instance :
            Orbit instance whose state and parameters are extracted from the input orbit_vector.

        Notes
        -----
        This function is mainly to retrieve output from (scipy) optimization methods and convert it back into Orbit
        instances. Because constrained parameters are not included in the optimization process, this is not
        as simple as merely slicing the parameters from the array, as the order of the elements is determined by
        the constraints. If non-scalar parameters are used, user will need to overwrite the Orbit.from_numpy_array() method.

        In order to completely specific the equations in the presence of constraints it is necessary to

        .. warning::
           If equation has components for any parameters:
           However, if the equations of motion have components for the parameters; this will in correctly overwrite
           the components stored in parameters and this will need an overwrite.

        """
        # The parameters and possible constants are expected to be ordered
        parameters_list = list(cdof.ravel()[self.size :])
        constants_list = list(args)
        # The issue with parsing the parameters is that we do not know which list element corresponds to
        # which parameter unless the constraints are checked. Parameter keys which are not in the constraints dict
        # are assumed to be constrained. Pop from param_list if parameters 1. exist, 2. are unconstrained.
        # Not having enough parameters to pop means something is going wrong in your matvec/rmatvec functions typically.
        if self.parameters is not None:
            parameters = tuple(
                parameters_list.pop(0)
                if (
                    not self.constraints.get(each_label, True)
                    and len(parameters_list) > 0
                )
                else constants_list.pop(0)
                if len(constants_list) > 0.0
                else 0.0
                for each_label in self.parameter_labels()
            )
        else:
            parameters = None
        return self.__class__(
            **{
                **vars(self),
                "state": np.reshape(cdof.ravel()[: self.size], self.shape),
                "parameters": parameters,
                **kwargs,
            }
        )

    def increment(self, other, step_size=1, **kwargs):
        """
        Incrementally add Orbit instances together

        Parameters
        ----------
        other : Orbit
            Represents the values to increment by.
        step_size : float
            Multiplicative factor for step size.

        Returns
        -------
        Orbit :
            New instance incremented by other instance's values.

        Notes
        -----
        Typically when this method is called, self is the current iterate and other is an optimization correction.

        """
        if self.parameters is not None:
            incremented_params = tuple(
                self_param + step_size * other_param  # assumed to be constrained if 0.
                for self_param, other_param in zip(self.parameters, other.parameters)
            )
        else:
            incremented_params = None

        return self.__class__(
            **{
                **vars(self),
                **kwargs,
                "state": self.state + step_size * other.state,
                "parameters": incremented_params,
            }
        )

    def jacobian(self, **kwargs):
        """
        Jacobian matrix evaluated at the current state.

        Parameters
        ----------
        kwargs :
            Included in signature for derived classes; no usage here.

        Returns
        -------
        np.ndarray :
            2-d numpy array equalling the Jacobian matrix of the governing equations evaluated at current state.

        """
        return np.zeros([self.size, self.cdof().size])

    def norm(self, order=None):
        """
        Norm of spatiotemporal state via numpy.linalg.norm

        """
        return np.linalg.norm(self.state.ravel(), ord=order)

    def plot(
        self, show=True, save=False, padding=False, fundamental_domain=False, **kwargs
    ):
        """
        Visualization method is equation/dimension dependent.

        """
        ...

    def rescale(self, magnitude, method="inf"):
        """
        Rescaling of the state in the 'physical' basis per strategy denoted by 'method'

        """
        state = self.transform(to=self.bases_labels()[0]).state
        if method == "inf":
            # rescale by infinity norm
            rescaled_state = magnitude * state / np.max(np.abs(state.ravel()))
        elif method == "L1":
            # rescale by L1 norm
            rescaled_state = magnitude * state / np.linalg.norm(state, ord=1)
        elif method == "L2":
            # rescale by L2
            rescaled_state = magnitude * state / np.linalg.norm(state)
        elif method == "LP":
            # rescale by L_p norm
            rescaled_state = np.sign(state) * np.abs(state) ** magnitude
        else:
            raise ValueError("Unrecognizable method.")
        return self.__class__(
            **{**vars(self), "state": rescaled_state, "basis": self.bases_labels()[0]}
        ).transform(to=self.basis)

    def to_h5(
        self,
        filename="",
        groupname="",
        dataname="",
        h5mode="a",
        verbose=False,
        include_cost=False,
        **kwargs,
    ):
        """
        Export current state information to HDF5 file

        Parameters
        ----------
        filename : str, default None
            filename to write/append to.
        groupname : str
            The name for a h5py.Group to save under. This is included to make hierarchical saving easier.
        dataname : str, default '0'
            Name of the h5py.Dataset wherein to store the Orbit in the h5_file at location filename. Should be
            HDF5 group name, i.e. '/A/B/C/...'
        h5mode : str
            Mode with which to open the file. Default is a, read/write if exists, create otherwise,
            other modes ['r+', 'a', 'w-', 'w']. See h5py.File for details. 'r' not allowed, because this is a function
            to write to the file.
        verbose : bool
            Whether or not to print save location and group
        include_cost : bool
            Whether or not to include cost as metadata; requires equation to be well-defined for current instance.
        kwargs : dict
            extra keyword arguments, in signature to allow for generalization

        Notes
        -----
        The mode 'a' would typically overwrite datasets but this is handled here by adding suffixes. This allows
        for storing multiple versions of data within the same file without overwriting the old data/creating a new
        file.

        The function could be much cleaner if the onus of responsibility for naming files, groups, datasets was
        put entirely upon the user. To avoid having write gratuitous amounts of code to name orbits, there
        are default options for the filename, groupname and dataname. groupname always acts as a prefix to dataname,
        it defaults to being an empty string. groupname is useful when there is a category of orbits (i.e. a family).

        """
        with h5py.File(filename or self.filename(extension=".h5"), mode=h5mode) as file:
            # When dataset==None then find the first string of the form orbit_# that is not in the
            # currently opened file. 'orbit' is the first value attempted.
            i = 0
            dataname = dataname or str(i)
            # Combine the group and dataset strings, accounting for possible missing/extra/inconsistent numbers of '/'
            group_and_dataset = "/".join(groupname.split("/") + dataname.split("/"))
            while group_and_dataset in file:
                # append _# so that multiple versions with the same name (typically determined by parameters, but
                # could have different values due to decimal truncation).
                try:
                    dataname = str(int(dataname) + 1)
                    group_and_dataset = "/".join(
                        groupname.split("/") + dataname.split("/")
                    )
                except ValueError:
                    group_and_dataset = "/".join(
                        groupname.split("/")
                        + "".join([dataname.rstrip("/"), "_", str(i)]).split("/")
                    )
                    i += 1

            if verbose:
                print(
                    'Writing dataset "{}" to file {}'.format(
                        group_and_dataset, filename
                    )
                )
            orbitset = file.create_dataset(group_and_dataset, data=self.state)
            # Get the attributes that aren't being saved as a dataset. Include class name so class can be parsed
            # upon import.
            orbitattributes = {
                **{k: v for k, v in vars(self).items() if k != "state"},
                "class": self.__class__.__name__,
            }
            for key, val in orbitattributes.items():
                # If h5py encounters a dtype which it does not know how to encode (dict, for example), skip it.
                try:
                    orbitset.attrs[key] = val
                except TypeError:
                    continue

            if include_cost:
                # This is included as a conditional statement because it seems strange to make importing/exporting
                # dependent upon full implementation of the governing equations
                try:
                    orbitset.attrs["cost"] = self.cost()
                except (ZeroDivisionError, ValueError, AttributeError):
                    print(
                        f"Unable to compute cost for instance {repr(self)}; data will not be saved to .h5 file."
                    )

    def filename(self, extension=".h5", decimals=3, cls_name=True):
        """
        Method for convenience and consistent/conventional file naming.

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
        More dimensions means longer filenames.

        Examples
        --------
        For an orbit with t=10, x=5.321 this would yield Orbit_t10p000_x5p321

        """
        if self.dimensions() is not None:
            # Of the form
            dimensional_string = "".join(
                [
                    "_"
                    + "".join(
                        [
                            self.dimension_labels()[i],
                            f"{d:.{decimals}f}".replace(".", "p"),
                        ]
                    )
                    for i, d in enumerate(self.dimensions())
                    if (d != 0) and (d is not None)
                ]
            )
        else:
            dimensional_string = ""

        if not cls_name and dimensional_string:
            return "".join([dimensional_string, extension]).lstrip("_")
        else:
            return "".join(
                [self.__class__.__name__, dimensional_string, extension]
            ).lstrip("_")

    def preprocess(self):
        """
        Check the "status" of a solution

        Notes
        -----
        This method is used to check to see, for example, if an orbit has converged to an equilibrium but is still
        being stored in a different Orbit class. It is beneficial to check and see whether orbit states can be
        "simplified".

        """
        return self

    def populate(self, attr="all", **kwargs):
        """
        Initialize random parameters or state or both.

        Parameters
        ----------
        attr : str
            Takes values 'state', 'parameters' or 'all'.

        Notes
        -----
        Produces a random state and or parameters depending on 'attr' value.

        """
        # Parameter generation only overwrites default-valued parameters. If random parameters are desired
        # then simply do not provide 'parameters' upon initialization.
        if attr in ["all", "parameters"]:
            self._populate_parameters(**kwargs)

        # State generation is typically an all-or-nothing process; therefore to prevent accidental overwrites an
        # extra condition is required to be satisfied.
        if attr in ["all", "state"]:
            if self.size == 0.0 or kwargs.get("overwrite", False):
                self._populate_state(**kwargs)
            else:
                error_str = " overwriting a non-empty state requires overwrite=True. "
                raise ValueError(error_str)
        # For chaining operations, return self instead of None
        return self

    def to_fundamental_domain(self, **kwargs):
        """
        Placeholder/signature for possible symmetry subclasses.

        """
        return self

    def from_fundamental_domain(self, **kwargs):
        """
        Placeholder/signature for possible symmetry subclasses.

        """
        return self

    def copy(self):
        """
        Return an instance with copies of copy-able attributes.

         """
        return self.__class__(
            **{k: v.copy() if hasattr(v, "copy") else v for k, v in vars(self).items()}
        )

    def constrain(self, *labels):
        """
        Set self constraints based on labels provided.

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

        Parameters not in the default constraints dict will always be assumed to be constrained.
        This is important for optimization methods, in the instance where you have constant parameters required
        to evaluation the governing equations.

        """
        # If string, package as tuple so 'in' can be used for comparison
        if isinstance(labels, str):
            labels = (labels,)
        # If a tuple then there are two options; multiple strings were passed or an actual tuple was passed.
        # Unpacking a string will yield a tuple of char which results in unintended behavior.
        elif isinstance(labels, tuple) and False not in tuple(
            isinstance(lab, str) for lab in labels
        ):
            pass
        elif isinstance(labels, tuple) and False not in tuple(
            isinstance(lab, tuple) for lab in labels
        ):
            labels = tuple(*labels)
        else:
            raise TypeError(
                "constraint labels must be str, tuple of str, or tuple of tuples"
            )

        # iterating over constraints items means that constant variables can never be unconstrained by accident.
        constraints = {
            key: True if key in labels else self._default_constraints().get(key, True)
            for key in self.parameter_labels()
        }
        setattr(self, "constraints", constraints)

    def mask(self, masking_array, invert=False, **kwargs):
        """
        Return an Orbit instance with a numpy masked array state

        Parameters
        ----------
        masking_array : np.ndarray
        invert : bool
            Whether or not to take the inverse mask.

        Returns
        -------

        Notes
        -----
        Typically used for plotting shadowing and clipping results.

        """
        if invert:
            # Sometimes shadowing results are returned as int
            masked_state = np.ma.masked_array(
                self.state, mask=np.invert(masking_array.astype(bool)), **kwargs
            )
        else:
            masked_state = np.ma.masked_array(
                self.state, mask=masking_array.astype(bool), **kwargs
            )
        return self.__class__(**{**vars(self), "state": masked_state})

    def plotting_dimensions(self):
        """
        Return the dimension intervals displayed in visualization techniques; only used in clipping function.

        Returns
        -------
        tuple of tuple :
            return the intervals of the dimensions; default is from 0 to period

        """
        return ((0, dim) for dim in self.dimensions())

    @classmethod
    def defaults(cls):
        """
        Dict of default values for constraints, parameter ranges, sizes, etc.

        Returns
        -------
        orbit_defaults_dictionary : dict
            Dictionary with three keys that indicate the default parameter ranges and shapes used in Orbit.populate()
            as well as the default parameter constraints.

        Notes
        -----
        More defaults can be included in subclassing

        """
        orbit_defaults_dictionary = {
            "parameter_ranges": cls._default_parameter_ranges(),
            "shape": cls._default_shape(),
            "constraints": cls._default_constraints(),
        }
        return orbit_defaults_dictionary

    @classmethod
    def _dimension_indexing_order(cls):
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

        return True, True, True, True

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Intervals (continuous) or iterables (discrete) used to populate parameters.

        Notes
        -----
        tuples or length two are *always* interpreted to be continuous intervals. If you have a discrete variable
        with two options, simply use a list instead of tuple. Discrete variables are populated by using random choice
        from the provided collection.

        """
        return {p_label: (0, 1) for p_label in cls.parameter_labels()}

    @classmethod
    def _default_constraints(self):
        """
        Sometimes parameters are necessary but constant; this allows for exclusion from optimization without hassle.

        Returns
        -------
        dict :
            Keys are parameter labels, values are bools indicating whether or not a parameter is constrained.

        """
        return {k: False for k in self.parameter_labels()}

    def _pad(self, size, axis=0, mode="constant", **kwargs):
        """
        Increase the size of the discretization along an axis.

        Parameters
        ----------
        size : int
            The new size of the discretization (not the current shape), restrictions typically imposed by equations.
        axis : int
            Axis to pad along per numpy conventions.

        Returns
        -------
        Orbit :
            Orbit instance whose state in the physical along numpy axis 'axis' basis has a number of discretization
            points equal to 'size', (self.bases_labels()[0][axis] == size after method call).

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
            padding_tuple = tuple(
                (padding_size + 1, padding_size) if i == axis else (0, 0)
                for i in range(len(self.shape))
            )
        else:
            padding_tuple = tuple(
                (padding_size, padding_size) if i == axis else (0, 0)
                for i in range(len(self.shape))
            )
        newdisc = tuple(
            size if i == axis else self.discretization[i]
            for i in range(len(self.discretization))
        )
        return self.__class__(
            **{
                **vars(self),
                "state": np.pad(self.state, padding_tuple, mode=mode, **kwargs),
                "discretization": newdisc,
            }
        ).transform(to=self.basis)

    def _truncate(self, size, axis=0):
        """
        Decrease the size of the discretization along an axis

        Parameters
        -----------
        size : int
            The new size of the discretization.
        axis : int
            Axis along which truncation occurs.

        Returns
        -------
        Orbit :
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
            truncate_slice = tuple(
                slice(truncate_size + 1, -truncate_size) if i == axis else slice(None)
                for i in range(len(self.shape))
            )
        else:
            truncate_slice = tuple(
                slice(truncate_size, -truncate_size) if i == axis else slice(None)
                for i in range(len(self.shape))
            )
        new_shape = tuple(
            size if i == axis else self.discretization[i]
            for i in range(len(self.shape))
        )
        return self.__class__(
            **{
                **vars(self),
                "state": self.state[truncate_slice],
                "discretization": new_shape,
            }
        ).transform(to=self.basis)

    def _parse_state(self, state, basis, **kwargs):
        """
        Parse and assign 'state', 'basis' and 'discretization' attributes.

        Parameters
        ----------
        state : ndarray
            Numpy array containing state information, can have any number of dimensions.
        basis : str
            The basis that the array 'state' is in.

        Notes
        -----

        Must assign attributes 'state', 'basis', 'discretization' as None if no input received, even though technically
        shape of empty array will be, for example, (0, 0). In other words, discretization being NoneType acts as a flag
        that the state is unpopulated.

        """
        if isinstance(state, np.ndarray):
            self.state = state
        elif state is None:
            self.state = np.array([], dtype=float).reshape(
                len(self._default_shape()) * [0]
            )
        else:
            raise ValueError(
                '"state" attribute may only be provided as NumPy array or None.'
            )

        if self.size > 0:
            # This seems redundant but typically the discretization needs to be inferred from the state
            # and the basis; as the number of variables is apt to change when symmetries are taken into account.
            self.basis = basis
            self.discretization = self.state.shape
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
        else:
            self.discretization = None
            self.basis = None

    def _parse_parameters(self, parameters, **kwargs):
        """
        Parse and set the parameters attribute.

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

        The subclass essentially defines what is and is not a constant; conversion between classes (and passing
        constraints in the process) can mistakenly unconstrain these constants. Therefore, if the key is not
        in the default constraints dict then it is assumed to be constant.
        This is not an issue because this is one of the fundamental requirements for subclassing.

        """
        # Get the constraints, making sure to not mistakenly unconstrain constants.
        self.constraints = {
            k: kwargs.get("constraints", self._default_constraints()).get(k, True)
            if k in self._default_constraints().keys()
            else True
            for k in self.parameter_labels()
        }
        if parameters is None:
            # None is a valid choice of parameters; it essentially means "populate all parameters" upon generation.
            self.parameters = parameters
        elif isinstance(parameters, tuple):
            # This does not check each tuple element; they can be whatever the user desires, technically.
            # This ensures all parameters are filled.
            if len(self.parameter_labels()) < len(parameters):
                # If more parameters than labels then we do not know what to call them by; truncate by using zip.
                self.parameters = tuple(
                    val for label, val in zip(self.parameter_labels(), parameters)
                )
            else:
                # if more labels than parameters, simply fill with the default missing value, 0.
                self.parameters = tuple(
                    val
                    for label, val in zip_longest(
                        self.parameter_labels(), parameters, fillvalue=0
                    )
                )
        else:
            # A number of methods require parameters to be an iterable, hence the tuple requirement.
            raise TypeError(
                '"parameters" is required to be a tuple or NoneType. '
                "singleton parameters need to be cast as tuple (val,)."
            )

    def _populate_parameters(self, **kwargs):
        """
        Should only be accessed through :meth:`Orbit.populate`

        Parameters
        ----------
        kwargs :
            p_ranges : dict
                keys are parameter_labels, values are uniform sampling intervals or iterables to sample from

        """
        # helper function so comprehension can be used later on; each orbit type typically has a default
        # range of good parameters; however, it is also often the case that using a user-defined range is desired
        # in order to target specific scales.
        def sample_from_generator(val, val_generator, overwrite=False):
            if overwrite or val is None:
                # If the generator is "interval like" then use uniform distribution.
                if isinstance(val_generator, tuple) and len(val_generator) == 2:
                    try:
                        pmin, pmax = val_generator
                        val = pmin + (pmax - pmin) * np.random.rand()
                    except TypeError as typ:
                        vestr = "".join(
                            [
                                "parameter generation requires tuples of length two to have numeric elements;",
                                "for non-numeric types use a list instead.",
                            ]
                        )
                        raise ValueError(vestr) from typ
                # If given a scalar,
                elif type(val_generator) in [str, int, bool, float, np.int32, np.float]:
                    val = val_generator
                else:
                    # Everything else treated as distribution to sample from; integer input selects from range(int)
                    # So that more complex input can be included, sample the positions of the elements in val_generator.
                    index_range = range(len(val_generator))
                    val = val_generator[np.random.choice(index_range)]

            return val

        # seeding takes a non-trivial amount of time, only set if explicitly provided.
        if isinstance(kwargs.get("seed", None), int):
            np.random.seed(kwargs.get("seed", None))

        # Can be useful to override default sample spaces to get specific cases.
        p_ranges = kwargs.get("parameter_ranges", self._default_parameter_ranges())
        # If *some* of the parameters were initialized, we want to save those values; iterate over the current
        # parameters if not None, else a list of zeros.
        parameter_iterable = self.parameters or len(self.parameter_labels()) * [None]
        if len(self.parameter_labels()) < len(parameter_iterable):
            # If more values than labels, then truncate and discard the additional values
            parameters = tuple(
                sample_from_generator(
                    val,
                    p_ranges.get(label, (0, 0)),
                    overwrite=kwargs.get("overwrite", False),
                )
                for label, val in zip(self.parameter_labels(), parameter_iterable)
            )
        else:
            # If more labels than parameters, fill the missing parameters with default values.
            parameters = tuple(
                sample_from_generator(
                    val,
                    p_ranges.get(label, (0, 0)),
                    overwrite=kwargs.get("overwrite", False),
                )
                for label, val in zip_longest(
                    self.parameter_labels(), parameter_iterable, fillvalue=None
                )
            )
        # Once all parameter values have been parsed, set the attribute.
        setattr(self, "parameters", parameters)

    def _populate_state(self, **kwargs):
        """
        Should only be accessed through :meth:`Orbit.populate`

        Parameters
        ----------
        kwargs : dict
            seed : str

        Notes
        -----
        Must populate and set attributes 'state', 'discretization' and 'basis'. The state is required to be a numpy
        array, the discretization is required to be its shape (tuple) in the basis specified by self.bases_labels()[0].
        Discretization is coupled to the state and its specific basis, hence why it is populated here.

        Historically, for the KSe, the strategy to define a specific state was to provide a keyword argument 'spectrum'
        which controlled a spectrum modulation strategy. This is not included in the base signature, because it is
        terminology specific to spectral methods.

        """
        # Just populate a random array; more intricate strategies should be written into subclasses.
        # Using standard normal distribution for values.
        numpy_seed = kwargs.get("seed", None)
        if isinstance(numpy_seed, int):
            np.random.seed(numpy_seed)
        # Presumed to be in physical basis unless specified otherwise; get the size of the state based on dimensions
        self.discretization = kwargs.get(
            "discretization", None
        ) or self.dimension_based_discretization(self.parameters, **kwargs)
        # Assign values from a random normal distribution to the state by default.
        self.state = np.random.randn(*self.discretization)
        # If no basis provided, state generation presumed to be in the physical basis.
        self.basis = kwargs.get("basis", None) or self.bases_labels()[0]


def convert_class(orbit_instance, orbit_type, **kwargs):
    """
    Utility for converting between different symmetry classes.

    Parameters
    ----------
    orbit_instance : Orbit or Orbit subclass instance
        The orbit instance to be converted
    orbit_type : type
        The target class that orbit will be converted to.

    Returns
    -------
    Orbit :
        New Orbit instance whose type is the provided `orbit_type`

    Notes
    -----
    This is for all practical purposes deprecated but it still provides readability so it has
    been kept as a convenience.
    To avoid conflicts with projections onto symmetry invariant subspaces, the orbit is always transformed into the
    physical basis prior to conversion; the instance is returned in the basis of the input, however.

    Include any and all attributes that might be relevant to the new orbit and those which transfer over from
    old orbit via usage of vars(orbit_instance) and kwargs. If for some reason an attribute should not be passed,
    then providing attr=None in the function call is how to handle it, as the values in kwargs overwrite
    the values in vars(orbit_instance) via dictionary unpacking.

    """
    # Note any keyword arguments will overwrite the values in vars(orbit_instance) or state or basis
    return orbit_type(
        **{
            **vars(orbit_instance),
            "state": orbit_instance.transform(
                to=orbit_instance.bases_labels()[0]
            ).state,
            "basis": orbit_instance.bases_labels()[0],
            **kwargs,
        }
    ).transform(to=orbit_instance.basis)
