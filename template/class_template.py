import numpy as np
import os
import sys

# Only because this file is located in the tutorial folder; if orbithunter installed then replace
sys.path.insert(0, os.path.abspath(".."))
from orbithunter.core import Orbit

__all__ = ["SymmetryOrbitEQN"]


class SymmetryOrbitEQN(Orbit):
    """
    Template for implementing other equations. Name is demonstration of orbithunter naming conventions

    Parameters
    ----------
    state : ndarray, default None
        If an array, it should contain the state values congruent with the 'basis' argument.
    basis : str, default None
        Which basis the array state is currently in.
    parameters : tuple, default None
        Parameters required to uniquely define the Orbit.
    discretization : tuple, default None
        The shape of the state array in configuration space, i.e. the 'physical' basis.
    constraints : dict, default None
        Dictionary whose labels are parameter labels, and values are bool. If True, then corresponding parameter
        will be treated as a constant during optimization routines.
    kwargs :
        Possible extra arguments for _parse_parameters and _parse_state.

    Notes
    -----
    The name of the base class for new equations should be Orbit + equation acronym, symmetry subclasses
    adding a prefix which describes said symmetry.

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
        """
        This code is ran upon calling an orbit type. i.e. Orbit()

        Parameters
        ----------
        state : ndarray, default None
            If an array, it should contain the state values congruent with the 'basis' argument.
        basis : str, default None
            Which basis the array state is currently in.
        parameters : tuple, default None
            Parameters required to uniquely define the Orbit.
        discretization : tuple, default None
            The shape of the state array in configuration space, i.e. the 'physical' basis.
        constraints : dict, default None
            Dictionary whose labels are parameter labels, and values are bool. If True, then corresponding parameter
            will be treated as a constant during optimization routines.
        kwargs :
            Extra arguments for _parse_parameters and _parse_state.

        Notes
        -----
        """
        pass  # your code for extra operations goes here.

        # After custom operations, the base Orbit class' routine is called.
        super().__init__(
            state=state,
            basis=basis,
            parameters=parameters,
            discretization=discretization,
            constraints=constraints,
            **kwargs,
        )

    @staticmethod
    def bases_labels():
        """
        Labels of the different bases that 'state' attribute can be in.

        Returns
        -------
        tuple :
            tuple of strings which label the possible bases that the state can be in.

        Examples
        --------
        For the Kuramoto-Sivashinsky equation:

        >>> return "field", "spatial_modes", "modes"

        """
        pass  # your code goes here

    @staticmethod
    def parameter_labels():
        """
        Strings to use to label dimensions. Generic 3+1 spacetime labels default.

        Returns
        -------
        tuple :
            tuple of strings which label the parameters.


        Examples
        --------
        For the base Orbit class this returns

        >>> "t", "x", "y", "z"

        """
        pass  # your code goes here

    @staticmethod
    def dimension_labels():
        """
        Strings to use to label dimensions/periods; typically a subset of parameter_labels.

        Returns
        -------
        tuple :
            tuple of strings which label the dimensions of the current equation

        Examples
        --------
        For classes where dimensions are the only parameters (i.e. Orbit) this returns

        >>> "t", "x", "y", "z"

        """
        pass  # your code goes here

    @staticmethod
    def discretization_labels():
        """
        Strings to use to label discretization variables. Generic 3+1 spacetime labels default.

        Returns
        -------
        tuple :
            tuple of strings which label the discretization variables of the current state.

        Examples
        --------
        The labels of the collocation grid; for Orbit:

        >>> "n", "i", "j", "k"

        """
        pass  # your code goes here

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
        pass  # your code goes here

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
        pass  # your code goes here

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
        pass  # your code goes here

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
        pass  # your code goes here

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

        Notes
        -----
        If physical scales need to be resolved, then discretization size usually scales with dimension; this
        is a method which should yield some heuristical methods for capturing this scaling. Microextensive equations
        should incrementally increase.

        """
        pass  # your code goes here

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
        These shapes are defined by the transforms but it is wasteful to transform simply to get the shape,
        and the amount of boilerplate code to constantly infer the shape justifies this method in most cases.

        """
        pass  # your code goes here

    def transform(self, to=None):
        """
        Method that handles all basis transformations. Undefined/trivial for Orbits with only one basis.

        Parameters
        ----------
        to : str
            The basis to transform into. If already in said basis, returns self (not a copy!)

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
        pass  # your code goes here

    def eqn(self, *args, **kwargs):
        """
        Return an instance whose state is an evaluation of the governing equations.

        Returns
        -------
        Orbit :
            Orbit instance whose state equals evaluation of governing equation. An array that is uniformly 0 indicates
            a solution.

        """
        # Not required but recommended for safety if there is considerable overhead transforming between bases.
        # assert (self.basis == self.bases_labels()[-1])
        pass  # your code goes here

    def costgrad(self, eqn, **kwargs):
        """
        Gradient of cost function; optional unless cost was defined

        Parameters
        ----------
        eqn : Orbit
            Orbit instance whose state is an evaluation of the governing equations
        kwargs : dict
            extra arguments for rmatvec method.

        Returns
        -------
        gradient :
            An Orbit instance whose state and parameters (i.e. orbit_vector) contain gradient information.

        """
        pass  # your code goes here

    def matvec(self, other, **kwargs):
        """
        Matrix-vector product of self.jacobian and other.orbit_vector

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_matvec :
            Orbit with values representative of the matrix-vector product; typically state variables only.

        Notes
        -----
        This method represents the matrix-vector product of the Jacobian matrix with an orbit vector of dimension
        self.size+len(self.parameters).

        """
        pass  # your code goes here

    def rmatvec(self, other, **kwargs):
        """
        Matrix-vector product of adjoint of self.jacobian and other.state

        Parameters
        ----------
        other : Orbit
            Orbit whose state represents the vector in the matrix-vector product.

        Returns
        -------
        orbit_rmatvec :
            Orbit with values representative of the adjoint-vector product. Typically state and parameter information.

        Notes
        -----
        This method represents the matrix-vector product of the adjoint Jacobian matrix with an orbit state of dimension
        self.size. Typically for these systems, the adjoint Jacobian has dimensions
        [self.size + len(self.parameters), self.size]; this returns a vector of the same dimension as self.orbit_vector;
        i.e. this *does* produce components corresponding to the parameters.

        """
        pass  # your code goes here

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
            2-d numpy array equalling the Jacobian matrix of the governing equations evaluated at self.state

        Notes
        -----
        Only need jacobian matrix if direct solvers are used (solvers which solve Ax = b using inverses of A).

        """
        pass  # your code goes here

    """
    Second Ordered Methods
    ----------------------
    For second order and trust region methods
    
    """

    def costhess(self, other, **kwargs):
        """
        Hessian matrix of the cost function

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
        Only required is custom cost function is used

        """
        pass  # your code goes here

    def costhessp(self, other, **kwargs):
        """
        Matrix-vector product with the Hessian of the cost function

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
        Only required is custom cost function is used

        """
        pass  # your code goes here

    """
    Visualization
    -------------
    

    """

    def plot(
        self, show=True, save=False, padding=False, fundamental_domain=False, **kwargs
    ):
        """
        Signature for plotting method.

        """
        pass  # your code goes here

    def preprocess(self):
        """
        Check the "status" of a solution

        Notes
        -----
        This was used for checking whether or not an Orbit had converged to a symmetry subspace or not. For example,
        OrbitKS could find equilibria; clearly it is better to store the data as such as it takes much less memory
        to do so.

        This method, if sensible, should check the type of Orbit vs. its state information, and convert to a different
        symmetry type if conditions are met.
        """
        pass  # your code goes here

    def to_fundamental_domain(self, **kwargs):
        """
        Map an orbit state to its fundamental domain.

        """
        pass  # your code goes here

    def from_fundamental_domain(self, **kwargs):
        """
        Recover the full state from its fundamental domain.

        """
        pass  # your code goes here

    @staticmethod
    def _default_shape():
        """
        The default array shape when dimensions are not specified.

        Returns
        -------
        tuple :
            tuple of int for the shape of the discretization when otherwise not determined.

        """
        pass  # your code goes here

    @classmethod
    def _default_parameter_ranges(cls):
        """
        Intervals (continuous) or iterables (discrete) used to populate parameters.

        Notes
        -----
        tuples or length two are *always* interpreted to be continuous intervals. If you have a discrete variable
        with two options, simply use a list instead of tuple. Discrete variables are populated by using random choice
        from the provided collection.

        Examples
        --------

        The following defaults:

        >>> defaults = {'t': (0, 10), 'n': ['hi', 'I', 'am', 'an', 'example'], 'constantnum':(0, 1, 2, 3)}

        Would, if a call to Orbit.populate() was made would initialize the parameters by choosing a real valued number
        between 0 and 10 for t, a string from the available options for n, and a number from the tuple of choices
        for constantnum. In other words, continuous ranges of values are always given by an "interval" tupole.

        """
        pass  # your code goes here

    def _default_constraints(self):
        """
        Sometimes parameters are necessary but constant; this allows for exclusion from optimization without hassle.

        Returns
        -------
        dict :
            Keys are parameter labels, values are bools indicating whether or not a parameter is constrained.

        """
        pass  # your code goes here

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

        User implementation will likely always have True for the number of dimensions.

        """
        pass  # your code goes here

    def _parse_state(self, state, basis, **kwargs):
        """
        Parse and assign 'state', 'basis' and 'discretization' attributes.

        Parameters
        ----------
        state : ndarray
            Numpy array containing state information, can have any number of dimensions.
        basis : str
            The basis that the array 'state' is in.

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
            self.basis = basis
            if basis is None:
                raise ValueError("basis must be provided when state is provided")
            # The only custom component of this function parses the discretization size based on the array in
            # the provided basis. For most equations, this will be the lattice dimensions or collocation grid dimensions
            # in configuration space. The reason why it is parsed is because constraints/subspaces may occur in
            # reciprocal space.
            pass  # your code goes here
        else:
            self.discretization = None
            self.basis = None

    def _parse_parameters(self, parameters, **kwargs):
        """
        Parse and set the parameters attribute.

        Notes
        -----
        Parameters are not required to be of numerical type if they are constrained. Typically expected to be scalar
        values. The key is to create a coherent orbit vector. It is recommended that categorical parameters should be
        assigned to a different attribute. The reason for this is that in numerical optimization, the orbit_vector;
        the concatenation of self.state and self.parameters is sent to the various algorithms.

        The subclass essentially defines what is and is not a constant; conversion between classes (and passing
        constraints in the process) can mistakenly unconstrain these constants. Therefore, if the key is not
        in the default constraints dict then it is assumed to be constant.

        """
        pass  # your code goes here
        super()._parse_parameters(parameters, **kwargs)

    def _populate_state(self, **kwargs):
        """
        Should only be accessed through :meth:`Orbit.populate`

        Parameters
        ----------
        kwargs : dict
            Takes the following keys by default, left general for subclass generalization

            `seed : int`
                The seed for the numpy random number generator

            `basis : str`
                One of the :meth:`orbithunter.core.Orbit.bases_labels` or ``None`` if independent.
                Defines the basis that the state array is returned after calling this method.

        Notes
        -----
        Must populate and set attributes 'state', 'discretization' and 'basis'. The state is required to be a numpy
        array, the discretization is required to be its shape (tuple) in the basis specified by self.bases_labels()[0].
        Discretization is coupled to the state and its specific basis, hence why it is populated here.

        """
        pass  # your code goes here

    """
    
    The following are very useful but the implementation by the base class should be sufficient for most purposes.  
    
    """

    @classmethod
    def glue_dimensions(cls, dimension_tuples, glue_shape, exclude_nonpositive=True):
        """
        Strategy for combining tile dimensions in gluing; default is arithmetic averaging.

        Parameters
        ----------
        dimension_tuples : tuple of tuples
            A tuple with a number of elements equal to the number of dimensions; each of these dimension-tuples contains
            a number of values equal to the number of orbits in the prospective gluing.
        glue_shape : tuple of ints
            The shape of the gluing being performed i.e. for a 2x2 orbit grid glue_shape would equal (2,2).
        exclude_nonpositive : bool
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
        of dimensions which reduces the cost. The strategy produced by this method is simply a baseline; there
        may be way better ways but it is likely highly dependent on equation.

        """
        pass  # your code goes here

    def _pad(self, size, axis=0):
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
        pass  # your code goes here

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
        pass  # your code goes here
