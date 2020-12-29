The following is a list of the attributes/methods of the Orbit core class that
are either mandatory or highly recommended to be written for subclassing purposes.

For any equation for 2-d scalar fields, many functions can be borrowed from orbit_ks.py 
(other than the governing equations of course).

Requirements
------------------

Attributes
--------------------------

state : ndarray or None
If None type provided, then random_state is called. 

Properties
----------
parameters : tuple
A tuple representing a collection of  parameters required to define the orbit. 
Typically dimensions + symmetry parameters. 

field_shape : tuple
The shape of the numpy array containing in the 'physical field' basis. 

Static Methods
--------------

dimension_labels : tuple of str
	The string labels for each dimension; for the KS equation these are 'T', 'L'

parameter_labels : tuple of str
	The string labels for all possible parameters. 

Initialization methods
----------------------

_parse_state :
A manner of processing input for state variable (vector/scalar field typically)

_parse_parameters :
A manner of processing input for parameters variable (vector/scalar field typically)

random_state :
Unless all data will be imported, need some manner of creating initial conditions

Dependencies for numerical optimization
---------------------------------------

dae:
Returns a class object whose state is the evaluation of the governing equations.

matvec: 
Returns class instance with state equal to $A\cdot x$ where $A$ is the Jacobian and x is the original state. 

rmatvec:
Returns class instance with state equal to $A^{\top}\cdot x$ where $A^{\top}$ is the Jacobian transpose
and x is the original state. (Recommended to use formal Lagrangian to derive adjoint equations).

transform : 
Function which converts to and from the relevant bases of the state.  ALL
Chebyshev/Fourier transforms should be accessed through this wrapper. 
(i.e. field to modes, modes to field).

from_numpy_array : 
Method for transforming numpy arrays into Orbits. 

residual :
The numerical residual of a state, currently defaults to 1/2|f|^2 where f is the governing equation.

cost_function_gradient : 
The derivative of the function which produces the residual 

Dependencies for spt techniques (clip, glue, etc.)
--------------------------------------------------

glue_parameters:
How to add/average parameters for gluing

reshape:
Interpolate/truncate to change the dimensionality of the tile and by consequence, the state.

@property
field_shape :
The discretization shape of the field, i.e. $(N_t, N_x, N_y, N_z, ...)$

@property
dimensions : 
The magnitude of each field dimension

@property
plotting_dimensions :
Returns the dimensions in plotting units, can be the same as dimensions 

Highly recommended
------------------

plot: 
some kind of visualization technique

rescale :
Rescale the maximum/minimum field values, useful for making life easier when constructing random
initial conditions. 


Required for reshape
--------------------

_pad :
interpolate via zero padding (or whatever you need it to do)

_truncate:
reduce dimensionality via truncation of modes






Future inclusions
-----------------
preconditioning : 
method which applies preconditioning operation to state / parameters 

@property
preconditioning_parameters: 
parameters required to apply preconditioning




If LSTSQ is desired to be used:

jacobian : (requires matrix representations of operators in dae, or at least
that is probably the easiest manner of construction)