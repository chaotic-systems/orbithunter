The following is a list of the attributes/methods of the Orbit core class that
are either mandatory or highly recommended to be written for subclassing purposes.

For any equation for 2-d scalar fields, many functions can be borrowed from orbit_ks.py 
(other than the governing equations of course).

Dependencies for numerical optimization
---------------------------------------

spatiotemporal_mapping:
Returns a class object whose state is the evaluation of the governing equations.

matvec: 
Returns class instance with state equal to $A\cdot x$ where $A$ is the Jacobian and x is the original state. 

rmatvec:
Returns class instance with state equal to $A^{\top}\cdot x$ where $A$ is the Jacobian and x is the original state. 

convert : 
Function which converts to and from the specified bases of the problem.

_parse_state :
A manner of processing input for state variable (vector/scalar field typically)

_parse_parameters :
A manner of processing input for parameters variable (vector/scalar field typically)

_random_initial_condition :
Unless all data will be imported, need some manner of creating initial conditions


Dependencies for spt techniques (clip, glue, etc.)
--------------------------------------------------

glue_parameters:
How to add/average parameters for gluing

reshape:
Interpolate/truncate to change the dimensionality of the tile and by consequence, the state.

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





properties, static methods
--------------------------

parameters : 
literally just a tuple of the parameters required for spatiotemporal_mapping/rmatvec/matvec etc. 



Future inclusions
-----------------
preconditioning : method which applies preconditioning operation to state / parameters 
***The inclusion of this ***

preconditioner : matrix representation of precondition












If LSTSQ is desired to be used:

Jacobian : (requires matrix representations of operators in spatiotemporal_mapping, or at least
that is probably the easiest manner of construction)