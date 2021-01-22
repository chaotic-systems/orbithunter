# orbithunter v2.0.0 beta
-------------------------
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner.

# Summary
--------------
This package enables the layman to solve nonlinear chaotic partial differential equations
by finding unstable periodic orbits by solving the spatiotemporal boundary value problem.
Currently there is only implementations of all techniques for the Kuramoto-Sivashinsky equation (KSE);
***HOWEVER***, the package was designed to maximize user-friendliness, generalizability and modularity. The
spatiotemporal techniques are supposively agnostic of equation; this has not been tested thoroughly for
arbitrary equations (preliminary testing via the spatiotemporal Henon map branch).

Everything revolves around the Orbit class. To incorporate other equations unofficially, modules should be written
and placed in their own directories like ```./orbithunter/ks/```. Otherwise I hope to collaborate with others through
the github framework.

The general usage of this package, currently, is to find exponentially unstable periodic orbits; 
solutions to the KSE with doubly periodic boundary conditions. These solutions can have a variety of
symmetries, as indicated by the subclasses of the OrbitKS class. 

# To-do
-----
- Finish/rewrite documentation. 
- Installation and setup documentation, listing the package on pypi/conda. 
- Create docker container or an equivalent. 
- Strengthening numerical optimization methods; this is never ending process, however
the first couple steps would be
	1. Custom GMRES implementation with variants like flexible gmres, deflated gmres, etc.
	2. Hessian based methods

- Inclusion of more equations, specifically (2+1)-dimensional Kolmogorov flow. 

# Known Bugs and issues
---------------------
-KSe shadowing, masking not returned properly; can be accessed via returned orbit's state, shadowing_output[0].mask

