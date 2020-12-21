# orbithunter Version 0.9.0
-------------------------
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner.

# Summary
--------------
This package enables the layman to solve nonlinear chaotic partial differential equations
by finding unstable periodic orbits by solving the spatiotemporal boundary value problem.
Currently there is only implementations of all techniques for the Kuramoto-Sivashinsky equation (KSE);
***HOWEVER***, the package was designed to maximize user-friendlinessl, generalizability and modularity. The
spatiotemporal techniques are supposively agnostic of equation; this has not been tested thoroughly for
arbitrary equations (preliminary testing via the spatiotemporal Henon map branch).

Everything revolves around the Orbit class. To incorporate other equations, modules should be written
and placed in their own directories like ```./orbithunter/ks/```.

The general usage of this package, currently, is to find exponentially unstable periodic orbits; 
solutions to the KSE with doubly periodic boundary conditions. These solutions can have a variety of
symmetries, as indicated by the subclasses of the OrbitKS class. 

# To-do
-----
- Finish/rewrite documentation. 
- Installation and setup documentation, listing the package on pypi or conda. 
- Create docker container or an equivalent. 
- 'Expensive' gluing which searches symmetry group orbits.
- Strengthening numerical optimization methods; this is never ending process, however
the first couple steps would be
	1.Custom GMRES implementation with variants like flexible gmres, deflated gmres, etc.
	2.Custom trust-region methods like: newton-krylov hookstep, dogleg, etc.

- Inclusion of more equations with and without different boundary conditions (i.e. Chebyshev polynomials)

# Known Bugs and issues
---------------------
-If RelativeOrbitKS in physical frame, plotting in comoving is not possible 
-KSe shadowing, masking not returned properly; can be accessed via returned orbit's state, shadowing_output[0].mask

