# orbithunter Version 0.9.0
-------------------------
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner.

# Introduction
--------------
This package enables the layman to solve nonlinear chaotic partial differential equations
by finding unstable periodic orbits by solving the spatiotemporal boundary value problem.
Currently there is only implementations of all techniques for the Kuramoto-Sivashinsky equation (KSE)

The general usage of this package is to find solutions to a system of differential algebraic equations
derived by substitution of a spatiotemporal Fourier mode basis into the KSE. 

The utilities are written in a general fashion such that subclassing the core Orbit class
should enable usage of most of the utilities. See 

# To-do
-----
- Installation and setup documentation, listing the package on pypi or conda. 
- Create docker container or 
- Update documentation
- 'Expensive' gluing which searches symmetry group orbits.
- Strengthening numerical optimization methods; this will forever be on the to-do list:
  
	1.Custom GMRES implementation with variants like flexible gmres, deflated gmres, etc.
	2.Custom trust-region methods like: newton-krylov hookstep, dogleg, etc.

- Inclusion of more equations with and without different boundary conditions (i.e. Chebyshev polynomials)

# Known Bugs and issues
---------------------
-Converting to and from RelativeOrbitKS when the field is in the comoving is not 


