# orbithunter Version 0.6.0
-------------------------
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner.

# Introduction
--------------
This package enables the layman to solve nonlinear chaotic partial differential equations
by finding unstable periodic orbits by solving the spatiotemporal boundary value problem using
spatiotemporal Fourier modes, currently only implemented for the Kuramoto-Sivashinsky equation.

The utilities are written in a general fashion such that subclassing the core Orbit class
should enable usage of most of the utilities. 

# To-do
-----
- Installation and setup documentation 
- Listing the package on pypi or conda. 
- Create docker or another similar container
- Update documentation
- Logging file utility to track symbolic tiling results
- Pairwise gluing

# Known Bugs and issues
---------------------
- matvec when used in LSQR, LSMR; unavailable for now. 


