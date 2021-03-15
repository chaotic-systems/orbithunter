# orbithunter v0.4b1 (Beta)
-------------------------
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner.

# Summary
--------------
This package enables the layman to solve nonlinear chaotic partial differential equations
by finding unstable periodic orbits by solving the spatiotemporal boundary value problem.
Currently there is only implementations of all techniques for the Kuramoto-Sivashinsky equation (KSE);
The package was designed to maximize user-friendliness, generalizability and modularity. The
spatiotemporal techniques are designed to be agnostic of equation; although only the KSE has been implemented
for now. 


Everything revolves around the Orbit class. To incorporate other equations unofficially, modules should be written
and placed in their own directories like ```./orbithunter/ks/```. Otherwise I hope to collaborate with others through
the github framework.

The general usage of this package, currently, is to find exponentially unstable periodic orbits; 
solutions to the KSE with doubly periodic boundary conditions. These solutions can have a variety of
symmetries, as indicated by the subclasses of the OrbitKS class. 


[Tutorials](notebooks/)
Check out these jupyter notebooks for a walkthrough of the various tools and utilities. 

[Implementing your equation](./docs/subclassing_guide.md)

[To-do](./docs/agenda.md)

[Known Bugs and issues](./docs/issues.md)
