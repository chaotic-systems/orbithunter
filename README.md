# orbithunter
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner
used in Ph.D. thesis work. Currently only solves the Kuramoto-Sivashinsky equation
(nonlinear PDE) with doubly periodic boundary conditions using 
spatiotemporal Fourier modes.
 

PEP 8 and Refactoring
---------------------
1. Reduce the try-except statement in __init__
2. Convert some variable names to pep8 conventions: N, M constant; T,L 
	not constant -> period, speriod 

To-do
-----
Implement flag for reference frame of relative periodic solutions. 
Convert research code scripts
Add persistent homology tools using 'Gudhi' package.
More numerical methods using scipy and its LinearOperator class.
Create base "Orbit" class which can be subclassed for different equations other than the Kuramoto-Sivashinsky equation.
Create testing suite using a package i.e. unit testing.

Known Bugs
----------
FFT matrices incorrect for discrete symmetry types, due to conversion from scipy.fftpack -> scipy.fft
Zero-padding of discrete symmetry fundamental domains (AntisymmetricOrbitKS, ShiftReflectionOrbitKS)
Zero-padding when not in comoving reference frame (RelativeOrbitKS)



