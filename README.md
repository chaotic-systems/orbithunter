# orbithunter
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner
used in Ph.D. thesis work. Currently only solves the Kuramoto-Sivashinsky equation
(nonlinear PDE) with doubly periodic boundary conditions using 
spatiotemporal Fourier modes.
 
Recent Changes
--------------
Changed the spatial modes retained for RelativeEquilibriumKS and EquilibriumKS
so that spatial transforms (and the corresponding matrices) can be inherited.

Now, this is wasteful computationally because of how the pseudospectral products
are computed. This is purely to maintain consistency and also allow for easier co-moving
transformations for RelativeEquilibriumKS

PEP 8 and Refactoring
---------------------
1. Reduce the try-except statement in __init__
2. Convert some variable names to pep8 conventions: N, M constant; T,L 
	not constant -> period, speriod 

To-do
-----
RelativeEquilibriumKS and EquilibriumKS need to be fully implemented. 
RelativeEquilibriumKS.status() needs to account for shift as part of "time derivative". 
Implement flag for reference frame of relative periodic solutions. 
Different methods for random initial condition generation. 
Convert all research code scripts
Add persistent homology tools using 'Gudhi' package.
More numerical methods using scipy and its LinearOperator class.
Neural networks for spatiotemporal symbolic dynamics classification
Create base "Orbit" class which can be subclassed for different equations other than the Kuramoto-Sivashinsky equation,
potentially. 
Create testing suite using a package i.e. unit testing.

Known Bugs and issues
---------------------
FFT matrices incorrect for discrete symmetry types, due to conversion from scipy.fftpack -> scipy.fft
If in co-moving frame, RelativeOrbitKS.calculate_shift() will return 0 instead of the proper spatial shift. 

Zero-padding of discrete symmetry fundamental domains (AntisymmetricOrbitKS, ShiftReflectionOrbitKS)
Zero-padding when not in comoving reference frame (RelativeOrbitKS)



