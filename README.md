# orbithunter
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner
used in Ph.D. thesis work. Currently only solves the Kuramoto-Sivashinsky equation
(nonlinear PDE) with doubly periodic boundary conditions using 
spatiotemporal Fourier modes.
 
Recent Changes
--------------
<<<<<<< HEAD
By using classmethod decorator and lru_cache from functools was able to make computations
more efficient. 

Benchmarking,
For the single test performed, the following were the average time per descent method step

	1. Old research code (numpy arrays only) 0.00149 secs per step 24.4680 secs total
	2. Orbithunter master branch  0.0016 secs per step, 26.33392 secs total
	3. Orbithunter orbithunter-cache-classmethods 0.00094 secs per step 15.4545 total


PEP 8 and Refactoring
---------------------
1. Redo __init__ methods to be less messy, call to super()
=======
RelativeOrbitKS method refactoring to accomodate caching.
Computational benchmarks
1. old research code : Total time 54.5456, per step 0.0033
1. master branch : Total time 50.0003, per step 0.0015
1. orbithunter-caching-classmethods : Total time 36.8843, per step 0.0011

PEP 8 and Refactoring
---------------------

>>>>>>> orbithunter-caching-classmethods

To-do
-----
Redo __init__ methods to be less messy, call to super()
RelativeEquilibriumKS and EquilibriumKS need to be fully implemented. 
RelativeEquilibriumKS.status() needs to account for shift as part of "time derivative".
RelativeEquilibriumKS and EquilibriumKS raising exceptions when .dt() or .dt_matrix() are called. 
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



