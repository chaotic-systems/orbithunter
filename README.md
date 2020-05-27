# orbithunter
Framework for solving chaotic nonlinear PDEs in spatiotemporal manner
used in Ph.D. thesis work. Currently only solves the Kuramoto-Sivashinsky equation
(nonlinear PDE) with doubly periodic boundary conditions by utilizing a 
spatiotemporal Fourier modes to represent the
equations and exploit spatiotemporal symmetries.
 

PEP 8
----------
1. Reduce the try-except statement in __init__
2. convert variable names to pep8 conventions: N, M constant; T,L not constant -> period, speriod (can't use l) ... t, x? 

Bug Fixes
----------


To-do
-----
Implement flag for reference frame of relative periodic solutions. 
Convert research code scripts to orbithunter framework

Bugs
----
FFT matrices incorrect for discrete symmetry types, due to conversion from scipy.fftpack -> scipy.fft
Zero-padding of fundamental domains



