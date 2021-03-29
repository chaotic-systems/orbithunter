orbithunter 0.7.0dev
====================

.. currentmodule:: orbithunter

Major Changes
-------------

-  Orbit.__getattr__ now provides more detail regarding errors fetching attributes, specifically those corresponding to parameter labels.
-  Numerical algorithm wrappers now have separate factory functions which produce the callables required for the various SciPy routines.
   This allows for more tidiness and separability between requirements for various routines. 
-  Many fixes in SciPy wrappers
-  Orbit not allow for assignment of `workers` attribute; determines cores to use for KSe fft transforms. Only useful if dimension approx >= 128. 
-  :meth:`Orbit.costgrad()` now optionally takes Orbit.eqn() to be more general; i.e. allow for costgrad which do not require eqn to be passed. 

Minor Changes
-------------

-  KSE Jacobians are now produced much more efficiently
-  Inplace computation of differentiation and FFTs now implemented for KSe. Uglier code but makes certain calculations more efficient. 
-  np.reshape calls replaced with usage of None/np.newaxis where possible; as it is typically faster.


Bug Fixes
---------
- When three or more methods were included, orbithunter.optimize.hunt was unable to aggregate runtime statistics due to type errors;
  was trying to extend lists with numbers instead of lists
- Certain keyword arguments that were meant for outer iteration loops (orbithunter routines) and not inner loops (scipy routines)
  were conflicting, causing unintended performance issues. Most notable was ``maxiter`` keyword meant for the number of outer loop iterations
  was determining the size of the Krylov subspace in ``scipy.optimize.newton_krylov``
- The outer-iteration function factories were actually in the completely wrong place; needed to be within while loop but they were not..
