orbithunter 0.7rc1
==================

.. currentmodule:: orbithunter

Major Changes
-------------

-  Orbit.__getattr__ now provides more detail regarding errors fetching attributes, specifically those corresponding to parameter labels.
-  Numerical algorithm wrappers now have separate factory functions which produce the callables required for the various SciPy routines.
   This allows for more tidiness and separability between requirements for various routines. 
-  Many fixes in SciPy wrappers
-  Orbit not allow for assignment of `workers` attribute; determines cores to use for KSe fft transforms. Only useful if dimension approx >= 128.
-  :meth:`Orbit.transform` now takes keyword argument "inplace"; this leverages SciPy's `overwrite_x` in their `rfft` and `irfft` functions, 
   and also overwrites orbit data in place as well. 
-  :meth:`Orbit.costgrad` now optionally takes Orbit.eqn() to be more general; i.e. allow for costgrad which do not require eqn to be passed. 
-  Jacobian for OrbitKS subclasses now much more efficient, many matrix related helper functions have been deprecated
   Memory usage cut down by approx 50% and time cut down by factor of 10.
-  Optimization methods now take "factories" which produce the requisite callables for numerical methods to allow for more customization.
   In other words, cost functions, their gradients, preconditioning can be handled without defining new class methods. 
-  Shadowing has been re-done. :func:`~shadowing.cover` now returns scores in untrimmed pivot format only. These scores can
   be processed using the new :func:`~shadowing.process_scores` function. Additionally, the mapping to orbit format scores
   and those which involve coordinate maps are done through this function as well. 
-  Example/provided shadowing metrics are now bundled into :func:`~shadowing.scoring_functions`.  
-  Hessian based methods are now supported but have not been thoroughly tested; worked with SciPy contributor to get finite
   difference methods to work (he did all of the actual coding, I just brought it to light). 
-  New method :meth:`~core.Orbit.concat` for simpler pair wise "gluing". Allows ease of gluing without having
   to comprise an array of orbits, its shape, etc. Developed with fundamental domains of discrete symmetries in mind.
-  Pairwise gluing was getting fundamental domains wrong. I have made it so glue and tile do not use fundamental
   domains but pairwise does. 

Minor Changes
-------------

-  KSE Jacobians are now produced much more efficiently
-  Inplace computation of differentiation and FFTs now implemented for KSe. Uglier code but makes certain calculations more efficient. 
-  np.reshape calls replaced with usage of None/np.newaxis where possible; as it is typically faster.


Bug Fixes
---------
-  :meth:`core.Orbit.__getitem__` was not updating the discretization parameters correctly; now forces parsing of the new state
   after slicing, as does the new `concat` method. 
-  When three or more methods were included, :func:`optimize.hunt` was unable to aggregate runtime statistics due to type errors;
   was trying to extend lists with numbers instead of lists
-  Certain keyword arguments that were meant for outer iteration loops (orbithunter routines) and not inner loops (scipy routines)
   were conflicting, causing unintended performance issues. Most notable was ``maxiter`` keyword meant for the number of outer loop iterations
   was determining the size of the Krylov subspace in ``scipy.optimize.newton_krylov``
-  The outer-iteration function factories were actually in the completely wrong place; needed to be within while loop but they were not..
-  Fixed fundamental domain gluing for this for ShiftReflectionOrbitKS by including roll 