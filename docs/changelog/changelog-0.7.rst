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
-  Trust region methods now "approximately supported" for the KSe; meaning that the code has been generalized to the point
   where the hessian product can be computed for the KSe, but one of the terms is missing because it has not yet been defined,
   namely the evaluation of F * d^2 F * v; because the system is stiff, however, the jacobian times itself seems to provide enough
   information to enable decrease of the cost functional. Getting the Hessian product is more tricky than confusing, as it involves
   manipulation of a rank 3 tensor. 
-  Shadowing, cover, fill have been rewritten to provide better performance/more consistent results based on window sizes. Now only
   computes scores at pivots valid for ALL window orbits. Previously pivots at the boundaries were taking only subsets of the windows
   due to whether the windows "fit" or not. 
-  New handling of constraints and constants for SciPy; orbit_vector has been split into cdof and constants in order to avoid inclusion
   in the definition of the methods which use LinearOperator objects. Previously, they were included but corrections were constrained
   to be zero. Additionally, the usage of constraints was not handled properly by the definition of :meth:`~Orbit.from_numpy_array`,
   as it was not accessing the correct values if constrained parameters appeared unconstrained parameters, relative to the order of
   parameter labels returned by :meth:`~Orbit.parameter_labels`. Optimization performed surprisingly well when all parameters were
   constrained, actually; may be worth describing alongside preconditioning. 


Minor Changes
-------------

-  KSE Jacobians are now produced much more efficiently; uglier and very confusing code to do this, however, as OrbitKS operations
   are being used very creatively to apply to a 3-d array even though they are only meant for 2-d arrays. 
-  Inplace computation of differentiation and FFTs now implemented for KSe. Uglier code but makes certain calculations more efficient. 
-  np.reshape calls replaced with usage of None/np.newaxis where possible; as it is typically faster.
-  Spatial rotations were not working because the frequencies were being unduely raveled. 
-  Added more generalized gradient descent; adjoint descent is now simply gradient descent with optimizations relevant to cost function $1/2 F^2$
-  Now can pass separate scipy keyword arguments for multiple methods via the `method_kwargs` keyword argument. Single dicts can still be
   passed to `scipy_kwargs` keyword argument. 
-  The function `fill` now uses the relative difference between threshold and score to determine which orbit performed the best.
-  Added the ability to return the coordinates of pivots that produced windows that were out of bounds; should only be non-empty for
   when coordinate mapping functions are provided.  

Bug and Error Fixes
-------------------
-  Continuation was using the old `OptimizeResult.status` in `while` loop, making the code within unreachable: major error.
-  Can now handle cases where mask becomes "full"; i.e. no pivots to iterate over in shadowing. 
-  :meth:`core.Orbit.__getitem__` was not updating the discretization parameters correctly; now forces parsing of the new state
   after slicing, as does the new `concat` method. 
-  When three or more methods were included, :func:`optimize.hunt` was unable to aggregate runtime statistics due to type errors;
   was trying to extend lists with numbers instead of lists
-  Certain keyword arguments that were meant for outer iteration loops (orbithunter routines) and not inner loops (scipy routines)
   were conflicting, causing unintended performance issues. Most notable was ``maxiter`` keyword meant for the number of outer loop iterations
   was determining the size of the Krylov subspace in ``scipy.optimize.newton_krylov``
-  The outer-iteration function factories were actually in the completely wrong place; needed to be within while loop but they were not..
-  Fixed fundamental domain gluing for this for ShiftReflectionOrbitKS by including roll 
-  Keyword argument conflicts with scipy handled. 

Known Issues
============
-  Handling of constraints with SciPy needs to be redone; the orbit_vector method should return only non-constant parameters.
