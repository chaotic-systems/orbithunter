orbithunter 1.0.5
=================

.. currentmodule:: orbithunter

Minor Changes
-------------

-  Added methods to help with possible memory issues when creating Hessian matrices.
-  Added `approximate` keyword argument for Hessian generation; useful when full Hessian is expensive.
-  Added support for different behaviors for coordinate maps and base/window transformations
-  Changed some arguments in core methods to better describe them.
-  Improvements to efficiency of cover and :meth:`OrbitCover.map`.
-  More documentation/transparency into what `:class:`OrbitCover` does.
-  Improvements to shadowing efficiency; map still needs more work in the very large memory usage case. 
-  Changed how masks are handled in shadowing computations; map previously recomputed the mask based on thresholds
   but it should be thresholds + provided mask. 

Bug and Error Fixes
-------------------

-  :func:`orbithunter.continuation.span_family` did not have a bug fix previously made in :func:`orbithunter.continuation.continuation`
   applied correctly.
-  :meth:`OrbitCover.cover` was not correctly accounting for early-termination of computations
-  :meth:`OrbitCover.cover` was not correctly handling empty pivot iterators.
-  Converting classes between classes with different # of parameters and default constraints was being done incorrectly.
