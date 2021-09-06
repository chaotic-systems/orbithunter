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
-  Sometimes trust region methods would provide complex valued parameters in the case of singular matrices. 
-  OrbitKS subclasses without discrete symmetry are handling complex valued state and parameter values in the trust region methods with hess_strategy='cs'
   they are able to do this because the state arrays are cast as complex before inverse time transforms are applied; discrete symmetry subclasses
   have to do spatial derivative of nonlinear term in spatial mode basis, so when transforming back to modes basis, complex valued input is passed
   to rfft. This has been handled, while trying to maintain the ability to use the finite difference strategy, by casting parameters to reals 
   in the newly overloaded :meth:`~OrbitKs.from_numpy_array`. The consequences of this complex overloading have not been investigated. 