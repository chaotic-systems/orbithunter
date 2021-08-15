orbithunter 1.0
===============

.. currentmodule:: orbithunter

Major Changes
-------------

-  Shadowing module completely redone with new :class:`shadowing.OrbitCover` class.
-  Fixes of numerical methods, function factories, and just errors in formulation with respect to hunt; all functionality should
   be available for classes which have implemented Hessian, however, the default can get surprisingly far
   using only the cost function's Hessian if the linear terms are dominant w.r.t. the equation's Hessian.
-  Optimization of `shadowing` module in terms of excessive or redundant computation, especially in :meth:`shadowing.OrbitCover.map`,
   now only returns mapped scores corresponding to those which satisfy the corresponding threshold.
-  Tutorial notebooks and docker container are now reasonable.    

Minor Changes
-------------

-  `include_zeros` changed to `include_zero_dimensions` in :func:`gluing.glue`

Bug and Error Fixes
-------------------

-  Many bugs related to pivots, trimming, and mapping in the :func:`shadowing.cover`
-  CG and CGS were not working for the KSe when parameters were constrained (square Jacobian); they are, for now,
   forced to evaluate the normal equations in this case.

Other
-----
- All numerical methods tested on KSe such that they all reduce the residual to some capacity now.


