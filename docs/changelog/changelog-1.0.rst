orbithunter 0.7
===============

.. currentmodule:: orbithunter

Major Changes
-------------

-  Shadowing module completely redone with new :class:`shadowing.OrbitCover` class.
-  Fixes of numerical methods, function factories, and just errors in formulation with respect to hunt; all functionality should
   be available for classes which have implemented Hessian. Everything but trust-region methods should work for those without.
-  Tutorial notebooks and docker container are now reasonable.    

Minor Changes
-------------

-  `include_zeros` changed to `include_zero_dimensions` in :func:`gluing.glue`

Bug and Error Fixes
-------------------
-  Many bugs related to pivots, trimming, and mapping in the :func:`shadowing.cover`


