orbithunter 0.5
===============

.. currentmodule:: orbithunter

Major Changes
-------------

- Renamed cost function related quantities; ``obj`` is reserved for objects in most places and so
  it didn't make sense. :meth:``orbithunter.core.Orbit.residual`` is now :meth:``orbithunter.core.Orbit.residual``
- ``orbithunter.core.Orbit.residual`` is now :meth:``orbithunter.core.Orbit.cost``
- ``orbithunter.core.Orbit.cost_function_gradient`` is now :meth:``orbithunter.core.Orbit.costgrad``
- Support for Hessian based methods has been updated through the new methods; also required tinkering with the scipy wrapper
  :func:`orbithunter.optimize._scipy_optimize_minimize_wrapper`
  
- New methods for second order methods

  * ``orbithunter.core.Orbit.costhess``
  * ``orbithunter.core.Orbit.costhessp``
  * ``orbithunter.core.Orbit.hessian``
  * ``orbithunter.core.Orbit.hessp``
  
 

Misc
----

- Lots of docs changes, clean-up. Still not 100% polished but it's getting there. 



