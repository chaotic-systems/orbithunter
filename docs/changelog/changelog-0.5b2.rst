orbithunter v0.5b2
==================

.. currentmodule:: orbithunter

Major Changes
-------------
- Major updates to documentation; lots of sphinxing learning and deployment.
- Reorganized class methods and privated a number of methods like OrbitKS.nonlinear
  and OrbitKS.rnonlinear; those which served as components of a larger computation.


Minor Changes
-------------

- Made the individual classmethod defaults private, with new classmethod
  :meth:`~Orbit.defaults` .
- Naming conventions in :mod:`persistent homology <orbithunter.persistent_homology>` module
  now obey numpydoc rst.





