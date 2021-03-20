Kuramoto-Sivashinsky Equation
-----------------------------

.. note:: For the Kuramoto-Sivashinsky equation there are six different :class:`~orbithunter.core.Orbit` subclasses

Which orbit class should I use?
===============================

Let $G$ denote a group of discrete spatiotemporal symmetries $G = \{e, \sigma, \tau, \sigma\tau\}$ which
represent the identity, spatial reflection, half-period time translation and spatiotemporal shift reflection (glide)
which is the composition of spatial reflection and half-period time translation.

+-----------------------------+---------------------------------------------------+--------------------+
| Orbit Class                 | Invariance                                        | Equivariance       |
+=============================+===================================================+====================+
| OrbitKS                     | None                                              | Discrete rotations |
+-----------------------------+---------------------------------------------------+--------------------+
| RelativeOrbitKS             | None                                              | Discrete rotations |
+-----------------------------+---------------------------------------------------+--------------------+
| RelativeEquilibriumOrbitKS  | None                                              | Discrete rotations |
+-----------------------------+---------------------------------------------------+--------------------+
| ShiftReflectionOrbitKS      | Spatial reflection + half-period time translation | $G$                |
+-----------------------------+---------------------------------------------------+--------------------+
| AntisymmetricOrbitKS        | Spatial reflection                                | $G$                |
+-----------------------------+---------------------------------------------------+--------------------+
| EquilibriumOrbitKS          | Spatial Reflection and time translation           | $G$                |
+-----------------------------+---------------------------------------------------+--------------------+



Orbit Types
===========

.. toctree::
   :maxdepth: 1

   orbitks
   relativeorbitks
   antisymmetricorbitks
   equilibriumorbitks
   relativeequilibriumorbitks
   shiftreflectionorbitks
   physics




