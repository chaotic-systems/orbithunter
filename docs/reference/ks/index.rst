.. _ks:

Kuramoto-Sivashinsky Equation
-----------------------------

.. note:: For the Kuramoto-Sivashinsky equation there are six different :class:`~orbithunter.core.Orbit` types

Which orbit class should I use?
===============================
$$G = (e, r_x, t/2, r_x t/2)$$

+-----------------------------+---------------------------------------------------+--------------------+
| Orbit Class                 | Invariance                                        | Equivariance       |
+=============================+===================================================+====================+
| OrbitKS                     | None                                              | Discrete rotations |
+-----------------------------+---------------------------------------------------+--------------------+
| RelativeOrbitKS             | None                                              | Discrete rotations |
+-----------------------------+---------------------------------------------------+--------------------+
| ShiftReflectionOrbitKS      | Spatial reflection + half-period time translation | $D2$               |
+-----------------------------+---------------------------------------------------+--------------------+
| AntisymmetricOrbitKS        | Spatial reflection                                | $D2$               |
+-----------------------------+---------------------------------------------------+--------------------+
| EquilibriumOrbitKS          | Spatial Reflection and time translation           | $D2$ /             |
+-----------------------------+---------------------------------------------------+--------------------+
| RelativeEquilibriumOrbitKS  | None                                              | $D2$               |
+-----------------------------+---------------------------------------------------+--------------------+


Orbit Types
===========

.. toctree::
   :maxdepth: 1

   antisymmetricorbitks
   equilibriumorbitks
   orbitks
   relativeorbitks
   relativeequilibriumorbitks
   shiftreflectionorbitks
   physics






