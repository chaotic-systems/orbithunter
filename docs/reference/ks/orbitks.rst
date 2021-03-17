=======
OrbitKS
=======


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: OrbitKS


Methods
=======

Binary Operators
----------------

.. note:: See :class:`orbithunter.core.Orbit` for details.

Assignment Operators
--------------------

.. note:: See :class:`orbithunter.core.Orbit` for details.

Numerical Methods
-----------------

.. autosummary::
   :toctree: generated/

   OrbitKS.orbit_vector
   OrbitKS.dx
   OrbitKS.dt
   OrbitKS.eqn
   OrbitKS.matvec
   OrbitKS.rmatvec
   OrbitKS.cost_function_gradient
   OrbitKS.precondition
   OrbitKS.jacobian

Visualization
-------------

.. autosummary::
   :toctree: generated/

   OrbitKS.plot
   OrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   OrbitKS.transform
   OrbitKS.resize

Symmetry Transformations
------------------------

.. autosummary::
   :toctree: generated/

   OrbitKS.reflection
   OrbitKS.roll
   OrbitKS.cell_shift
   OrbitKS.rotate
   OrbitKS.shift_reflection


Static
------

.. autosummary::
   :toctree: generated/

   OrbitKS.default_shape
   OrbitKS.minimal_shape
   OrbitKS.minimal_shape_increments
   OrbitKS.bases
   OrbitKS.discretization_labels
   OrbitKS.parameter_labels
   OrbitKS.dimension_labels

Other
-----

.. autosummary::
   :toctree: generated/

   OrbitKS.to_h5
   OrbitKS.preprocess
   OrbitKS.plotting_dimensions