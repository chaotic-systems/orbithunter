=====
Orbit
=====



Overview
========

.. currentmodule:: orbithunter.core
.. autoclass:: Orbit

Methods
=======

Binary Operators
----------------

.. autosummary::
   :toctree: generated/

   Orbit.__add__
   Orbit.__radd__
   Orbit.__sub__
   Orbit.__rsub__
   Orbit.__mul__
   Orbit.__rmul__
   Orbit.__truediv__
   Orbit.__floordiv__
   Orbit.__pow__
   Orbit.__mod__

Assignment Operators
--------------------

.. autosummary::
   :toctree: generated/

   Orbit.__iadd__
   Orbit.__isub__
   Orbit.__imul__
   Orbit.__ipow__
   Orbit.__itruediv__
   Orbit.__ifloordiv__
   Orbit.__imod__

Miscellaneous Built-ins
------------------------

.. autosummary::
   :toctree: generated/

   Orbit.__str__
   Orbit.__repr__
   Orbit.__getattr__
   Orbit.__getitem__

Properties and Defaults
-----------------------

.. autosummary::
   :toctree: generated/

   Orbit.shape
   Orbit.size

State Transformations
---------------------
.. autosummary::
   :toctree: generated/

   Orbit.populate
   Orbit.resize

Symmetry Transformations
------------------------

.. autosummary::
   :toctree: generated/

   Orbit.reflection
   Orbit.roll
   Orbit.cell_shift
   Orbit.to_fundamental_domain
   Orbit.from_fundamental_domain

Numerical Methods
-----------------

.. autosummary::
   :toctree: generated/

   Orbit.abs
   Orbit.dot
   Orbit.norm
   Orbit.orbit_vector
   Orbit.residual
   Orbit.dx
   Orbit.dt
   Orbit.eqn
   Orbit.matvec
   Orbit.rmatvec
   Orbit.cost_function_gradient
   Orbit.precondition
   Orbit.rescale
   Orbit.from_numpy_array
   Orbit.increment
   Orbit.jacobian


Discretization and Dimension
----------------------------

.. autosummary::
   :toctree: generated/

   Orbit.shapes
   Orbit.dimensions
   Orbit.glue_dimensions
   Orbit.dimension_based_discretization

Reading and Writing
-------------------

.. autosummary::
   :toctree: generated/

   Orbit.filename
   Orbit.to_h5

Static
------
.. autosummary::
   :toctree: generated/

   Orbit.bases
   Orbit.parameter_labels
   Orbit.dimension_labels
   Orbit.discretization_labels
   Orbit.default_shape
   Orbit.minimal_shape
   Orbit.minimal_shape_increments


Other
-----

.. autosummary::
   :toctree: generated/

   Orbit.copy
   Orbit.mask
   Orbit.constrain
   Orbit.glue_dimensions
   Orbit.periodic_dimensions

