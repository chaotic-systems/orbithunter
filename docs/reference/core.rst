.. _core:

================
Base Orbit Class
================

The base Orbit class

.. _coreoverview:

Overview
========

.. currentmodule:: orbithunter.core
.. autoclass:: Orbit

Methods
=======

Initialization
--------------

Examples of these methods are included in in :ref:`coreoverview`

.. autosummary::
   :toctree: generated/

   Orbit.__init__
   Orbit.populate
   Orbit._populate_state
   Orbit._populate_parameters
   Orbit._parse_state
   Orbit._parse_parameters

See `Python Docs <https://docs.python.org/2/reference/datamodel.html#special-method-names>`_ for the definition of 'special'

Special Methods
---------------

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
   Orbit.__iadd__
   Orbit.__isub__
   Orbit.__imul__
   Orbit.__ipow__
   Orbit.__itruediv__
   Orbit.__ifloordiv__
   Orbit.__imod__
   Orbit.__str__
   Orbit.__repr__
   Orbit.__getattr__
   Orbit.__getitem__

Properties
----------

.. autosummary::
   :toctree: generated/

   Orbit.shape
   Orbit.size
   Orbit.ndim


State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   Orbit.reflection
   Orbit.roll
   Orbit.cell_shift
   Orbit.to_fundamental_domain
   Orbit.from_fundamental_domain
   Orbit.resize
   Orbit._pad
   Orbit._truncate

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   Orbit.orbit_vector
   Orbit.eqn
   Orbit.matvec
   Orbit.rmatvec
   Orbit.jacobian
   Orbit.hess
   Orbit.hessp
   Orbit.cost
   Orbit.costgrad
   Orbit.costhess
   Orbit.costhessp
   Orbit.abs
   Orbit.dot
   Orbit.norm
   Orbit.rescale
   Orbit.from_numpy_array
   Orbit.increment

Discretization and Dimension
----------------------------

.. autosummary::
   :toctree: generated/

   Orbit.shapes
   Orbit.dimensions
   Orbit.glue_dimensions
   Orbit.periodic_dimensions
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

   Orbit.bases_labels
   Orbit.parameter_labels
   Orbit.dimension_labels
   Orbit.discretization_labels
   Orbit.minimal_shape
   Orbit.minimal_shape_increments

Other
-----

.. autosummary::
   :toctree: generated/

   Orbit.copy
   Orbit.mask
   Orbit.constrain
   Orbit.preprocess

Defaults
--------

.. autosummary::
   :toctree: generated/

   Orbit.defaults
   Orbit._default_shape
   Orbit._default_parameter_ranges
   Orbit._default_constraints


Utility Functions
=================

.. autofunction:: convert_class

