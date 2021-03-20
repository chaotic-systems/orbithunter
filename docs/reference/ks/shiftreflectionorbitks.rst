.. _shiftreflectionorbitks:

======================
ShiftReflectionOrbitKS
======================


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: ShiftReflectionOrbitKS


Methods
=======

.. note::
    See also :class:`~orbithunter.core.Orbit` and :class:`~orbithunter.ks.OrbitKS`.

Initialization
--------------
.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.__init__
   ShiftReflectionOrbitKS.populate
   ShiftReflectionOrbitKS._populate_state
   ShiftReflectionOrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for all definitions.


Properties
----------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.shape
   ShiftReflectionOrbitKS.size
   ShiftReflectionOrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.shapes
   ShiftReflectionOrbitKS.dimensions
   ShiftReflectionOrbitKS.glue_dimensions
   ShiftReflectionOrbitKS.dimension_based_discretization
   ShiftReflectionOrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.orbit_vector
   ShiftReflectionOrbitKS.abs
   ShiftReflectionOrbitKS.dot
   ShiftReflectionOrbitKS.norm
   ShiftReflectionOrbitKS.dx
   ShiftReflectionOrbitKS.dt
   ShiftReflectionOrbitKS.eqn
   ShiftReflectionOrbitKS.matvec
   ShiftReflectionOrbitKS.rmatvec
   ShiftReflectionOrbitKS.precondition
   ShiftReflectionOrbitKS.jacobian

Visualization
-------------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.plot
   ShiftReflectionOrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.transform
   ShiftReflectionOrbitKS.resize
   ShiftReflectionOrbitKS.reflection
   ShiftReflectionOrbitKS.roll
   ShiftReflectionOrbitKS.cell_shift
   ShiftReflectionOrbitKS.rotate
   ShiftReflectionOrbitKS.shift_reflection
   ShiftReflectionOrbitKS.to_fundamental_domain
   ShiftReflectionOrbitKS.from_fundamental_domain
   ShiftReflectionOrbitKS._pad
   ShiftReflectionOrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.bases_labels
   ShiftReflectionOrbitKS.minimal_shape
   ShiftReflectionOrbitKS.minimal_shape_increments
   ShiftReflectionOrbitKS.discretization_labels
   ShiftReflectionOrbitKS.parameter_labels
   ShiftReflectionOrbitKS.dimension_labels
   ShiftReflectionOrbitKS.periodic_dimensions
   ShiftReflectionOrbitKS.positive_indexing

Other
-----

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.copy
   ShiftReflectionOrbitKS.mask
   ShiftReflectionOrbitKS.constrain
   ShiftReflectionOrbitKS.preprocess
   ShiftReflectionOrbitKS.selection_rules


Defaults
--------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.defaults
   ShiftReflectionOrbitKS._default_shape
   ShiftReflectionOrbitKS._default_parameter_ranges
   ShiftReflectionOrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   ShiftReflectionOrbitKS.filename
   ShiftReflectionOrbitKS.to_h5



