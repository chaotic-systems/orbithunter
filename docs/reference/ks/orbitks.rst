.. _orbitks:

=======
OrbitKS
=======


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: OrbitKS

Methods
=======

Initialization
--------------
.. autosummary::
   :toctree: generated/

   OrbitKS.__init__
   OrbitKS.populate
   OrbitKS._populate_state
   OrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for more details.

Properties
----------

.. autosummary::
   :toctree: generated/

   OrbitKS.shape
   OrbitKS.size
   OrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   OrbitKS.shapes
   OrbitKS.dimensions
   OrbitKS.glue_dimensions
   OrbitKS.dimension_based_discretization
   OrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   OrbitKS.orbit_vector
   OrbitKS.abs
   OrbitKS.dot
   OrbitKS.norm
   OrbitKS.dx
   OrbitKS.dt
   OrbitKS.eqn
   OrbitKS.matvec
   OrbitKS.rmatvec
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
   OrbitKS.reflection
   OrbitKS.roll
   OrbitKS.cell_shift
   OrbitKS.rotate
   OrbitKS.shift_reflection
   OrbitKS.to_fundamental_domain
   OrbitKS.from_fundamental_domain
   OrbitKS._pad
   OrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   OrbitKS.bases_labels
   OrbitKS.minimal_shape
   OrbitKS.minimal_shape_increments
   OrbitKS.discretization_labels
   OrbitKS.parameter_labels
   OrbitKS.dimension_labels
   OrbitKS.periodic_dimensions
   OrbitKS.positive_indexing

Other
-----

.. autosummary::
   :toctree: generated/

   OrbitKS.copy
   OrbitKS.mask
   OrbitKS.constrain
   OrbitKS.preprocess

Defaults
--------

.. autosummary::
   :toctree: generated/

   OrbitKS.defaults
   OrbitKS._default_shape
   OrbitKS._default_parameter_ranges
   OrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   OrbitKS.filename
   OrbitKS.to_h5




