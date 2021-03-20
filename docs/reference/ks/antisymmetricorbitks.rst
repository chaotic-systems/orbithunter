.. _antisymmetricorbitks:

====================
AntisymmetricOrbitKS
====================


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: AntisymmetricOrbitKS


Methods
=======

.. note::
    See also :class:`~orbithunter.core.Orbit` and :class:`~orbithunter.ks.OrbitKS`.

Initialization
--------------
.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.__init__
   AntisymmetricOrbitKS.populate
   AntisymmetricOrbitKS._populate_state
   AntisymmetricOrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for all definitions.


Properties
----------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.shape
   AntisymmetricOrbitKS.size
   AntisymmetricOrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.shapes
   AntisymmetricOrbitKS.dimensions
   AntisymmetricOrbitKS.glue_dimensions
   AntisymmetricOrbitKS.dimension_based_discretization
   AntisymmetricOrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.orbit_vector
   AntisymmetricOrbitKS.abs
   AntisymmetricOrbitKS.dot
   AntisymmetricOrbitKS.norm
   AntisymmetricOrbitKS.dx
   AntisymmetricOrbitKS.dt
   AntisymmetricOrbitKS.eqn
   AntisymmetricOrbitKS.matvec
   AntisymmetricOrbitKS.rmatvec
   AntisymmetricOrbitKS.precondition
   AntisymmetricOrbitKS.jacobian

Visualization
-------------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.plot
   AntisymmetricOrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.transform
   AntisymmetricOrbitKS.resize
   AntisymmetricOrbitKS.reflection
   AntisymmetricOrbitKS.roll
   AntisymmetricOrbitKS.cell_shift
   AntisymmetricOrbitKS.rotate
   AntisymmetricOrbitKS.shift_reflection
   AntisymmetricOrbitKS.to_fundamental_domain
   AntisymmetricOrbitKS.from_fundamental_domain
   AntisymmetricOrbitKS._pad
   AntisymmetricOrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.bases_labels
   AntisymmetricOrbitKS.minimal_shape
   AntisymmetricOrbitKS.minimal_shape_increments
   AntisymmetricOrbitKS.discretization_labels
   AntisymmetricOrbitKS.parameter_labels
   AntisymmetricOrbitKS.dimension_labels
   AntisymmetricOrbitKS.periodic_dimensions
   AntisymmetricOrbitKS.positive_indexing

Other
-----

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.copy
   AntisymmetricOrbitKS.mask
   AntisymmetricOrbitKS.constrain
   AntisymmetricOrbitKS.preprocess
   AntisymmetricOrbitKS.selection_rules

Defaults
--------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.defaults
   AntisymmetricOrbitKS._default_shape
   AntisymmetricOrbitKS._default_parameter_ranges
   AntisymmetricOrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   AntisymmetricOrbitKS.filename
   AntisymmetricOrbitKS.to_h5



