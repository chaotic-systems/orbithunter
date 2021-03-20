.. _equilibriumorbitks:

==================
EquilibriumOrbitKS
==================


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: EquilibriumOrbitKS


Methods
=======

.. note::
    See also :class:`~orbithunter.core.Orbit`, :class:`~orbithunter.ks.OrbitKS`,
    and  :class:`~orbithunter.ks.AntisymmetricOrbitKS`,

Initialization
--------------
.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.__init__
   EquilibriumOrbitKS.populate
   EquilibriumOrbitKS._populate_state
   EquilibriumOrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for all definitions.


Properties
----------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.shape
   EquilibriumOrbitKS.size
   EquilibriumOrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.shapes
   EquilibriumOrbitKS.dimensions
   EquilibriumOrbitKS.glue_dimensions
   EquilibriumOrbitKS.dimension_based_discretization
   EquilibriumOrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.orbit_vector
   EquilibriumOrbitKS.abs
   EquilibriumOrbitKS.dot
   EquilibriumOrbitKS.norm
   EquilibriumOrbitKS.dx
   EquilibriumOrbitKS.dt
   EquilibriumOrbitKS.eqn
   EquilibriumOrbitKS.matvec
   EquilibriumOrbitKS.rmatvec
   EquilibriumOrbitKS.precondition
   EquilibriumOrbitKS.jacobian

Visualization
-------------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.plot
   EquilibriumOrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.transform
   EquilibriumOrbitKS.resize
   EquilibriumOrbitKS.reflection
   EquilibriumOrbitKS.roll
   EquilibriumOrbitKS.cell_shift
   EquilibriumOrbitKS.rotate
   EquilibriumOrbitKS.shift_reflection
   EquilibriumOrbitKS.to_fundamental_domain
   EquilibriumOrbitKS.from_fundamental_domain
   EquilibriumOrbitKS._pad
   EquilibriumOrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.bases_labels
   EquilibriumOrbitKS.minimal_shape
   EquilibriumOrbitKS.minimal_shape_increments
   EquilibriumOrbitKS.discretization_labels
   EquilibriumOrbitKS.parameter_labels
   EquilibriumOrbitKS.dimension_labels
   EquilibriumOrbitKS.periodic_dimensions
   EquilibriumOrbitKS.positive_indexing

Other
-----

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.copy
   EquilibriumOrbitKS.mask
   EquilibriumOrbitKS.constrain
   EquilibriumOrbitKS.preprocess

Defaults
--------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.defaults
   EquilibriumOrbitKS._default_shape
   EquilibriumOrbitKS._default_parameter_ranges
   EquilibriumOrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   EquilibriumOrbitKS.filename
   EquilibriumOrbitKS.to_h5



