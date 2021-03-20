.. _relativeequilibriumorbitks:

==========================
RelativeEquilibriumOrbitKS
==========================


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: RelativeEquilibriumOrbitKS


Methods
=======

.. note::
    See also :class:`~orbithunter.core.Orbit`, :class:`~orbithunter.ks.OrbitKS`
    and :class:`~orbithunter.ks.RelativeOrbitKS`.

Initialization
--------------
.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.__init__
   RelativeEquilibriumOrbitKS.populate
   RelativeEquilibriumOrbitKS._populate_state
   RelativeEquilibriumOrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for all definitions.

Properties
----------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.shape
   RelativeEquilibriumOrbitKS.size
   RelativeEquilibriumOrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.shapes
   RelativeEquilibriumOrbitKS.dimensions
   RelativeEquilibriumOrbitKS.glue_dimensions
   RelativeEquilibriumOrbitKS.dimension_based_discretization
   RelativeEquilibriumOrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.orbit_vector
   RelativeEquilibriumOrbitKS.abs
   RelativeEquilibriumOrbitKS.dot
   RelativeEquilibriumOrbitKS.norm
   RelativeEquilibriumOrbitKS.dx
   RelativeEquilibriumOrbitKS.dt
   RelativeEquilibriumOrbitKS.eqn
   RelativeEquilibriumOrbitKS.matvec
   RelativeEquilibriumOrbitKS.rmatvec
   RelativeEquilibriumOrbitKS.precondition
   RelativeEquilibriumOrbitKS.jacobian
   RelativeEquilibriumOrbitKS.calculate_spatial_shift


Visualization
-------------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.plot
   RelativeEquilibriumOrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.transform
   RelativeEquilibriumOrbitKS.resize
   RelativeEquilibriumOrbitKS.reflection
   RelativeEquilibriumOrbitKS.roll
   RelativeEquilibriumOrbitKS.cell_shift
   RelativeEquilibriumOrbitKS.rotate
   RelativeEquilibriumOrbitKS.shift_reflection
   RelativeEquilibriumOrbitKS.to_fundamental_domain
   RelativeEquilibriumOrbitKS.from_fundamental_domain
   RelativeEquilibriumOrbitKS.change_reference_frame
   RelativeEquilibriumOrbitKS._pad
   RelativeEquilibriumOrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.bases
   RelativeEquilibriumOrbitKS.minimal_shape
   RelativeEquilibriumOrbitKS.minimal_shape_increments
   RelativeEquilibriumOrbitKS.discretization_labels
   RelativeEquilibriumOrbitKS.parameter_labels
   RelativeEquilibriumOrbitKS.dimension_labels
   RelativeEquilibriumOrbitKS.positive_indexing

Other
-----

:meth:`RelativeEquilibriumOrbitKS.periodic_dimensions` is not 'static', unlike its parent; this is due to unavoidable
symmetry specific considerations. For this reason, the staticmethod decorator was not used.

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.copy
   RelativeEquilibriumOrbitKS.mask
   RelativeEquilibriumOrbitKS.constrain
   RelativeEquilibriumOrbitKS.preprocess
   RelativeEquilibriumOrbitKS.periodic_dimensions

Defaults
--------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.defaults
   RelativeEquilibriumOrbitKS._default_shape
   RelativeEquilibriumOrbitKS._default_parameter_ranges
   RelativeEquilibriumOrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   RelativeEquilibriumOrbitKS.filename
   RelativeEquilibriumOrbitKS.to_h5



