.. _relativeorbitks:

===============
RelativeOrbitKS
===============


Overview
========
.. currentmodule:: orbithunter
.. autoclass:: RelativeOrbitKS


Methods
=======

.. note::
    See also :class:`~orbithunter.core.Orbit` and :class:`~orbithunter.ks.OrbitKS`.

Initialization
--------------
.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.__init__
   RelativeOrbitKS.populate
   RelativeOrbitKS._populate_state
   RelativeOrbitKS._populate_parameters

Special Methods
---------------

"Special" methods also known as "magic" or "dunder" (double underscore) methods
account for most basic Math operations and other operations pertaining to NumPy arrays.

.. note:: See :class:`~orbithunter.core.Orbit` for all definitions.

Properties
----------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.shape
   RelativeOrbitKS.size
   RelativeOrbitKS.ndim


Discretization and Dimensions
-----------------------------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.shapes
   RelativeOrbitKS.dimensions
   RelativeOrbitKS.glue_dimensions
   RelativeOrbitKS.dimension_based_discretization
   RelativeOrbitKS.plotting_dimensions

Math Functions
--------------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.orbit_vector
   RelativeOrbitKS.abs
   RelativeOrbitKS.dot
   RelativeOrbitKS.norm
   RelativeOrbitKS.dx
   RelativeOrbitKS.dt
   RelativeOrbitKS.eqn
   RelativeOrbitKS.matvec
   RelativeOrbitKS.rmatvec
   RelativeOrbitKS.precondition
   RelativeOrbitKS.jacobian
   RelativeOrbitKS.calculate_spatial_shift


Visualization
-------------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.plot
   RelativeOrbitKS.mode_plot

State Transformations
---------------------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.transform
   RelativeOrbitKS.resize
   RelativeOrbitKS.reflection
   RelativeOrbitKS.roll
   RelativeOrbitKS.cell_shift
   RelativeOrbitKS.rotate
   RelativeOrbitKS.shift_reflection
   RelativeOrbitKS.to_fundamental_domain
   RelativeOrbitKS.from_fundamental_domain
   RelativeOrbitKS.change_reference_frame
   RelativeOrbitKS._pad
   RelativeOrbitKS._truncate

Static
------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.bases_labels
   RelativeOrbitKS.minimal_shape
   RelativeOrbitKS.minimal_shape_increments
   RelativeOrbitKS.discretization_labels
   RelativeOrbitKS.parameter_labels
   RelativeOrbitKS.dimension_labels
   RelativeOrbitKS.positive_indexing

Other
-----

:meth:`RelativeOrbitKS.periodic_dimensions` is not 'static', unlike its parent; this is due to unavoidable
symmetry specific considerations. For this reason, the staticmethod decorator was not used.

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.copy
   RelativeOrbitKS.mask
   RelativeOrbitKS.constrain
   RelativeOrbitKS.preprocess
   RelativeOrbitKS.periodic_dimensions

Defaults
--------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.defaults
   RelativeOrbitKS._default_shape
   RelativeOrbitKS._default_parameter_ranges
   RelativeOrbitKS._default_constraints

Reading and Writing Data
------------------------

.. autosummary::
   :toctree: generated/

   RelativeOrbitKS.filename
   RelativeOrbitKS.to_h5



