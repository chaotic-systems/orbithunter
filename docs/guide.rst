.. _guide:

===============
Developer Guide
===============

Preface
=======

The following guide demonstrates the class methods required for full functionality
in the orbithunter framework. This documentatino is presented much like how one might document their
own equation module for inclusion in the main orbithunter branch. The creator of orbithunter,
`Matthew Gudorf <https://www.linkedin.com/in/mgudorf/>`_ painstakingly developed the framework to
be agnostic of the equations. That is, the techniques and tools should generalize to any equation,
so long as the proper class methods are written. Because of this, the following is presented
as a template for each technique or submodule. Implementation of the methods in each section
should enable the functionality of the corresponding orbithunter module.

Orbithunter was designed with field equations in mind; that is, where the Orbit state
array is a continuous function with respect to its dimensions. While it has not been tested,
the package should work all the same as long as the user treats the vector components as a "discrete dimension",
i.e. arrays of shape (#, #, #, #, 3) or something similar.

.. warning::
   Be sure to check the base :class:`orbithunter.core.Orbit` class before writing your methods; there very well may
   be methods which are not included here because they are written as to generalize to other Orbit classes.  

SymmetryOrbitEQN Class
======================

As mentioned in the preface, ``orbithunter`` has tried to do most of the heavy lifting. For the user, the task
of implementing a module for a new equation is as simple as implementing a certain subset of methods that are equation and or
dimension dependent. If the following list of methods is completed, then all orbithunter utilities should be
available. The lion's share is implementing the spatiotemporal equations and its gradients. The main successes have
been made using spectral methods, which leverage expansions in terms of global, spatiotemporal basis functions. It
is believed that there is an intimate link between these expansions and keeping the spatiotemporal domain sizes variable
quantities, and so only usage of spectral methods is recommended. For periodic boundary conditions we recommended
using a Fourier basis and for aperiodic boundary conditions Chebyshev polynomial bases are recommended. 

.. note::
   This is also available as a ``.py`` file in the tutorials under `class_template <https://github.com/mgudorf/orbithunter/tutorial/class_template.py>`_

.. currentmodule:: tutorial.class_template
.. autoclass:: SymmetryOrbitEQN

Methods
-------

Static Methods
^^^^^^^^^^^^^^

Methods decorated with ``@staticmethod``

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SymmetryOrbitEQN.bases_labels
   SymmetryOrbitEQN.parameter_labels
   SymmetryOrbitEQN.discretization_labels
   SymmetryOrbitEQN.dimension_labels
   SymmetryOrbitEQN.minimal_shape
   SymmetryOrbitEQN.minimal_shape_increments
   SymmetryOrbitEQN.continuous_dimensions

Governing Equations
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SymmetryOrbitEQN.eqn
   SymmetryOrbitEQN.matvec
   SymmetryOrbitEQN.rmatvec
   SymmetryOrbitEQN.jacobian

Numerical Optimization
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN.cost
   SymmetryOrbitEQN.costgrad

Second Order Numerical Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The methods for usage of second^order numerical methods which returns the Hessian matrix or
the matrix^vector product thereof. The SciPy implementations of the numerical methods that use these
are fully developed but the orbithunter API still requires testing.


.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN.hessp
   SymmetryOrbitEQN.hess
   SymmetryOrbitEQN.costhess
   SymmetryOrbitEQN.costhessp

Attribute Related
^^^^^^^^^^^^^^^^^

Initialization and basis transformation methods.

.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN.transform
   SymmetryOrbitEQN._populate_state
   SymmetryOrbitEQN._parse_state

.. note:: See also :meth:`orbithunter.core.Orbit.constrain`

Defaults
^^^^^^^^

.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN._default_shape
   SymmetryOrbitEQN._default_parameter_ranges
   SymmetryOrbitEQN._default_constraints
   SymmetryOrbitEQN._dimension_indexing_order


State Transformations
^^^^^^^^^^^^^^^^^^^^^

These methods are recommended but optional methods; the base orbit class has simple implementations for all of these

.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN.glue_dimensions
   SymmetryOrbitEQN._pad
   SymmetryOrbitEQN._truncate


Other
^^^^^

The methods in this section are ones which really cannot be generalized at all, methods which may be heavily reliant
on equation and methods which do not really fit anywhere else on this list. .

.. autosummary::
   :toctree: generated/

   SymmetryOrbitEQN.plot
   SymmetryOrbitEQN.from_numpy_array
   SymmetryOrbitEQN.dimension_based_discretization
   SymmetryOrbitEQN.periodic_dimensions