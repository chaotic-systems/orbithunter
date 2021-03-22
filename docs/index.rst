.. _contents:

Orbithunter |version|

Framework for Nonlinear Dynamics and Chaos
==========================================

Orbithunter serves as a framework for solving chaotic nonlinear partial differential equations
in "spatiotemporal form", which is to say, as `boundary value problems <https://en.wikipedia.org/wiki/Boundary_value_problem>`_ (typically with periodic
boundary conditions) which manifest as `differential algebraic equations <https://en.wikipedia.org/wiki/Differential-algebraic_system_of_equations>`_.


Features
--------

- An object oriented approach of differential algebraic equations.
- Vectorized (tensorized really) computations using NumPy broadcasting and tensor operations.
- A general-purpose framework for finding, visualizing and manipulating these solutions
- High-level access to SciPy API for usage with differential algebraic equations. 
- New spatiotemporal techniques developed in `PhD thesis <https://github.com/mgudorf/orbithunter/blob/main/docs/spatiotemporal_tiling_of_the_KSe.pdf>`_

Orbithunter uses [NumPy]_ and [SciPy]_
for its numerical calculations. Its design emphasizes user-friendliness and modularity;
giving quick and easy access to high-level numerical operations.

Checkout the resources included in the github repository for more help and tutorials! 

Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 2

   install
   guide
   reference/index
   agenda
   issues
   extra_resources
   faq
   bibliography
   news
   license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`


Special Thanks
--------------

I'd like to thank the open source projects [Networkx]_, [NumPy]_, [SciPy]_ and many other packages
for being great guides of how to setup, document, and structure Python packages, in addition to the great tools
they provide. 