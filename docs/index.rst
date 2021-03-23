.. _contents:

Orbithunter v |version|: Framework for Nonlinear Dynamics and Chaos
===================================================================

Orbithunter serves as a framework for solving chaotic nonlinear partial differential equations
via variational formulation. In other words, equations are posed as `boundary value problems <https://en.wikipedia.org/wiki/Boundary_value_problem>`_ (typically with periodic
boundary conditions) which manifest as `differential algebraic equations <https://en.wikipedia.org/wiki/Differential-algebraic_system_of_equations>`_.

This is in stark contrast with typical dynamical systems formulation of chaotic systems, which use an initial value problem in time.  
The argument in favor of orbithunter's BVP/DAE formulation is that by definition, the 
hyperbolic dynamical systems under investigation suffer from exponential instabilities. 
This relegates forecasting and prediction to a finite time window.
Orbithunter believes that this is a clear indication that posing the problem as an initial value problem is incorrect; one must create a setting
where dynamics no longer take place.  

Want a `crash course on chaos <http://chaosbook.org/course1/about.html>`_ or a `good reference text <http://chaosbook.org/>`_ (not https secure)? 
Alternatively, if you would like a more traditional approach to computational chaos in fluid dynamics, check out `openpipeflow <https://openpipeflow.org/index.php?title=Main_Page>`_
or `channelflow 2.0 <https://www.channelflow.ch/>`_. If you're looking ahead to the future, the `next big thing <https://diffeq.sciml.ai/v2.0/>`_ might be
based in Julia.

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

Thank you to `Predrag Cvitanovic <http://www.cns.gatech.edu/~predrag/>`_ for his courses and guidance throughout my PhD thesis. 