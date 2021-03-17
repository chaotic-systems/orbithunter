.. _contents:

Orbithunter v0.5b1 (Beta)

Framework for Solving Spatiotemporal PDEs
=========================================

orbithunter serves as a framework for solving chaotic nonlinear partial differential equations.

.. `PhD thesis <https://github.com/mgudorf/orbithunter/tree/main/docs/spatiotemporal_tiling_of_the_KSe.pdf>`_

Features
--------

- An object oriented approach to solutions of partial differential equations
- A general-purpose framework for finding, visualizing and manipulating these solutions
- High-level access to SciPy API, particularly its minimize and optimize packages.
- New spatiotemporal techniques developed in <PhD thesis>'_

Orbithunter uses `NumPy <https://numpy.org/doc/>`_ and `SciPy <https://www.scipy.org/docs.html>`_
for its numerical calculations. Its design emphasizes user-friendliness and modularity;
giving quick and easy access to high-level numerical operations.
Currently there is only an implementation of the Kuramoto-Sivashinsky equation (KSE);

Checkout the resources included in the github repository for help and tutorials! 

.. `Tutorial notebooks <https://github.com/mgudorf/orbithunter/tree/main/notebooks>`_

Check out these jupyter notebooks for a walkthrough of the various tools and utilities. 

Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 1

   install
   tutorial
   reference/index
   guide
   agenda
   faq
   issues
   news
   license
   external
   bibliography

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`