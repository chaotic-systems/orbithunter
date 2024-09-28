orbithunter 1.3.0
==================

.. currentmodule:: orbithunter.shadowing

Description and Motivation
--------------------------

Shadowing module was really confusing; it was never clear what was actually being returned by `trim` and `map`
functions, the state the data was in, etc. Now, the various functions have been refactored into
the :func:`orbithunter.shadowing.OrbitCovering` class. The code has been changed so that the default
functionality is to produce all possible scores unless explicitly passed `pivot_mask` array.

Major Changes
-------------

-  Complete refactoring of OrbitCover -> OrbitCovering class to be more obvious and more accessible
-  `cover` has been converted into class method `score`
- `trim` has been converted into a class method
- `map` has been converted into a class method
- Parameters are now provided to respective class methods as opposed to :func:`orbithunter.shadowing.OrbitCovering.__init__`
- Array shape parameters such as hull, core, etc. are now always derived instead of provided.
- :func:`orbithunter.shadowing.OrbitCovering` class methods now return new class instances with transformed score array
- `joblib` and `tqdm` added to requirements


Minor Changes
-------------
- Handmade print statements changed to tqdm bars