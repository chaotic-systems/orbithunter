orbithunter 1.0.6
=================

.. currentmodule:: orbithunter

Major Changes
-------------
- Shadowing handles masking differently. Now OrbitCover attribute `scores` is a numpy masked array so that the mask
  does not need to be tracked or confused with the other type of mask, the scoring pivot mask which exists
  simultaneously.
- Removed early termination from `:func:shadowing.cover`, as the number of originally unmasked pivots and the number
  of valid pivots are not necessarily the same, leading to unintended behaviors.

Bug and Error Fixes
-------------------

