orbithunter 0.6.0
=================

.. currentmodule:: orbithunter

Major bug fix from refactoring

This is 


Major Changes
-------------

The mapping from pivot scores to spacetime regions of the base orbit field
is holding the performance of :func:`orbithunter.shadowing.shadow` and :func:`orbithunter.shadowing.cover`
back. It's still required for :func:`orbithunter.shadowing.fill`, but currently
the distinction between cover and fill seems blurred. It is still useful to map the scores
to orbit spacetime for masking purposes, though. Therefore the changes moving forward are going
to map scores only if the threshold is met; it's not very useful information otherwise. 



Misc
----

- More docs



