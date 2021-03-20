orbithunter 0.5rc1
==================

.. currentmodule:: orbithunter

Minor Changes
-------------

- staticmethod bases renamed to bases_labels to better match other staticmethods
- :meth:`orbithunter.ks.RelativeOrbitKS.change_reference_frame` keyword argument changed from 'to' to 'frame' because
  of potential conflicts with :meth:`orbithunter.ks.RelativeOrbitKS.transform`.
  
  
Other
-----

- Large expansion of docs, I believe it merits a rough draft ready for `readthedocs <http://readthedocs.org/>`_



