Major Changes
-------------
1. Added new interface for default optimization values via keyword "precision" and "computation_time" 
2. Changing the arguments for gluing, clipping until satisfied.


Minor Changes
-------------
1. Changing the arguments for _slices_from_window  in clipping module.
2. Adding persistent homology wrappers. 

Bug fixes
---------

1. EquilibriumOrbitKS()'s __init__ was overwriting L parameter with 0, leading to division by 0. 
2. Fixed bugs related to discretization in _correct_aspect_ratios
3. Fixed edge case of "gluing" a single symbol.
