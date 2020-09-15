Major Changes
-------------
1. Added Orbit() super class for generalization to equations other than Kuramoto-sivashinsky. 
2. Re-did gluing to generalize
3. Re-did discretization to generalize. 
4. Renamed files 
5. To maintain a constant signature in __init__ functions, re-did how keyword arguments are handled.

Minor Changes
-------------
1. optimize.py now only refers to the product of the orbit state's shape to get default max iterations
and tolerances, as opposed to KSE specific parameters.
2. Similar to 1. except for the function's defined with _scipy_sparse_linalg_solver_wrapper.


Bug fixes
---------
1. EquilibriumOrbitKS()'s __init__ was overwriting L parameter with 0, leading to division by 0. 