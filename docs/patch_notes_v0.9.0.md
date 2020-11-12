Patch notes v0.9.0

Major changes
-------------
1. Massive reorganization and refactoring to increase modularity 
2. Removal of many utilities from main `orbithunter.__init__` to clean up the namespace 
2. Removal of all default pieces of .io functions, now imports KS presets. 
3. Renaming of `convert` to `transform`
5. Reorganization of KS data 
6. Changed how orbits are constrained in continuation
7. Moved all KS specific code to its module; affects `io.py` and `gluing.py`


Minor changes
-------------
1. Antisymmetric subspace integration included in KSE integration function.
2. KSE integration now needs specific important from orbithunter.ks.physics. 

Bug fixes
---------
1. Fixed bug where continuation was overshooting the bounds in `dimension_continuation`

Known bugs/potential issues
------------------
If orbit is transformed into RelativeOrbitKS, it is unable to plot the field in the fundamental
domain unless explicitly manipulated. As it is the orbithunter policy to in fact disallow most
operations in the physical frame, I am leaving this as it is.    

To-do
-----
1. Documentation needs a lot of work
2. Notebooks need to be cleaned up