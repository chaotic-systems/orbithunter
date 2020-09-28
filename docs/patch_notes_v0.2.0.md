Major Changes
-------------
1. Breaking up parameters property into its components. The way it was previously written was trying to do
exactly what parameters do, except bundling them in a single dict. Nonsensical in the context of classes.
2. Redid continuation.py to account for Orbits of Equilibrium type, in addition it now works on more than one dimension
at a time. 
3. 'rmatvec_parameters' method added to account for differences in default parameters for rmatvec matrix-vector
product calculations, mainly because of RelativeOrbit type spatial shifts not being initialized properly. 

Minor Changes
-------------

1. Added convenience function for rediscretizing/normalizing the discretizations of tiling dictionaries. 
2. Added convenience function for generating all symbol array for different symbol array shapes, in addition
to selecting only the symbol arrays unique up to rotations.
3. Changed default tolerances, got rid of "descent factor" 
4. Changed how verify_integrity works, now looks at the field norm instead of mode, also a less strict threshold. 



Bug fixes
---------
1. Issues related to how time discretization was being stored/tracked for RelativeEquilibriumOrbitKS and
EquilibriumOrbitKS. Will update docs to very explicitly describe what can and cannot be done. 