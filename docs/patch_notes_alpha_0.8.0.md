Patch notes v0.8.0

Major changes
-------------
1. mask_orbit : now accepts an iterable of window dimensions rather than a single window. 
2. Preconditioning defaults and their inclusion in jacobian removed. 
3. adjoint/transpositions are now represented by their own functions
4. glue parameters updated for subclasses. 
5. shadowing function added to physics_ks, calculates norm between window
and subdomains of a trajectory or orbit; returns a masked orbit where the
unmasked regions correspond to small values in the l2 norm of the difference of squares,
$|window^2 - base^2|$. 

Minor changes
-------------

1. Defaults for calculate spatial shift changed
2. time integration can now be chosen to store only a single point
or trajectory segment. 
3. Verbosity print outs in optimize.py and removal of OrbitResult for now. 

Known bugs/issues
------------------
Plotting when RelativeOrbitKS when in physical reference frame prevents plotting comoving frame. 
Technically the documentation implies that every operation should be performed in the comoving frame
so this is unintended but not 