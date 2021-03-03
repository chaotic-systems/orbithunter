orbithunter v0.1.0
----------------

Major changes
-------------
1. Re-do of the initial condition generation for KSE. For more control, Fourier mode
modulation split into spatial and temporal components. 
2. Added a masking method for Orbit class for convenience. 
3. "fill" algorithm; very similar to cover algorithm and in fact improves handling of boundaries.
Fill disallows overlaps, equivalent to the replacement=False setting for cover. 
4. Tweaked the default discretization sizes somewhat. 

Notes
-----
The previous 'cover' algorithm did not take into account for shadowing along the boundaries
correctly. This is fixed in 'fill' but the changes for cover are postponed for now. 
The idea of scanning_mask function is bad. While keeping track of the pivots and whether or not
they are filled is different than tracking the filling of the orbit masks. However, this was
based upon the "strides" subsampling aspect, which in hindsight is not very useful. 
Changing in next version. 