orbithunter v0.2.0
----------------

Major changes
-------------
1. Nearly full rewrite of shadowing.py; fill and cover now not only allow for partial
overlaps with windows that are out of bounds, but now they also accept functions as keyword arguments
for a variety of controls, the most important of which is coordinate mapping. 

"Coordinate mapping" is the process of taking a d-dimensional array of indices and mapping them such that
they can be used to slice complex geometries, more complex than d-dimensional hypercubes of field values.
The inspiration behind this was the parallelipiped like drifting orbits seen in KSe flow, where regions of
nonzero local galilean velocity are witnessed. This functionality is accessed via the keyword argument 
'coordinate_map'. These functions must take input of the form np.indices(), which is an axis of shape (d, window_shape).
The functions must return an array of the shape (d, ...) such that the i-th entry (i, ...) represents the coordinates
in the i-th axis. 

This functionality can be arbitrarily complex; partial overlaps/out of bounds and wrapping around periodic axes 
is also included to allow for anything that I can possibly imagine anyone wanting for the time being. 

pivot_iterator is back but awkward utilities like scanning_dimensions are gone. Additionally, the returning
of dictionaries with awkwardly shaped masks as values have been replaced by numpy arrays; the "key" is now simply
the index of the first axis. 

2. Moved some symmetry operations from ks/orbits.py to core.py, roll and cell_shift, as they were generalizable anyway.

3. Changed __init__ to account for "fast" or "smart" initialization; when class instances are constructed 
by providing all attribute information, parsing is not required, and hence should not be used as it slows things down. 
