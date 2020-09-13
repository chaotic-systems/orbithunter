This document serves as the beginning for the semantic versioning of orbithunter, version 0.0.

Version 0.0 notes:

Orbithunter core
----------------

orbit.py

Major changes
-------------
1. Refactoring of __init__, __repr__
2. from_numpy_array completely reworked 
3. self.status() merged with verify_integrity() from io.py into self.verify_integrity()
4. _parse_parameters method. 
4. New method of passing parameters as dictionary.
   This allows for less esoteric referencing of parameters, for example parameters[3] is
   much less legible than parameters['N'], as 'N' is a mandatory attribute. Unfortunately 
   for caching the parameters passed to differentiation is still in tuple form. Could used
   collections.namedtuple but its not really worth the additional import. 
5. The way that passing numerical constraints on parameters has been completely revamped. 
   Constraints are now an attribute which is assigned in 
6. Plotting changes to better accomodate relative sizes between orbits. 
7. random_initial_conditions() method written with multiple options. 
8. The time discretization of EquilibriumOrbitKS and RelativeEquilibriumOrbitKS 
   will be maintained during transformations to and from the spatiotemporal mode basis (and
   change of reference frames for RelativeEquilibriumOrbitKS) but any other operations 
   in the spatiotemporal mode basis will save the state as having N=1. This is to
   maintain the parsing of the state's shape parameters instead of passing them as arguments. 

Minor changes
1. Renamed transform methods to be private methods to indicate to use convert instead. 
2. return_modes keyword changed to return_array to prevent confusion of returning an instance in
	modes basis vs. returning a numpy array of modes.
3. Parameters passed to differentiation functions now a part of parameter dictionary: note, must be tuple
   to be hashable for lru_cache. 
4. 

Major fixes
Time transform matrices for EquilibriumOrbitKS and RelativeEquilibriumOrbitKS and
normalization factor fixes. 


arrayops.py
1. Calculate spatial shift moved from RelativeOrbitKS method to function.
   Also, bug fixes so that it is now working as intended. 
   
discretization.py
1. _parameter_based_discretization changed to accept new parameters format. 

generate.py
1. Everything removed to be replaced. 

glue.py (name change to _glue)
1. concat -> glue
2. glue function rewritten, other functions need to be rewritten still. 
3. helper functions for KS tiling. 

integration.py
1. integration function changed to accept and return orbit instances.
2. integrate -> integrate_kse because it does not generalize at all.

io.py
1. changed how filenames are parsed; may have introduced bugs. 
2. Added flag for whether or not to check the integrity of the import. 

optimize.py
3. Renamed many keyword arguments and regularized them between all functions so that
orbithunter kwargs are distinct from scipy kwargs. 

