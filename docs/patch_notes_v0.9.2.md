Patch notes v0.9.1 

Major changes
-------------


Minor changes
-------------
Rescale method removed from init, also will always return in modes basis for KS. Onus put on the user to perform
extra operations. 

Other changes
-------------
Refactored _pad methods using numpy's pad for readability; slightly slower but this operation
is typically not used with high frequency. 

Bug fixes
---------


Known bugs/potential issues
------------------
1. KSe shadowing, masking not returned properly; can be accessed via returned orbit's state.
2. Not providing nonzero_parameters=True yields division by 0; i.e. OrbitKS() disallowed, for now.

To-do
-----
1. Documentation needs a lot of work
2. Notebooks need to be cleaned up
3. Removal of all "local" directory references.
4. Inclusion of example data. 