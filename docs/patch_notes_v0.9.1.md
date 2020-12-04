Patch notes v0.9.1 

Major changes
-------------
1. Massive reorganization and refactoring to increase modularity 
2. More renaming of methods for brevity. 


Minor changes
-------------

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