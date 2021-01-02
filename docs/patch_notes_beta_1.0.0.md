Patch notes beta v1.0.0

Relabelling the semantic versioning as beta because of not spacing it properly in the original go around.

Major changes
-------------
- Entirety of hdf5 usage
- Unincluded Gudhi in core 
- Migration of classmethods to arrayops
- Renaming of many functions 
- Tileset and importation utilities removed/replaced with importlib & getattr calls
- Many different/redundant properties and bad design choices removed; much cleaner now.

Minor changes
-------------
-Glue parameters now generalized
-dx(), dx_matrix(), etc. now generalized

Other changes
-------------
- Added/correction documentation

Bug fixes
---------


Known bugs/potential issues
------------------
1. KSe shadowing, masking not returned properly; can be accessed via returned orbit's state.
2. _correct_aspect_ratios can only handle even discretizations, biased towards kse. 

To-do
-----
1. Documentation needs to be updated/cleaned up work
2. Notebooks need to be cleaned up
4. Inclusion of example data/finding a place to host the remainder. 