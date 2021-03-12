orbithunter v0.3.0

Major Changes
-------------
1. "generate" translated to "populate" because I felt it was more reflective of the in-place operation.

2. More tests for core and KSE, reorganized and combined into single tests_basic.py 

Minor Changes
-------------


Bug fixes
---------
1. hdf5 metadata is imported as list; new "fast" init method didn't type check the 'discretization' variable
and so it was incorrectly being assigned as a list, causing issues for .resize(), which compares 'discretization'
to a tuple. 



