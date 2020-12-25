Major Changes
-------------
1. Renamed persistent homology functions
2. Renamed variables and keyword arguments in continuation functions
3. Moving as much as possible to the Orbit class defined in core.py

Minor Changes
-------------
1. Clipping now accepts class constructors
2. Fixed issues with convolutional neural nets and array shapes.
3. convolutional neural net now takes "dimension" keyword which determines the dimension of the convolution and
average pooling layers

Bug Fixes
---------
1. Equilibria were supposed to be detected by the magnitude of the time derivative but if the time period was 0 then
it was unable to calculate the time derivative. 

Other
-----
1. Cleaning up the official data and figs to be useful.
2. Adding more to notebooks