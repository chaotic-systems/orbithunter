Major Changes
-------------
1. Errors in clipping window sizes correspondence with tile dimensions fixed.
2. Transferring functionality from rediscretize(Orbit, new_shape=tuple) -> Orbit.reshape(tuple) (numpy)
3. Conventions for RelativeOrbitKS shift have changed to better represent thesis details
4. Moving functions out of discretization.py
5. Moving functions only used in gluing.py to gluing.py....


Minor Changes
-------------
1. Tidying up of optimize.py functions and verbosity 
2. Fixed issues with convolutional neural nets and array shapes.
3. convolutional neural net now takes "dimension" keyword which determines the dimension of the convolution and
average pooling layers

Bug Fixes
---------
1. Equilibria with shape tuple of len 1 error fixed. 

Other
-----
1. Lots of notebook work
2. Defaults changed for _random_initial_condition
3. dimension_continuation bugs w.r.t. constraining parameters. 
4. Changed kwargs in calculate_spatial_shift