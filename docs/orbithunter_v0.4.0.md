orbithunter v0.4.0

Major Changes
-------------
1. Discrete symmetry subclasses now handle bases differently. They force odd ordered derivatives to
occur in the spatial basis. This leads to more sensible results instead of the zero-valued field that
would occur otherwise. 

2. More tests for core and KSE, fixed and expanded the derivative norm tests; they hadn't taken the
projections from transform operators into account. 

Bug fixes
---------
1. New iteration of self.constrain() was incorrectly assigning False to parameters that were not in 
self.parameter_labels(), causing self.from_numpy_array() to fail. 

Notes
-----
Getting readthedocs and github pages setup along with setup. 