Patch notes beta v2.0.0

Major Changes
-------------
To maximize generability and efficiency, a number of major processes
have been overhauled. The largest of these three changes falls
into three categories.

1. Core
There have been a number of "revolutionary" changes to the main
framework posed in ```core.py```. Namely, the state generation
upon instantiation has been removed. While convenient for my purposes,
I realized that for equations with high dimensionality that this would
be a huge drag on efficiency. Now, instances which are not provided state
or parameter information will have its main data fields remain undefined.
Specifically, the parameters attribute will be None and the state attribute
will be initialized to an empty NumPy array whose shape is (0,0,0,0) for 
a 3+1 dimensional system, for example.  

2. IO
The hierarchical data format (hdf5) and the accompanying utilities
has been generalized to avoid equation specific naming conventions.

More importantly, I realized that I was drastically underutilizing
the benefit of such a format. In other words, instead of having a 
directory structure for file organization, everything is encapsulated into
the corresponding files. Therefore, instead of being strewn everywhere,
all OrbitKS data (at least that which does not serve a special purpose) 
is now in a single file named ```OrbitKS.h5```. I still need to figure out
the best place for users to retrieve the data, however.  

3. Gluing
-```orbithunter.gluing._correct_aspect_ratio``` has a new implementation, one which is
more general and much less confusing; not really necessary for the user but good to document
for the backend. It now allows for *more* distortion; in other words the strip-wise gluing should
be used with more care. 

4. Continuation
Name changes; instead of ```dimension_continuation```
it is simply continuation. The distinction is still made between this and ```discretization_continuation```. 
Additionally, more than one variable can be constrained at the same time now 
(although it is still not advised). To distinguish between the constraint being incremented and
others, the others need to be provided separately. 