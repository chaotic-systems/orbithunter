Major Changes
-------------
1. How preconditioning is working in optimize.py
3. Integration fixed and passed preliminary testing

Minor Changes
-------------
1. Redid the default parameter based discretization values to be more efficient. i.e. FFT favors powers of 2, but a FFT
with 34 points is still faster than 64. 

Major issues
------------
1. There is a large error in the wrapping for LSQR and LSMR. Still working on how to fix it. 
2. The way that preconditioning is working for all methods other than gradient descent is incorrect I believe
3. matvec needs to be refactored I believe. 
