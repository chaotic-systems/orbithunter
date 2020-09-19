Minor changes
-------------
1. Replaced the ubiquitous usage of individual parameter kwargs "T=self.T, L=self.L, S=self.S" to
'orbit_parameters=self.orbit_parameters'. l.

Bug fixes
---------
1. Fixed bugs in changing reference frames which incorrectly overwrote self.frame='physical'. 
3. Fixed runtime warnings that occurred in optimize.py when discretization was very small.