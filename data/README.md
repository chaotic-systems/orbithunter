# Data description

All data relevant to orbithunter is stored in .h5 files. Each of these files contains
the information required to define a field of (1+1) spacetime dimensions which, if
the residual is small enough, satisfies the Kuramoto-Sivashinsky equation in 
the collocation sense.

The data is stored as a physical field because this is the most universal representation;
it does not rely on orbithunter specific FFTs to utilize for other purposes. 

The entries in the .h5 file are

```markdown
	'time_discretization' : The number of collocation points in the time dimension (rows of field array)
	'space_discretization' : The number of collocation points in space (columns of field array)
	'space_period' : The spatial extent of the field
	'time_period' : The temporal extent of the field
	'spatial_shift' : The SO(2) shift for relative periodic solutions, 0 for solutions w/o this symmetry
	'residual' : The value of the cost function 1/2 |F|^2. A value equal to zero means that the field is a solution.
	'field' : The value of the scalar velocity field u(x,t) at the spatiotemporal collocation points.  
```

The last note is that some of these variables, e.g. time period for equilibria, spatial shift
for non-relative periodic solutions, are included to maintain a uniform save format. This
enables the user to import and instantiate the field data as having a different symmetry
if so desired. 