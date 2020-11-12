# Data description

All data relevant to orbithunter is stored in .h5 files. Each of these files contains
the information required to define a field of (1+1) spacetime dimensions which, if
the residual is small enough, satisfies the Kuramoto-Sivashinsky equation in 
the collocation sense.

The data is stored as a physical field because this is the most universal representation;
it does not rely on orbithunter specific FFTs to utilize for other purposes. 

The entries in the .h5 file are

```markdown
	discretization : dimensions of collocation grid
	field : orbit state in physical field basis
	parameters : tile dimensions and any extra parameters
	residual (optional) : the residual of the solution w.r.t. its governing equation. 
```

The last note is that some of these variables, e.g. time period for equilibria, spatial shift
for non-relative periodic solutions, are included to maintain a uniform save format. This
enables the user to import and instantiate the field data as having a different symmetry
if so desired. 