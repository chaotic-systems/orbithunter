Notes on scipy.optimize.minimize numerical methods that require Hessian matrix (or its product H*x)

From scipy docs:
```
Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

Method trust-ncg uses the Newton conjugate gradient trust-region algorithm [5] for unconstrained minimization. 
This algorithm requires the gradient and either the Hessian or a function that computes the product of the Hessian with a given vector. Suitable for large-scale problems.

Method trust-krylov uses the Newton GLTR trust-region algorithm [14], [15] for unconstrained minimization.
This algorithm requires the gradient and either the Hessian or a function that computes the product of the Hessian with a given vector. 
Suitable for large-scale problems. 
On indefinite problems it requires usually less iterations than the trust-ncg method and is recommended for medium and large-scale problems.

Method trust-exact is a trust-region method for unconstrained minimization in which quadratic subproblems are solved almost exactly [13].
This algorithm requires the gradient and the Hessian (which is not required to be positive definite).
It is, in many situations, the Newton method to converge in fewer iteraction and the most recommended for small and medium-size problems.
```

Right now the Hessian is somewhat annoying to implement so I am avoiding these methods for now 09/07/2020