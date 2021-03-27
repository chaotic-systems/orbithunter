Issues
======

.. currentmodule:: orbithunter

** Trust Region (second-order, Hessian based) Numerical Methods ** 

Using the built-in finite difference strategies 


** Conflicts with SciPy kwargs **

Certain :func:`orbithunter.optimize.hunt` keyword argumentts are conflicting with keyword arguments of SciPy routines.
For example, the ``maxiter`` keyword argument which is meant to control the 'outer iteration' of various methods is
being passed to ``scopy.optimize.newton_krylov`` and affecting the ability to converge there; as it is saying essentially
only use the first Krylov vector $Jx$ to form the Krylov subspace. 

**Slicing**

The current implementation :meth:`orbithunter.core.Orbit.__getitem__` infers the sliced state array's
dimensions from the shape of the sliced array. NumPy's `advanced indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
allows for many different combinations and types of keys: ``int, Ellipsis, tuple, slice, bool, np.ndarray`` and more
can be used for indexing. The inference and computation of the new :meth:`Orbit.dimensions` based on this shape but
becomes nonsensical when the number of axes changes; we could just set any removed axes dimension to be equal to zero,
however, there is a critical ambiguity which makes this more difficult than it may seem. For example assume the
state array has shape (2, 2, 2, 2). Slicing via state[:, 0, : , :] and state[:, :, 0, :] will both result in an
array of shape (2, 2, 2); but they are not physically equivalent, as one should be (2, 1, 2, 2) and the other
should be (2, 2, 1, 2). This can technically be alleviated by using a slice instead, i.e. state[:, :1, :, :] but
this leaves much to be desired.

**Parameter parsing/passing**

Creating an instance w/o parsing; by passing values for the five main attributes allows for
fewer parameters than labels; normally this would be filled with zeros. If not designed carefully, then an IndexError
may be raised when trying to access a labelled parameter attribute that doesn't exist. This is being recorded here
because while it is actually intended, it can still be confusing in the traceback; chaining an AttributeError in
future release to indicate this is where this is coming from. The clause in :meth:`Orbit.__init__` details this
somewhat but I feel it needs to be more obvious. 

**Shadowing**

The mapping from pivot scores to spacetime regions of the base orbit field
is holding the performance of :func:`orbithunter.shadowing.shadow` and :func:`orbithunter.shadowing.cover`
back. It's still required for :func:`orbithunter.shadowing.fill`, but currently
the distinction between cover and fill seems blurred. It is still useful to map the scores
to orbit spacetime for masking purposes, though. Therefore the changes moving forward are going
to map scores only if the threshold is met; it's not very useful information otherwise. 

**OrbitKS and its subclasses' speed**

Future releases will allow for assignment of ```Orbit.workers = int``` so that SciPy parallelism can be
utilized in OrbitKS transforms; included in ``Orbit`` so others can use as well. This will increase speed
of numerical calculations; in the large dimension limit. Additionally, it was realized that the current
deployment of how Jacobian matrices are constructed is incredibly inefficient and scales terribly. 
A test case of rewrite brought the time (for a very large array) from 6 minute to 1 minute and RAM usage
from 12 gb to 8 gb. Theoretical lower bound for RAM usage in this case is 2 GB but I believe it would
require allowing overwrites in the SciPy FFTs which would require yet another property akin to ``workers``
unless inplace operations are included in the transform methods. Still under development. 




