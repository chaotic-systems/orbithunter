Issues
======

.. currentmodule:: orbithunter

** Trust Region (second-order, Hessian based) Numerical Methods ** 

Using the built-in finite difference strategies 


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


** Constrained optimization ** 

Currently constraints are handled on the orbithunter end by simply disallowing changes to be made to the parameter
independent of the return of the optimization function. The equations which explicitly construct a matrix do not include
these constants in the optimization but (incorrectly) they are included in the scipy functions due to the orbit_vector
definition not taking constraints, which I believe is just an error on my part. 




