Issues
======

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



