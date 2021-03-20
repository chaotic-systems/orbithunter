Install
=======

orbithunter requires Python 3.7, 3.8, or 3.9.  If you do not already
have a Python environment configured on your computer, please see the
instructions for installing the full `scientific Python stack
<https://scipy.org/install.html>`_.

It is assumed that the default Python environment already configured on
your computer and you intend to install ``orbithunter`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------


Install the current release of ``orbithunter`` with ``pip``::

    $ pip install orbithunter

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade orbithunter

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user orbithunter

Alternatively, you can manually download ``orbithunter`` from
`GitHub <https://github.com/mgudorf/orbithunter>`_  or
`PyPI <https://pypi.python.org/pypi/orbithunter>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Extra packages
--------------

.. note::
   Some optional packages are required for full functionality of all orbithunter modules.
   The two modules which are not supported by the default install are :mod:`orbithunter.persistent_homology`
   and :mod:`orbithunter.machine_learning`. These act as an API that allows interaction with 
   ``gudhi``, ``scikit-learn``, and ``tensorflow`` packages.

The following extra packages provide additional functionality. See the
files in the ``requirements/`` directory for information about specific
version requirements.

- `Gudhi <http://pygraphviz.github.io/>`_ and `Python optimal transport <https://pythonot.github.io/>`_ for topological data analysis
- `scikit-learn <https://scikit-learn.org/stable/>`_
- `tensorflow <https://www.tensorflow.org/>`_

To install ``orbithunter`` and extra packages, do::

    $ pip install orbithunter[extra]

To explicitly install all optional packages, do::

    $ pip install ipykernel jupyterlab ipython gudhi pot scikit-learn tensorflow

Or, install any optional package (e.g., ``gudhi``) individually::

    $ pip install gudhi

To get `POT <https://pythonot.github.io/>`_ to install correctly, you need to have cython and
numpy BEFORE you run::

	$ pip install pot

Testing
-------

Orbithunter uses the Python ``pytest`` testing package.  You can learn more
about pytest on their `homepage <https://pytest.org>`_.

Test a source distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can test the complete package from the unpacked source directory with::

    pytest orbithunter

Test an installed package
^^^^^^^^^^^^^^^^^^^^^^^^^

From a shell command prompt you can test the installed package with::

   pytest --pyargs orbithunter

