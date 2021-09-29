Install
=======

To get the latest stable release of orbithunter running locally, the package can either be installed locally via
the Python package installation manager `pip` or can be accessed via a Docker container.

Docker installation
-------------------

To facilitate scientific developments outside of the core development, the latest release has been used
to create a Docker image. To the uninitated, Docker is a platform for containerizing applications, which, broadly
speaking, makes it so the Python dependencies and issues arising from differences in operating systems can be avoided.
The Docker image itself runs a jupyter notebook kernel, allowing the jupyter notebook GUI to be used in the local
machine's browser. Crudely speaking, the containing is acting as a server/virtual machine,
serving the jupyter notebook application, saving all local files and code to the Docker container.

Important notes: the port opened to allow the local machine to interact with the container is hard-coded to be 8887,
meaning that trying to run two containers on the same machine won't work currently. This is simply because I'm new
to Docker and haven't learned how to do anything else yet.

Acquiring the Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^

In exchange for avoiding the local installation of Python, orbithunter and its requirements the user must install
the `Docker application <https://www.docker.com/products/docker-desktop>`_ . During the installation process, Docker
is going to ask for permissions multiple times and might require the installation of WSL 2 files (linux
compatibility) if not already present.

Once the application is installed, the Docker image can be pulled from the remote repository by opening command line
and typing::

    docker pull orbithunter/orbithunter:latest

At which point, the image will be downloaded. This image will be displayed in the Docker application under
the Local portion of the Images tab.

Running the Docker image
^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to instantiate the container is to run the following in command line, replacing `<container name>`
with whatever is the desired name.::

    docker run --name <container name> -p 8887:8887 orbithunter/orbithunter:latest

The `-p` switch opens the local port 8887 allowing for the jupyter notebook GUI to be opened in a browser. This prints
output, the last bit of which will look like the following.
The notebook accomplished by copying the last (for some reason it has to be the latter) URL to a browser (only ever
been tested on chrome + windows). ::

    Or copy and paste one of these URLs:
        http://e3954f15092d:8887/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
     or http://127.0.0.1:8887/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Once the jupyter GUI is open, it can be utilized as per usual. All computations and output data
will be saved to the container; which will be randomly named if no name was provided. To restart the container
after it has been turned off, run the command::

    docker start <container_name>


Installation via pip
--------------------

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

Install the current release of ``orbithunter`` with ``pip``::

    pip install orbithunter

To upgrade to a newer release use the ``--upgrade`` flag::

    pip install --upgrade orbithunter

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    pip install --user orbithunter

Alternatively, you can manually download ``orbithunter`` from
`GitHub <https://github.com/orbithunter/orbithunter>`_  or
`PyPI <https://pypi.python.org/pypi/orbithunter>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    pip install .

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

    pip install orbithunter[extra]

To explicitly install all optional packages, do::

    pip install ipykernel jupyterlab ipython gudhi pot scikit-learn tensorflow

Or, install any optional package (e.g., ``gudhi``) individually::

    pip install gudhi



.. warning::
	To get `POT <https://pythonot.github.io/>`_ to install correctly, you need to have cython and
	numpy BEFORE you run::

		pip install pot



Testing
-------

Orbithunter uses the Python ``pytest`` testing package.  You can learn more
about pytest on their `homepage <https://pytest.org>`_.

Test a source distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^

After navigating to the downloaded source directory, the tests can be evaluated by the following commands::

	pytest .

or for more control, pytest arguments can be included

    pytest --pyargs .
	
or for individual files e.g. ``test_basic.py``
	
	pytest --pyargs test_basic.py

For those that are unaware ``.`` is synonymous with "evaluate in the current directory". Pytest will automatically
search for the tests folder and any file that begins with the prefix "test". 