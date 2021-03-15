#!/usr/bin/env python3
""" orbithunter serves as a framework for solving chaotic nonlinear partial differential equations.

It provides:

- An object oriented approach to solutions of partial differential equations
- A general-purpose framework for finding, visualizing and manipulating these solutions
- High-level access to SciPy API, particularly its minimize and optimize packages.
- New spatiotemporal techniques developed by Matthew Gudorf

Orbithunter uses NumPy (https://github.com/numpy/numpy) and SciPy for its numerical calculations.
Its design emphasizes user-friendliness and modularity; giving quick and easy access to
high-level numerical operations.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
# long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='orbithunter',  # Required

    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.4b1',  # Required
    description='Framework for solving spatiotemporal partial differential equations.',
    # long_description=long_description
    # long_description_content_type='text/markdown',

    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional

    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://mgudorf.github.io/orbithunter/',  # Optional
    author='Matthew Gudorf',
    author_email='matthew.gudorf@gmail.com',  # Optional
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License'
    ],
    keywords=[
        'pde',
        'partial differential equation',
        'numeric',
        'numerical simulation',
        'solver',
        'framework',
        'periodic orbit'
    ],
    #
    # package_dir={'': 'orbithunter'},  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().

    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    py_modules=['orbithunter.persistent_homology',
                'orbithunter.machine_learning',
                'orbithunter.clipping',
                'orbithunter.continuation',
                'orbithunter.core',
                'orbithunter.gluing',
                'orbithunter.io',
                'orbithunter.optimize',
                'orbithunter.shadowing'],

    packages=find_packages(include=['orbithunter', 'orbithunter.*']),
                           # exclude=["data"]),
    # include_package_data=False,
    # package_data={'': ['data']},
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.7',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.rg/en/latest/requirements.html
    install_requires=['matplotlib>=3.1.3',
                        'numpy>=1.18.1',
                        'scipy>=1.4.1',
                        'pytest>=5.3.5',
                        'h5py>=2.10.0'],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'test': ['pytest'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'bug reports': 'https://github.com/pypa/sampleproject/issues',
        'documentation': 'https://readthedocs.org/projects/orbithunter/',
        'source': 'https://github.com/mgudorf/orbithunter/',
        'tracker': 'https://github.com/orbithunter/docs/issues',
        'home page': 'https://mgudorf.github.io/orbithunter/',
        'tutorial': 'https://github.com/mgudorf/orbithunter/tree/main/notebooks',
        'frequently asked questions': 'https://github.com/mgudorf/orbithunter/tree/main/docs/faq.md'
    }
)

# import pathlib
# from setuptools import setup
#
# # The directory containing this file
# HERE = pathlib.Path(__file__).parent
#
# # The text of the README file
# README = (HERE / "README.md").read_text()
#
# # This call to setup() does all the work
# setup(
#     name="realpython-reader",
#     version="1.0.0",
#     description="Read the latest Real Python tutorials",
#     long_description=README,
#     long_description_content_type="text/markdown",
#     url="https://github.com/realpython/reader",
#     author="Real Python",
#     author_email="info@realpython.com",
#     license="MIT",
#     classifiers=[
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.7",
#     ],
#     packages=["reader"],
#     include_package_data=True,
#     install_requires=["feedparser", "html2text"],
#     entry_points={
#         "console_scripts": [
#             "realpython=reader.__main__:main",
#         ]
#     },
# )