from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.rst").read_text(encoding="utf-8")

setup(
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="orbithunter",  # Required
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.5b1",  # Required
    description="Framework for solving spatiotemporal partial differential equations.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://mgudorf.github.io/orbithunter/",  # Optional
    author="Matthew Gudorf",
    author_email="matthew.gudorf@gmail.com",  # Optional
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10"
    ],
    keywords=[
        "pde",
        "partial differential equation",
        "numeric",
        "numerical simulation",
        "solver",
        "framework",
        "periodic orbit",
    ],
    py_modules=[
        "orbithunter.persistent_homology",
        "orbithunter.machine_learning",
        "orbithunter.clipping",
        "orbithunter.continuation",
        "orbithunter.core",
        "orbithunter.gluing",
        "orbithunter.io",
        "orbithunter.optimize",
        "orbithunter.shadowing",
    ],
    packages=find_packages(include=["orbithunter", "orbithunter.*"]),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3.1.3",
        "pyqt5>=5.15.4",  # matplotlib backend prone to failure; this is immediate remedy.
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "h5py>=2.10.0",
    ],
    extras_require={  
        "test": ["pytest>=5.3.5", "pytest-datafiles>=2.0"],
        "notebooks": ["ipykernel>=5.1.4", "jupyterlab>=1.2.6"],
        "tools": ["gudhi", "scikit-learn", "tensorflow"],
    },
    project_urls={
        "bug reports": "https://github.com/mgudorf/orbithunter/issues",
        "documentation": "https://readthedocs.org/projects/orbithunter/",
        "source": "https://github.com/mgudorf/orbithunter/",
        "tracker": "https://github.com/orbithunter/docs/issues",
        "home page": "https://mgudorf.github.io/orbithunter/",
        "tutorials": "https://github.com/mgudorf/orbithunter/tree/main/notebooks",
        "frequently asked questions": "https://github.com/mgudorf/orbithunter/tree/main/docs/faq.rst",
    },
)
