from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.rst").read_text(encoding="utf-8")

with open("orbithunter/__init__.py") as file:
    for line in file:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break


def parse_requirements_file(filename):
    with open(filename) as file:
        requires = [l.strip() for l in file.readlines() if not l.startswith("#")]
    return requires


install_requires = parse_requirements_file("requirements/default.txt")
extras_require = {
    dep: parse_requirements_file("requirements/" + dep + ".txt")
    for dep in ["developer", "docs", "extra", "full", "test"]
}

setup(
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="orbithunter",  # Required
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,  # Required
    description="Framework for Nonlinear Dynamics and Chaos",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://orbithunter.readthedocs.io/en/latest/index.html",  # Optional
    author="Matthew Gudorf",
    author_email="matthew.gudorf@gmail.com",  # Optional
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows :: Windows 10",
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
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require=extras_require,
    project_urls={
        "documentation": "https://orbithunter.readthedocs.io/en/latest/index.html",
        "issues": "https://orbithunter.readthedocs.io/en/latest/issues.html",
        "bugs": "https://github.com/mgudorf/orbithunter/issues",
        "source": "https://github.com/mgudorf/orbithunter",
        "guide": "https://orbithunter.readthedocs.io/en/latest/guide.html",
        "tutorials": "https://github.com/mgudorf/orbithunter/tree/main/notebooks",
        "faq": "https://orbithunter.readthedocs.io/en/latest/faq.html",
    },
    package_data={"orbithunter": ["requirements/*.txt"]},
)
