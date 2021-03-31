# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath(".."))

# The full version, including dev info
import orbithunter

version = orbithunter.__version__

# The short X.Y version.
release = orbithunter.__version__[:3]

# -- Project information -----------------------------------------------------
project = "orbithunter"
copyright = "2021, Matthew Gudorf"
author = "Matthew Gudorf"

# generate autosummary pages
autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False


# -- General configuration ---------------------------------------------------
master_doc = "index"
extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.log_cabinet",
    "sphinx_issues",
    "sphinx_rtd_theme",
    # "sphinx.ext.autosectionlabel",
    "nb2plots",
    "texext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "changelog/*"]


# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {"collapse_navigation": True, "sticky_navigation": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
