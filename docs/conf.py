# Configuration file for the Sphinx documentation builder.
#
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

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath("../fetch3/"))
sys.path.insert(0, os.path.abspath("../optimize/"))





# -- Project information -----------------------------------------------------

project = 'FETCH3.14'
copyright = '2022, Justine Missik'
author = 'Justine Missik'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'm2r2'


]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True  # Turn on sphinx.ext.autosummary
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_permalinks_icon = '<span>#</span>'
# html_theme = 'press'
# html_theme = 'sphinx_material'
# html_theme = 'sphinx_book_theme'
html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinxawesome_theme'
# html_theme = 'furo'

# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/jemissik/fetch3_nhl",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # Whether icon should be a FontAwesome class, or a local file
        }
   ]
}


# For todo extension
todo_include_todos = True

# Autosummary
# autosummary_generate = True
