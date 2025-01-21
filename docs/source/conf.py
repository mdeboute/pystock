import datetime
import os
import sys

import pystock

sys.path.insert(0, os.path.abspath("../../"))

project = "pystock"
copyright = "%s, Martin Debouté" % str(datetime.datetime.now().year)
author = "Martin Debouté"
release = pystock.__version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "autodocsumm",
    "sphinx_design",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.programoutput",
]
autodoc_default_options = {
    "autosummary": False,
    "imported-members": False,
    "exclude-members": "Path",
    "undoc-members": True,
}

autoclass_content = "both"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_css_files = ["css/theme.css"]
html_theme_options = {
    # "icon_links": [
    #     {
    #         "name": "Website",
    #         "url": "https://...",
    #         "icon": "_static/assets/logo.png",
    #         "type": "local",
    #     },
    # ],
    "footer_start": ["copyright"],
    "footer_end": [],
    "navbar_end": ["navbar-icon-links"],
    "header_links_before_dropdown": 4,
}

html_sidebars = {"release_notes": [], "usage/index": [], "installation": []}

html_context = {"default_mode": "light"}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
# html_logo = "_static/assets/logo.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#
# html_favicon = "_static/assets/favicon-32x32.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "pystock_doc"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

add_module_names = False
always_document_param_types = True
typehints_defaults = None
html_title = "pystock"
html_show_sourcelink = False
