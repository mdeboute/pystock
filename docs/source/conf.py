import pystock

project = "pystock"
copyright = "2025, Martin Debouté"
author = "Martin Debouté"
release = pystock.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "sphinx.ext.coverage",
]
auto_doc_default_options = {"autosummary": True}

exclude_patterns = []

html_theme = "pydata_sphinx_theme"
