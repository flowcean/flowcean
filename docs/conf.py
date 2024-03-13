import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))

project = "AGenC"
copyright = "2023, The authors of the AGenC project"
author = "The authors of the AGenC project"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.spelling",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "label"
html_theme = "furo"
myst_enable_extensions = [
    "colon_fence",
]
myst_footnote_transition = False
suppress_warnings = [
    "myst.footnote",
]