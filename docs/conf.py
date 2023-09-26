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
]
html_theme = "furo"
myst_enable_extensions = [
    "colon_fence",
]
