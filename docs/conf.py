# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Oblivion SDK'
copyright = '2023, Yessine'
author = 'Yessine'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'myst_parser',  # Add MyST parser for markdown
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Enable markdown files with MyST parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Create _static directory if it doesn't exist
import os
if not os.path.exists(os.path.join(os.path.dirname(__file__), '_static')):
    os.makedirs(os.path.join(os.path.dirname(__file__), '_static'))