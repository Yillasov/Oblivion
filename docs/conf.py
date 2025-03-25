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
    'sphinx_markdown_builder',  # Using sphinx-markdown-builder
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Enable markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}