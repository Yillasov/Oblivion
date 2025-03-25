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
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Enable markdown files with CommonMark parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Enable CommonMark parser for markdown
import commonmark

def docstring_processor(app, what, name, obj, options, lines):
    md = '\n'.join(lines)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    lines.extend(rst.splitlines())

def setup(app):
    app.connect('autodoc-process-docstring', docstring_processor)