import os
import sys

extensions = [
    "sphinx.ext.autodoc",  # Imports modules and docs
    "sphinx.ext.intersphinx",  # Links to external libs docs
    "sphinx.ext.napoleon",  # Converts docs to rst format
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

source_suffix = [".md", ".rst"]
master_doc = "index"

# General information about the project.
project = "AEStream"
# copyright = ""
# author = ""

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.6"
# The full version, including alpha/beta/rc tags.
release = "0.6.2"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "source_edit_link": "https://github.com/norse/aestream/edit/feature-docs/docs/{filename}",
}


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None


# Output file base name for HTML help builder.
htmlhelp_basename = "aestreamdoc"
