import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'QSER'
copyright = '2026, Ahmad Muhammad and Fatih Külahcı'
author = 'Ahmad Muhammad and Fatih Külahcı'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
