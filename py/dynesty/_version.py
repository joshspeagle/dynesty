#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single source of truth for the installed package version.

This lives in its own leaf module (importing nothing from the rest of
``dynesty``) so that other modules can read the version without creating
an import cycle through the package ``__init__``.

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dynesty")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
