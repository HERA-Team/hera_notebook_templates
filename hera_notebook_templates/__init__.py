# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

from . import utils
from . import utils_h1c

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera_notebook_templates")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
