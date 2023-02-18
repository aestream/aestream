"""
This is printed from aestream/__init__.py
AEStream library for streaming address-event representations.

Please refer to https://github.com/norse/aestream for usage
"""
try:
    import torch
except ImportError:
    import logging

    logging.debug("Failed to import Torch: AEStream is running in Numpy mode")
    del logging

from aestream._input import FileInput, UDPInput

modules = []
try:
    from aestream_ext import USBInput

    modules.append("USBInput")
except ImportError:
    import logging

    logging.debug("Failed to import AEStream USB Input")
    del logging

__all__ = ["FileInput", "UDPInput"] + modules

del modules
