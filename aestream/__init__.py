"""
AEStream library for streaming address-event representations.

Please refer to https://github.com/aestream/aestream for usage
"""
import logging

try:
    import torch
except ImportError:
    logging.debug("Failed to import Torch: AEStream is running in Numpy mode")

# Import AEStream modules
from aestream.aestream_ext import Event
from aestream._input import FileInput, UDPInput

modules = []
try:
    from aestream._input import USBInput

    modules.append("USBInput")
except ImportError:
    logging.debug("Failed to import AEStream USB Input")

try:
    import aestream._genn as genn

    modules.append("genn")
except ImportError as ex:
    logging.debug("Failed to import GeNN: AEStream cannot use GeNN device")

try:
    from aestream._input import SpeckInput

    modules.append("SpeckInput")
except ImportError:
    logging.debug("Failed to import ZMQ: AEStream cannot use ZMQ input")

del logging

__all__ = ["Event", "FileInput", "UDPInput"] + modules
del modules
