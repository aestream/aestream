"""
AEStream library for streaming address-event representations.

Please refer to https://github.com/aestream/aestream for usage
"""
try:
    import torch
except ImportError:
    import logging

    logging.debug("Failed to import Torch: AEStream is running in Numpy mode")
    del logging

# Import AEStream modules
from aestream.aestream_ext import Event
from aestream._input import FileInput, UDPInput

modules = []
try:
    from aestream._input import USBInput

    modules.append("USBInput")
except ImportError:
    import logging

    logging.debug("Failed to import AEStream USB Input")
    del logging

try:
    import aestream._genn as genn
    modules.append("genn")
except ImportError as ex:
    import logging

    logging.debug("Failed to import GeNN: AEStream cannot use GeNN device")
    
__all__ = ["Event", "FileInput", "UDPInput"] + modules
del modules
