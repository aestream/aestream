"""
AEStream library for streaming address-event representations.

Please refer to https://github.com/aestream/aestream for usage
"""
import logging
from importlib.metadata import version, PackageNotFoundError

# Import AEStream modules
from aestream.aestream_ext import Backend, Camera, Event, drivers
from aestream._input import FileInput, UDPInput


try:
    __version__ = version("aestream")
except PackageNotFoundError:
    # package is not installed
    pass

modules = []

if "caer" in drivers or "metavision" in drivers:
    from aestream._input import USBInput

    modules.append("USBInput")
else:
    logging.debug("Failed to import AEStream USB Input")

try:
    import aestream._genn as genn

    modules.append("genn")
except ImportError as ex:
    logging.debug("Failed to import GeNN: AEStream cannot use GeNN device")

if "zmq" in drivers:
    from aestream._input import SpeckInput

    modules.append("SpeckInput")
else:
    logging.debug("Failed to import ZMQ: AEStream cannot use ZMQ input")

del logging

__all__ = ["Backend", "Camera", "drivers", "Event", "FileInput", "UDPInput"] + modules
del modules
