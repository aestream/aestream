"""
AEStream library for streaming address-event representations.

Please refer to https://github.com/norse/aestream for usage
"""
import torch
from .aestream_ext import UDPInput, USBInput

__all__ = [
    "UDPInput",
    "USBInput"
]