"""Module for interface Fanpy with external quantum chemistry packages."""

__all__ = [
    "ProjectedSchrodingerFanCI",
    "ProjectedSchrodingerPyCI",
]

from .legacy import ProjectedSchrodingerFanCI
from .pyci import ProjectedSchrodingerPyCI
