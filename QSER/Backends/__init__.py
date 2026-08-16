"""
QSER Backend Module.

Provides backend-agnostic computation engines:
- NumPyBackend: Default CPU backend
- TorchBackend: GPU + Autograd
- JAXBackend: GPU + XLA + Autograd
- OpenFOAMBackend: Production C++ engine
"""

from .Base import Backend
from .NumPyBackend import NumPyBackend
from .TorchBackend import TorchBackend
from .JAXBackend import JAXBackend
from .OpenFOAMBackend import OpenFOAMBackend
from .Factory import get_backend

__all__ = [
    "Backend",
    "NumPyBackend",
    "TorchBackend",
    "JAXBackend",
    "OpenFOAMBackend",
    "get_backend"
]
