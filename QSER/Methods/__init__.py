"""
QSER Numerical Methods Module.

Supported methods:
- FVM: Finite Volume Method (Version 1.0)
- PINN: Physics-Informed Neural Networks (Version 1.0)
- FEM: Finite Element Method (Future)
- Spectral: Spectral Methods (Future)
"""

from .Base import NumericalMethod
from .FVM import FVM
from .PINN import PINN

__all__ = [
    "NumericalMethod",
    "FVM",
    "PINN"
]
