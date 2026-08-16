"""
QSER: Universal Source-Environment-Response Framework
Version: 1.0.0

Any Method. Any Backend. One Decomposition.

R = S - E
LE = LdS
LcS = F
"""

__version__ = "1.0.0"
__author__ = "Ahmad, QSER Development Team and Deepseek"
__license__ = "MIT"
__github__ = "https://github.com/1030ahmad1030/QSER"

from .QSER import QSER
from .Core import Solver
from .Mesh import Structured1D, Structured2D, Structured3D
from .Physics import Transport
from .Backends import NumPyBackend, TorchBackend
