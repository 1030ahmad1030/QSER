"""
QSER Mesh Module.

Provides structured meshes with gradient method selection:
- Structured1D: 1D structured mesh
- Structured2D: 2D structured mesh
- Structured3D: 3D structured mesh

Gradient Methods (selectable via set_gradient_method()):
- least_squares: Least-Squares Cell-Based (Fluent default)
- green_gauss_node: Green-Gauss Node-Based
- green_gauss_cell: Green-Gauss Cell-Based
- pointLinear: Point-Based Linear Interpolation
- 5point: 5-Point Central Difference (4th order)
- spectral: Spectral Method (future)
"""

from .Structured1D import Structured1D
from .Structured2D import Structured2D
from .Structured3D import Structured3D
from .interpolation import Interpolation
from .quality import MeshQuality
from .boundary import Boundary

__all__ = [
    "Structured1D",
    "Structured2D",
    "Structured3D",
    "Interpolation",
    "MeshQuality",
    "Boundary",
]
