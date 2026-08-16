"""
QSER Mesh Quality Module.

Provides mesh quality metrics:
- Orthogonality: Face orthogonality
- Skewness: Cell skewness
- Aspect Ratio: Cell aspect ratio
- Cell Volume: Cell volumes
- Quality Report: Full quality assessment
"""

from .quality import MeshQuality

__all__ = ["MeshQuality"]
