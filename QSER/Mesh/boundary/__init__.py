"""
QSER Boundary Module.

Provides boundary condition management:
- Patch-based boundary assignment
- Dirichlet, Neumann, Robin, Periodic, Wall, Source BCs
- Custom user-defined BCs
- Time and space dependent BCs
"""

from .boundary import Boundary

__all__ = ["Boundary"]
