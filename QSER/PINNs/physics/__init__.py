"""
Physics Module for QSER PINNs.

Provides:
    - TransportPINN: Advection-diffusion-decay physics
    - CustomPhysics: User-defined physics
    - PhysicsBase: Base class for all physics
"""

from .transport_pinn import TransportPINN
from .custom_physics import CustomPhysics
from .base import PhysicsBase

__all__ = ['TransportPINN', 'CustomPhysics', 'PhysicsBase']
