"""
Physics Base Class
==================

Base class for all QSER PINN physics implementations.
"""

import numpy as np
from abc import ABC, abstractmethod
from QSER.Backends import get_backend


class PhysicsBase(ABC):
    """
    Base class for QSER PINN physics.
    
    Provides coefficient processing and abstract methods for Lc, Ld, L, source.
    
    Attributes:
        v: Advection velocity (scalar, array, or callable)
        D: Diffusion coefficient (scalar, array, or callable)
        lambda_: Decay constant (scalar, array, or callable)
        F: Source term (scalar, array, or callable)
    """
    
    def __init__(self, v=0.1, D=0.05, lambda_=0.01, F=1.0, backend='torch'):
        self.v = v
        self.D = D
        self.lambda_ = lambda_
        self.F = F
        self.backend_name = backend
        self.backend = get_backend(backend)
    
    def _process_coefficient(self, coeff, x, t=None):
        """
        Process coefficient to a value.
        
        Supports:
            - Scalar: Returns the scalar
            - Array: Indexes at x
            - Callable: Evaluates at (x, t)
        """
        if callable(coeff):
            if t is not None:
                return coeff(x, t)
            else:
                return coeff(x)
        elif isinstance(coeff, (np.ndarray, list)):
            # Assume x is the index or coordinate array
            return coeff
        else:
            return coeff
    
    @abstractmethod
    def Lc(self, field, x, t=None):
        """Conservative operator: ∂/∂t + v·∇."""
        pass
    
    @abstractmethod
    def Ld(self, field, x, t=None):
        """Dissipative operator: -D∇² + λ."""
        pass
    
    @abstractmethod
    def L(self, field, x, t=None):
        """Full operator: Lc + Ld."""
        pass
    
    @abstractmethod
    def source(self, x, t=None):
        """Source term F."""
        pass
