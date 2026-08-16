"""
Custom Physics
==============

User-defined physics for QSER PINNs.

Allows users to define their own Lc, Ld, L, and source operators.

Usage:
    from QSER.PINNs.Physics import CustomPhysics
    
    # Define custom operators
    def Lc(field, x, t):
        field_t = torch.autograd.grad(field, t, ...)[0]
        return field_t
    
    def Ld(field, x, t):
        lap = Laplacian()
        return -lap.compute(field, x)
    
    def L(field, x, t):
        return Lc(field, x, t) + Ld(field, x)
    
    def source(x, t):
        return torch.zeros_like(x[..., 0])
    
    physics = CustomPhysics(
        Lc=Lc,
        Ld=Ld,
        L=L,
        source=source,
        backend='torch'
    )
"""

from .base import PhysicsBase


class CustomPhysics(PhysicsBase):
    """
    User-defined physics for QSER PINNs.
    
    Allows users to define their own Lc, Ld, L, and source.
    
    Args:
        Lc: Conservative operator (callable)
        Ld: Dissipative operator (callable)
        L: Full operator (callable)
        source: Source term (callable)
        v: Optional velocity (for consistency)
        D: Optional diffusion (for consistency)
        lambda_: Optional decay (for consistency)
        F: Optional source (for consistency)
        backend: Backend (default: 'torch')
    
    Usage:
        # User defines all operators
        physics = CustomPhysics(
            Lc=my_Lc,
            Ld=my_Ld,
            L=my_L,
            source=my_source
        )
        
        # User defines only L (standard mode)
        physics = CustomPhysics(
            L=my_L,
            source=my_source
        )
    """
    
    def __init__(self, Lc=None, Ld=None, L=None, source=None,
                 v=0.1, D=0.05, lambda_=0.01, F=1.0, backend='torch'):
        super().__init__(v, D, lambda_, F, backend)
        
        # Store user-defined operators
        self._Lc = Lc
        self._Ld = Ld
        self._L = L
        self._source = source
        
        # Validate at least L is provided
        if L is None and (Lc is None or Ld is None):
            raise ValueError(
                "CustomPhysics requires either L, or both Lc and Ld."
            )
    
    def Lc(self, field, x, t=None):
        """
        Conservative operator (user-defined or default).
        """
        if self._Lc is not None:
            if t is not None:
                return self._Lc(field, x, t)
            else:
                return self._Lc(field, x)
        else:
            # Default: time derivative only
            if t is not None:
                import torch
                field_t = torch.autograd.grad(
                    field, t,
                    grad_outputs=torch.ones_like(field),
                    create_graph=True,
                    retain_graph=True
                )[0]
                return field_t
            else:
                return torch.zeros_like(field)
    
    def Ld(self, field, x, t=None):
        """
        Dissipative operator (user-defined or default).
        """
        if self._Ld is not None:
            if t is not None:
                return self._Ld(field, x, t)
            else:
                return self._Ld(field, x)
        else:
            # Default: zero
            return torch.zeros_like(field)
    
    def L(self, field, x, t=None):
        """
        Full operator (user-defined or Lc + Ld).
        """
        if self._L is not None:
            if t is not None:
                return self._L(field, x, t)
            else:
                return self._L(field, x)
        else:
            return self.Lc(field, x, t) + self.Ld(field, x, t)
    
    def source(self, x, t=None):
        """
        Source term (user-defined or default).
        """
        if self._source is not None:
            if t is not None:
                return self._source(x, t)
            else:
                return self._source(x)
        else:
            # Default: zero source
            import torch
            return torch.zeros_like(x[..., 0])
    
    def residual(self, field, x, t=None):
        """
        Compute PDE residual: L(field) - F
        """
        return self.L(field, x, t) - self.source(x, t)
