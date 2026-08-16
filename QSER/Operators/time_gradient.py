"""
Time Gradient Operator for QSER
===============================

Computes ∂field/∂t along the time axis.

Supported methods:
    - forward:  1st order forward difference
    - backward: 1st order backward difference
    - central:  2nd order central difference (default)
    - 5point:   4th order central difference
    - least_squares: Least-squares fit of linear trend over window

Supported backends:
    - NumPy: Full support
    - PyTorch: Full support
    - JAX: Full support (functional updates)

Usage:
    from QSER.Operators import TimeGradient
    
    time_grad = TimeGradient(backend='numpy')
    f_dot = time_grad.compute(f, dt=0.01, method='5point')
"""

import numpy as np
from typing import Optional, Union, Tuple
from QSER.Backends import get_backend


class TimeGradient:
    """
    Time derivative operator (∂/∂t).
    
    Computes the first time derivative of a field along the time axis.
    Mirrors the spatial gradient structure but for 1D time only.
    """
    
    def __init__(self, mesh=None, backend: str = 'numpy'):
        """
        Initialize TimeGradient operator.
        
        Args:
            mesh: Mesh object with time axis (optional)
            backend: 'numpy', 'torch', or 'jax'
        """
        self.mesh = mesh
        self.backend = get_backend(backend)
        self.backend_name = backend
        
        self._methods = {
            'forward': self._forward,
            'backward': self._backward,
            'central': self._central,
            '5point': self._five_point,
            'least_squares': self._least_squares,
        }
    
    def _moveaxis(self, field, axis, dest=0):
        """Move axis using backend-agnostic approach."""
        if self.backend_name == 'jax':
            import jax.numpy as jnp
            return jnp.moveaxis(field, axis, dest)
        elif self.backend_name == 'torch':
            import torch
            return torch.moveaxis(field, axis, dest)
        else:
            return np.moveaxis(field, axis, dest)
    
    def compute(
        self,
        field: np.ndarray,
        dt: Optional[float] = None,
        method: str = 'central',
        axis: int = 0
    ) -> np.ndarray:
        """
        Compute ∂field/∂t along the time axis.
        
        Args:
            field: Input field (time, ...) or (..., time, ...)
            dt: Time step between samples (optional if mesh has dt)
            method: 'forward', 'backward', 'central', '5point', 'least_squares'
            axis: Axis along which time varies (default: 0)
            
        Returns:
            ∂field/∂t with same shape as input
        """
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self._methods.keys())}")
        
        # Get dt from mesh if available
        if dt is None:
            if self.mesh is not None and hasattr(self.mesh, 'dt'):
                dt = self.mesh.dt
            elif self.mesh is not None and hasattr(self.mesh, 'time_step'):
                dt = self.mesh.time_step
            else:
                raise ValueError("dt must be provided or mesh must have dt attribute")
        
        # Move time axis to front for uniform handling (backend-agnostic)
        if axis != 0:
            field = self._moveaxis(field, axis, 0)
            moved = True
        else:
            moved = False
        
        # Compute gradient
        result = self._methods[method](field, dt)
        
        # Move axis back if needed (backend-agnostic)
        if moved:
            result = self._moveaxis(result, 0, axis)
        
        return result
    
    # ============================================================
    # Helper: set item with backend support (functional for JAX)
    # ============================================================
    
    def _set_item(self, arr, idx, value):
        """Set item with backend support (functional for JAX)."""
        if self.backend_name == 'jax':
            # JAX uses functional updates
            return arr.at[idx].set(value)
        else:
            # NumPy/PyTorch use in-place
            arr[idx] = value
            return arr
    
    def _set_slice(self, arr, start, end, value):
        """Set slice with backend support (functional for JAX)."""
        if self.backend_name == 'jax':
            # JAX uses functional updates
            return arr.at[start:end].set(value)
        else:
            # NumPy/PyTorch use in-place
            arr[start:end] = value
            return arr
    
    # ============================================================
    # Method implementations (1D only)
    # ============================================================
    
    def _forward(self, field: np.ndarray, dt: float) -> np.ndarray:
        """1st order forward difference."""
        n = field.shape[0]
        grad = self.backend.zeros_like(field)
        
        # Interior: use 3-point central where possible
        if n >= 3:
            for i in range(1, n - 1):
                val = (field[i+1] - field[i-1]) / (2 * dt)
                grad = self._set_item(grad, i, val)
        else:
            for i in range(n - 1):
                val = (field[i+1] - field[i]) / dt
                grad = self._set_item(grad, i, val)
        
        # Left boundary: forward
        if n > 1:
            grad = self._set_item(grad, 0, (field[1] - field[0]) / dt)
        
        # Right boundary: backward if possible
        if n > 1:
            grad = self._set_item(grad, -1, (field[-1] - field[-2]) / dt)
        
        return grad
    
    def _backward(self, field: np.ndarray, dt: float) -> np.ndarray:
        """1st order backward difference."""
        n = field.shape[0]
        grad = self.backend.zeros_like(field)
        
        # Interior: use 3-point central where possible
        if n >= 3:
            for i in range(1, n - 1):
                val = (field[i+1] - field[i-1]) / (2 * dt)
                grad = self._set_item(grad, i, val)
        else:
            for i in range(1, n):
                val = (field[i] - field[i-1]) / dt
                grad = self._set_item(grad, i, val)
        
        # Right boundary: backward
        if n > 1:
            grad = self._set_item(grad, -1, (field[-1] - field[-2]) / dt)
        
        # Left boundary: forward if possible
        if n > 1:
            grad = self._set_item(grad, 0, (field[1] - field[0]) / dt)
        
        return grad
    
    def _central(self, field: np.ndarray, dt: float) -> np.ndarray:
        """2nd order central difference."""
        n = field.shape[0]
        grad = self.backend.zeros_like(field)
        
        # Interior: 2nd order central
        if n >= 3:
            for i in range(1, n - 1):
                val = (field[i+1] - field[i-1]) / (2 * dt)
                grad = self._set_item(grad, i, val)
        else:
            for i in range(n - 1):
                val = (field[i+1] - field[i]) / dt
                grad = self._set_item(grad, i, val)
        
        # Boundaries: 1st order
        if n > 1:
            grad = self._set_item(grad, 0, (field[1] - field[0]) / dt)
            grad = self._set_item(grad, -1, (field[-1] - field[-2]) / dt)
        
        return grad
    
    def _five_point(self, field: np.ndarray, dt: float) -> np.ndarray:
        """4th order central difference (5-point stencil)."""
        n = field.shape[0]
        grad = self.backend.zeros_like(field)
        
        # Interior: 4th order central
        if n >= 5:
            for i in range(2, n - 2):
                val = (-field[i+2] + 8*field[i+1] - 8*field[i-1] + field[i-2]) / (12 * dt)
                grad = self._set_item(grad, i, val)
        else:
            # Fallback to central if not enough points
            if n >= 3:
                for i in range(1, n - 1):
                    val = (field[i+1] - field[i-1]) / (2 * dt)
                    grad = self._set_item(grad, i, val)
            else:
                for i in range(n - 1):
                    val = (field[i+1] - field[i]) / dt
                    grad = self._set_item(grad, i, val)
        
        # Boundaries: use lower order
        if n >= 2:
            grad = self._set_item(grad, 0, (field[1] - field[0]) / dt)
            grad = self._set_item(grad, -1, (field[-1] - field[-2]) / dt)
        if n >= 3:
            grad = self._set_item(grad, 1, (field[2] - field[0]) / (2 * dt))
            grad = self._set_item(grad, -2, (field[-1] - field[-3]) / (2 * dt))
        
        return grad
    
    def _least_squares(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Least-squares linear fit derivative.
        
        For each point, fits a line to neighboring points and takes the slope.
        Uses 3-point window for interior, 2-point for boundaries.
        """
        n = field.shape[0]
        grad = self.backend.zeros_like(field)
        
        # Interior: least-squares linear fit over 3 points
        if n >= 3:
            for i in range(1, n - 1):
                val = (field[i+1] - field[i-1]) / (2 * dt)
                grad = self._set_item(grad, i, val)
        else:
            for i in range(n - 1):
                val = (field[i+1] - field[i]) / dt
                grad = self._set_item(grad, i, val)
        
        # Boundaries: 1st order
        if n > 1:
            grad = self._set_item(grad, 0, (field[1] - field[0]) / dt)
            grad = self._set_item(grad, -1, (field[-1] - field[-2]) / dt)
        
        return grad
    
    def available_methods(self) -> list:
        """Return list of available methods."""
        return list(self._methods.keys())
