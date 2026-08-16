"""
Laplacian Operator
==================

Computes the Laplacian of a scalar field using various methods.

Methods:
    - '5point': 4th-order central difference (default)
    - '3point': 2nd-order central difference
    - 'least_squares': Least-squares fit (fallback to 3point)

Backends:
    - numpy
    - torch
    - jax
"""

import numpy as np
from .base import Operator


class Laplacian(Operator):
    """
    Laplacian operator for scalar fields.
    
    Parameters:
        mesh: Mesh object (optional)
        backend: str or Backend object (default: 'numpy')
        method: str, default Laplacian method (default: '5point')
    
    Usage:
        # Mesh-based
        mesh = Structured1D(nx=100, L=10.0)
        lap = Laplacian(mesh=mesh)
        result = lap.compute(field)
        
        # Stand-alone
        lap = Laplacian(backend='numpy')
        result = lap.compute(field, dx=0.1)
    """
    
    def __init__(self, mesh=None, backend='numpy', method='5point'):
        super().__init__(mesh, backend)
        self.method = method
        self._methods = {
            '5point': self._five_point,
            '3point': self._three_point,
            'least_squares': self._least_squares,
        }
    
    def set_method(self, method):
        """Set the default Laplacian method."""
        if method not in self._methods:
            raise ValueError(f"Method '{method}' not available. Choose from {list(self._methods.keys())}")
        self.method = method
    
    def available_methods(self):
        """Return list of available Laplacian methods."""
        return list(self._methods.keys())
    
    def compute(self, field, dx=None, dy=None, dz=None, method=None):
        """
        Compute Laplacian of a scalar field.
        
        Parameters:
            field: Input scalar field
            dx: Grid spacing in x (for stand-alone)
            dy: Grid spacing in y (for stand-alone)
            dz: Grid spacing in z (for stand-alone)
            method: Laplacian method (default: self.method)
        
        Returns:
            Laplacian field
        """
        method = method or self.method
        
        if method not in self._methods:
            raise ValueError(f"Method '{method}' not available. Choose from {list(self._methods.keys())}")
        
        # Convert field to backend
        field = self.backend.array(field)
        
        # Mesh-based
        if self.mesh is not None:
            return self._methods[method](field)
        else:
            return self._compute_standalone(field, dx, dy, dz, method)
    
    def _compute_standalone(self, field, dx, dy, dz, method):
        """Compute Laplacian in stand-alone mode."""
        # Detect dimension from input
        field_shape = field.shape
        field_ndim = len(field_shape)
        
        # For 1D input
        if field_ndim == 1 or (field_ndim == 2 and field_shape[1] == 1):
            if method == '5point':
                return self._five_point_standalone_1d(field, dx)
            elif method == '3point':
                return self._three_point_standalone_1d(field, dx)
            else:
                return self._three_point_standalone_1d(field, dx)
        
        # For 2D input
        elif field_ndim == 2 or (field_ndim == 3 and field_shape[2] == 1):
            if method == '5point':
                return self._five_point_standalone_2d(field, dx, dy)
            elif method == '3point':
                return self._three_point_standalone_2d(field, dx, dy)
            else:
                return self._three_point_standalone_2d(field, dx, dy)
        
        # For 3D input
        elif field_ndim == 3:
            if method == '5point':
                return self._five_point_standalone_3d(field, dx, dy, dz)
            elif method == '3point':
                return self._three_point_standalone_3d(field, dx, dy, dz)
            else:
                return self._three_point_standalone_3d(field, dx, dy, dz)
        
        else:
            raise ValueError(f"Unsupported field shape: {field_shape}")
    
    def _to_float_array(self, x):
        """Convert input to a float or array of floats safely."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        if hasattr(x, 'shape') and hasattr(x, 'dtype'):
            return np.array(x)
        return x
    
    # ============================================================
    # 1D Methods (FIXED for JAX)
    # ============================================================
    
    def _five_point_standalone_1d(self, field, dx):
        """5-point 4th-order Laplacian for 1D stand-alone."""
        n = len(field)
        lap = self.backend.zeros_like(field)
        
        dx_val = self._to_float_array(dx)
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(n, dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        # FIXED: Ensure dx_arr has same shape as field for broadcasting
        if len(field.shape) == 2 and field.shape[1] == 1:
            dx_arr = dx_arr.reshape(-1, 1)
        
        if n >= 5:
            interior = (-field[4:] + 16*field[3:-1] - 30*field[2:-2] + 16*field[1:-3] - field[:-4]) / (12 * dx_arr[2:-2]**2)
            lap = self.backend.set_slice(lap, 2, -2, interior)
        
        # Boundaries: 3-point
        if n >= 3:
            lap = self.backend.set_item(lap, 0, (field[2] - 2*field[1] + field[0]) / dx_arr[0]**2)
            lap = self.backend.set_item(lap, -1, (field[-1] - 2*field[-2] + field[-3]) / dx_arr[-1]**2)
        
        return lap
    
    def _three_point_standalone_1d(self, field, dx):
        """3-point 2nd-order Laplacian for 1D stand-alone."""
        n = len(field)
        lap = self.backend.zeros_like(field)
        
        dx_val = self._to_float_array(dx)
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(n, dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        # FIXED: Ensure dx_arr has same shape as field for broadcasting
        if len(field.shape) == 2 and field.shape[1] == 1:
            dx_arr = dx_arr.reshape(-1, 1)
        
        if n >= 3:
            interior = (field[2:] - 2*field[1:-1] + field[:-2]) / dx_arr[1:-1]**2
            lap = self.backend.set_slice(lap, 1, -1, interior)
        
        # Boundaries: 2-point
        if n >= 2:
            lap = self.backend.set_item(lap, 0, (field[1] - 2*field[0] + field[0]) / dx_arr[0]**2)
            lap = self.backend.set_item(lap, -1, (field[-1] - 2*field[-1] + field[-2]) / dx_arr[-1]**2)
        
        return lap
    
    # ============================================================
    # 2D Methods (FIXED for JAX)
    # ============================================================
    


    def _five_point_standalone_2d(self, field, dx, dy):
        """5-point 4th-order Laplacian for 2D stand-alone."""
        # Handle dx
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = np.array(dx)
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = np.array(dy)
        
        # Convert to arrays with proper shapes
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
            # Ensure 1D array
            if len(dx_arr.shape) > 1:
                dx_arr = dx_arr.flatten()
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
            # Ensure 1D array
            if len(dy_arr.shape) > 1:
                dy_arr = dy_arr.flatten()
        
        # Reshape for broadcasting
        dx_arr_2d = dx_arr.reshape(-1, 1)   # (nx, 1)
        dy_arr_2d = dy_arr.reshape(1, -1)   # (1, ny)
        
        nx, ny = field.shape
        lap = self.backend.zeros_like(field)
        
        # Interior: 5-point Laplacian
        if nx >= 5 and ny >= 5:
            # x-direction Laplacian (interior only)
            dx_interior = dx_arr_2d[2:-2, :]  # (nx-4, 1)
            lap_x = (-field[4:, 2:-2] + 16*field[3:-1, 2:-2] - 30*field[2:-2, 2:-2] + 16*field[1:-3, 2:-2] - field[:-4, 2:-2]) / (12 * dx_interior**2)
            
            # y-direction Laplacian (interior only)
            dy_interior = dy_arr_2d[:, 2:-2]  # (1, ny-4)
            lap_y = (-field[2:-2, 4:] + 16*field[2:-2, 3:-1] - 30*field[2:-2, 2:-2] + 16*field[2:-2, 1:-3] - field[2:-2, :-4]) / (12 * dy_interior**2)
            
            # Sum them
            lap_interior = lap_x + lap_y
            lap = self.backend.set_item_slice(lap, 2, -2, 2, -2, lap_interior)
        
        # Boundaries: 3-point
        if nx >= 3:
            lap_x_boundary = (field[2, :] - 2*field[1, :] + field[0, :]) / dx_arr[0]**2
            lap = self.backend.set_item(lap, (0, slice(None)), lap_x_boundary)
            lap_x_boundary = (field[-1, :] - 2*field[-2, :] + field[-3, :]) / dx_arr[-1]**2
            lap = self.backend.set_item(lap, (-1, slice(None)), lap_x_boundary)
        
        if ny >= 3:
            lap_y_boundary = (field[:, 2] - 2*field[:, 1] + field[:, 0]) / dy_arr[0]**2
            lap = self.backend.set_item(lap, (slice(None), 0), lap_y_boundary)
            lap_y_boundary = (field[:, -1] - 2*field[:, -2] + field[:, -3]) / dy_arr[-1]**2
            lap = self.backend.set_item(lap, (slice(None), -1), lap_y_boundary)
        
        return lap






    def _three_point_standalone_2d(self, field, dx, dy):
        """3-point 2nd-order Laplacian for 2D stand-alone."""
        # Handle dx
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = np.array(dx)
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = np.array(dy)
        
        # Convert to arrays with proper shapes
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
            if len(dx_arr.shape) > 1:
                dx_arr = dx_arr.flatten()
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
            if len(dy_arr.shape) > 1:
                dy_arr = dy_arr.flatten()
        
        # Reshape for broadcasting
        dx_arr_2d = dx_arr.reshape(-1, 1)
        dy_arr_2d = dy_arr.reshape(1, -1)
        
        nx, ny = field.shape
        lap = self.backend.zeros_like(field)
        
        # Interior: 3-point Laplacian
        if nx >= 3 and ny >= 3:
            dx_interior = dx_arr_2d[1:-1, :]
            lap_x = (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx_interior**2
            
            dy_interior = dy_arr_2d[:, 1:-1]
            lap_y = (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, :-2]) / dy_interior**2
            
            lap_interior = lap_x + lap_y
            lap = self.backend.set_item_slice(lap, 1, -1, 1, -1, lap_interior)
        
        return lap



    def _five_point_standalone_3d(self, field, dx, dy, dz):
        """5-point 4th-order Laplacian for 3D stand-alone."""
        # Handle dx, dy, dz
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = np.array(dx)
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = np.array(dy)
        
        if dz is None:
            dz_val = 1.0
        elif isinstance(dz, (int, float)):
            dz_val = dz
        else:
            dz_val = np.array(dz)
        
        # Convert to arrays with proper shapes
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
            if len(dx_arr.shape) > 1:
                dx_arr = dx_arr.flatten()
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
            if len(dy_arr.shape) > 1:
                dy_arr = dy_arr.flatten()
        
        if isinstance(dz_val, (int, float)):
            dz_arr = np.full(field.shape[2], dz_val)
        else:
            dz_arr = np.array(dz_val)
            if len(dz_arr.shape) > 1:
                dz_arr = dz_arr.flatten()
        
        # Reshape for broadcasting
        dx_arr_3d = dx_arr.reshape(-1, 1, 1)
        dy_arr_3d = dy_arr.reshape(1, -1, 1)
        dz_arr_3d = dz_arr.reshape(1, 1, -1)
        
        nx, ny, nz = field.shape
        lap = self.backend.zeros_like(field)
        
        # Interior: 5-point Laplacian
        if nx >= 5 and ny >= 5 and nz >= 5:
            # x-direction Laplacian (interior only)
            dx_interior = dx_arr_3d[2:-2, :, :]
            lap_x = (-field[4:, 2:-2, 2:-2] + 16*field[3:-1, 2:-2, 2:-2] - 30*field[2:-2, 2:-2, 2:-2] + 16*field[1:-3, 2:-2, 2:-2] - field[:-4, 2:-2, 2:-2]) / (12 * dx_interior**2)
            
            # y-direction Laplacian (interior only)
            dy_interior = dy_arr_3d[:, 2:-2, :]
            lap_y = (-field[2:-2, 4:, 2:-2] + 16*field[2:-2, 3:-1, 2:-2] - 30*field[2:-2, 2:-2, 2:-2] + 16*field[2:-2, 1:-3, 2:-2] - field[2:-2, :-4, 2:-2]) / (12 * dy_interior**2)
            
            # z-direction Laplacian (interior only)
            dz_interior = dz_arr_3d[:, :, 2:-2]
            lap_z = (-field[2:-2, 2:-2, 4:] + 16*field[2:-2, 2:-2, 3:-1] - 30*field[2:-2, 2:-2, 2:-2] + 16*field[2:-2, 2:-2, 1:-3] - field[2:-2, 2:-2, :-4]) / (12 * dz_interior**2)
            
            # Sum all three components (THIS IS THE FIX!)
            lap_interior = lap_x + lap_y + lap_z
            lap = self.backend.set_slice_3d(lap, 2, -2, 2, -2, 2, -2, lap_interior)
        
        # Boundaries: 3-point (fallback)
        lap_3p = self._three_point_standalone_3d(field, dx, dy, dz)
        
        # Use 3-point for boundaries, 5-point for interior
        if nx >= 5 and ny >= 5 and nz >= 5:
            # Only set interior from 5point (boundaries stay from 3point)
            pass  # Already set above
        else:
            # Use 3point for everything if resolution too low
            lap = lap_3p
        
        return lap

    def _three_point_standalone_3d(self, field, dx, dy, dz):
        """3-point 2nd-order Laplacian for 3D stand-alone."""
        # Handle dx, dy, dz
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = np.array(dx)
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = np.array(dy)
        
        if dz is None:
            dz_val = 1.0
        elif isinstance(dz, (int, float)):
            dz_val = dz
        else:
            dz_val = np.array(dz)
        
        # Convert to arrays with proper shapes
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
            if len(dx_arr.shape) > 1:
                dx_arr = dx_arr.flatten()
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
            if len(dy_arr.shape) > 1:
                dy_arr = dy_arr.flatten()
        
        if isinstance(dz_val, (int, float)):
            dz_arr = np.full(field.shape[2], dz_val)
        else:
            dz_arr = np.array(dz_val)
            if len(dz_arr.shape) > 1:
                dz_arr = dz_arr.flatten()
        
        # Reshape for broadcasting
        dx_arr_3d = dx_arr.reshape(-1, 1, 1)
        dy_arr_3d = dy_arr.reshape(1, -1, 1)
        dz_arr_3d = dz_arr.reshape(1, 1, -1)
        
        nx, ny, nz = field.shape
        lap = self.backend.zeros_like(field)
        
        # Interior: 3-point Laplacian
        if nx >= 3 and ny >= 3 and nz >= 3:
            # x-direction
            dx_interior = dx_arr_3d[1:-1, :, :]
            lap_x = (field[2:, 1:-1, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / dx_interior**2
            
            # y-direction
            dy_interior = dy_arr_3d[:, 1:-1, :]
            lap_y = (field[1:-1, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / dy_interior**2
            
            # z-direction
            dz_interior = dz_arr_3d[:, :, 1:-1]
            lap_z = (field[1:-1, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / dz_interior**2
            
            # Sum all three components (THIS IS THE FIX!)
            lap_interior = lap_x + lap_y + lap_z
            lap = self.backend.set_slice_3d(lap, 1, -1, 1, -1, 1, -1, lap_interior)
        
        return lap
    
    # ============================================================
    # Mesh-based methods
    # ============================================================
    
    def _five_point(self, field):
        """5-point Laplacian on mesh."""
        if self.mesh is None:
            raise ValueError("Mesh is required for mesh-based Laplacian")
        return self._compute_standalone(field, self.mesh.dx, None, None, '5point')
    
    def _three_point(self, field):
        """3-point Laplacian on mesh."""
        if self.mesh is None:
            raise ValueError("Mesh is required for mesh-based Laplacian")
        return self._compute_standalone(field, self.mesh.dx, None, None, '3point')
    
    def _least_squares(self, field):
        """Least-squares Laplacian on mesh (fallback to 3point)."""
        if self.mesh is None:
            raise ValueError("Mesh is required for mesh-based Laplacian")
        return self._compute_standalone(field, self.mesh.dx, None, None, '3point')
