"""
Gradient Operator
=================

Computes the gradient of a scalar field using various methods.

Methods:
    - '5point': 4th-order central difference (default)
    - 'least_squares': Cell-based least squares (2nd order)
    - 'green_gauss_node': Green-Gauss node-based (future)
    - 'green_gauss_cell': Green-Gauss cell-based (future)
    - 'pointLinear': Point linear (future)
    - 'spectral': Spectral (future)

Backends:
    - numpy
    - torch
    - jax
"""

import numpy as np
from .base import Operator
from QSER.Backends import get_backend


class Gradient(Operator):
    """
    Gradient operator for scalar fields.
    
    Parameters:
        mesh: Mesh object (optional)
        backend: str or Backend object (default: 'numpy')
        method: str, default gradient method (default: '5point')
    
    Usage:
        # Mesh-based
        mesh = Structured1D(nx=100, L=10.0)
        grad = Gradient(mesh=mesh)
        result = grad.compute(field)
        
        # Stand-alone
        grad = Gradient(backend='numpy')
        result = grad.compute(field, dx=0.1)
    """
    
    def __init__(self, mesh=None, backend='numpy', method='5point'):
        super().__init__(mesh, backend)
        self.method = method
        self._methods = {
            '5point': self._five_point,
            'least_squares': self._least_squares,
            'green_gauss_node': self._green_gauss_node,
            'green_gauss_cell': self._green_gauss_cell,
            'pointLinear': self._point_linear,
            'spectral': self._spectral,
        }
    
    def set_method(self, method):
        """Set the default gradient method."""
        if method not in self._methods:
            raise ValueError(f"Method '{method}' not available. Choose from {list(self._methods.keys())}")
        self.method = method
    
    def available_methods(self):
        """Return list of available gradient methods."""
        return list(self._methods.keys())
    
    def compute(self, field, dx=None, dy=None, dz=None, points=None, method=None, axis=-1):
        """
        Compute gradient of a scalar field.
        
        Parameters:
            field: Input scalar field
            dx: Grid spacing in x (for stand-alone)
            dy: Grid spacing in y (for stand-alone)
            dz: Grid spacing in z (for stand-alone)
            points: Point coordinates (for mesh-based)
            method: Gradient method (default: self.method)
            axis: Axis along which to compute (for 1D stand-alone)
        
        Returns:
            Gradient field
        """
        method = method or self.method
        
        if method not in self._methods:
            raise ValueError(f"Method '{method}' not available. Choose from {list(self._methods.keys())}")
        
        # Mesh-based
        if self.mesh is not None:
            return self._methods[method](field)
        else:
            return self._compute_standalone(field, dx, dy, dz, points, method, axis)
    
    def _compute_standalone(self, field, dx, dy, dz, points, method, axis):
        """Compute gradient in stand-alone mode."""
        # Convert field to backend
        field = self.backend.array(field)
        
        # Detect dimension from input
        field_shape = field.shape
        field_ndim = len(field_shape)
        
        # For 1D input
        if field_ndim == 1 or (field_ndim == 2 and field_shape[1] == 1):
            # 1D (or 2D with second dimension 1)
            if method == '5point':
                return self._five_point_standalone_1d(field, dx, axis)
            elif method == 'least_squares':
                return self._least_squares_standalone_1d(field, dx, axis)
            else:
                return self._five_point_standalone_1d(field, dx, axis)
        
        # For 2D input
        elif field_ndim == 2 or (field_ndim == 3 and field_shape[2] == 1):
            # 2D
            if method == '5point':
                return self._five_point_standalone_2d(field, dx, dy)
            elif method == 'least_squares':
                return self._least_squares_standalone_2d(field, dx, dy)
            else:
                return self._five_point_standalone_2d(field, dx, dy)
        
        # For 3D input
        elif field_ndim == 3:
            if method == '5point':
                return self._five_point_standalone_3d(field, dx, dy, dz)
            elif method == 'least_squares':
                return self._least_squares_standalone_3d(field, dx, dy, dz)
            else:
                return self._five_point_standalone_3d(field, dx, dy, dz)
        
        else:
            raise ValueError(f"Unsupported field shape: {field_shape}")
    
    def _to_float_array(self, x):
        """Convert input to a float or array of floats safely."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        # If it's a backend tensor, convert to numpy safely
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        if hasattr(x, 'shape') and hasattr(x, 'dtype'):
            return np.array(x)
        return x
    
    def _five_point_standalone_1d(self, field, dx, axis):
        """5-point 4th-order gradient for 1D stand-alone."""
        n = len(field)
        grad = self.backend.zeros_like(field)
        
        # Convert dx to float or array
        dx_val = self._to_float_array(dx)
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(n, dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        # FIXED: Ensure dx_arr has same shape as field for broadcasting
        if len(field.shape) == 2 and field.shape[1] == 1:
            dx_arr = dx_arr.reshape(-1, 1)
        
        if n >= 5:
            # Interior: 5-point formula
            interior_grad = (-field[4:] + 8*field[3:-1] - 8*field[1:-3] + field[:-4]) / (12 * dx_arr[2:-2])
            grad = self.backend.set_slice(grad, 2, -2, interior_grad)
        
        # Boundaries: 2-point (1st order)
        grad = self.backend.set_item(grad, 0, (field[1] - field[0]) / dx_arr[0])
        if n > 1:
            grad = self.backend.set_item(grad, -1, (field[-1] - field[-2]) / dx_arr[-1])
        if n > 2:
            grad = self.backend.set_item(grad, 1, (field[2] - field[0]) / (2 * dx_arr[1]))
        if n > 3:
            grad = self.backend.set_item(grad, -2, (field[-1] - field[-3]) / (2 * dx_arr[-2]))
        
        return grad
    
    def _least_squares_standalone_1d(self, field, dx, axis):
        """Least-squares gradient for 1D stand-alone."""
        n = len(field)
        grad = self.backend.zeros_like(field)
        
        dx_val = self._to_float_array(dx)
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(n, dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        # FIXED: Ensure dx_arr has same shape as field for broadcasting
        if len(field.shape) == 2 and field.shape[1] == 1:
            dx_arr = dx_arr.reshape(-1, 1)
        
        if n >= 3:
            for i in range(1, n-1):
                x_vals = np.array([-dx_arr[i-1], 0, dx_arr[i+1]])
                u_vals = np.array([field[i-1], field[i], field[i+1]])
                x_mean = np.mean(x_vals)
                u_mean = np.mean(u_vals)
                grad_i = np.sum((x_vals - x_mean) * (u_vals - u_mean)) / np.sum((x_vals - x_mean)**2)
                grad = self.backend.set_item(grad, i, grad_i)
        
        grad = self.backend.set_item(grad, 0, (field[1] - field[0]) / dx_arr[0])
        if n > 1:
            grad = self.backend.set_item(grad, -1, (field[-1] - field[-2]) / dx_arr[-1])
        
        return grad
    
    def _five_point_standalone_2d(self, field, dx, dy):
        """5-point 4th-order gradient for 2D stand-alone (FIXED for JAX)."""
        # Handle dx
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = dx
        
        # Handle dy
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = dy
        
        # Convert to arrays if scalar
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
        
        nx, ny = field.shape
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)
        
        # Interior x-gradient: 5-point (FIXED: uses set_item_slice)
        if nx >= 5:
            if len(dx_arr.shape) == 2:
                dx_interior = dx_arr[2:-2, :]
            else:
                dx_interior = dx_arr[2:-2, None]
            interior_x = (-field[4:, :] + 8*field[3:-1, :] - 8*field[1:-3, :] + field[:-4, :]) / (12 * dx_interior)
            grad_x = self.backend.set_item_slice(grad_x, 2, -2, 0, ny, interior_x)
        
        # Interior y-gradient: 5-point (FIXED: uses set_item_slice)
        if ny >= 5:
            if len(dy_arr.shape) == 2:
                dy_interior = dy_arr[:, 2:-2]
            else:
                dy_interior = dy_arr[None, 2:-2]
            interior_y = (-field[:, 4:] + 8*field[:, 3:-1] - 8*field[:, 1:-3] + field[:, :-4]) / (12 * dy_interior)
            grad_y = self.backend.set_item_slice(grad_y, 0, nx, 2, -2, interior_y)
        
        # Boundaries: 2-point
        if nx > 1:
            if len(dx_arr.shape) == 2:
                grad_x = self.backend.set_item(grad_x, (0, slice(None)), (field[1, :] - field[0, :]) / dx_arr[0, :])
                grad_x = self.backend.set_item(grad_x, (-1, slice(None)), (field[-1, :] - field[-2, :]) / dx_arr[-1, :])
            else:
                grad_x = self.backend.set_item(grad_x, (0, slice(None)), (field[1, :] - field[0, :]) / dx_arr[0])
                grad_x = self.backend.set_item(grad_x, (-1, slice(None)), (field[-1, :] - field[-2, :]) / dx_arr[-1])
        
        if ny > 1:
            if len(dy_arr.shape) == 2:
                grad_y = self.backend.set_item(grad_y, (slice(None), 0), (field[:, 1] - field[:, 0]) / dy_arr[:, 0])
                grad_y = self.backend.set_item(grad_y, (slice(None), -1), (field[:, -1] - field[:, -2]) / dy_arr[:, -1])
            else:
                grad_y = self.backend.set_item(grad_y, (slice(None), 0), (field[:, 1] - field[:, 0]) / dy_arr[0])
                grad_y = self.backend.set_item(grad_y, (slice(None), -1), (field[:, -1] - field[:, -2]) / dy_arr[-1])
        
        # Stack to create (nx, ny, 2) output
        result = self.backend.zeros((nx, ny, 2))
        result = self.backend.set_item(result, (slice(None), slice(None), 0), grad_x)
        result = self.backend.set_item(result, (slice(None), slice(None), 1), grad_y)
        
        return result
    

    def _least_squares_standalone_2d(self, field, dx, dy):
        """
        Least-squares gradient for 2D stand-alone.
        
        Uses a 4-point stencil (left, right, bottom, top) to compute
        the least-squares fit for the gradient components.
        """
        # Convert field to backend array
        field = self.backend.array(field)
        nx, ny = field.shape
        
        # Handle dx and dy - ensure they are scalars for uniform grids
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float, np.float64, np.float32)):
            dx_val = float(dx)
        else:
            # If dx is an array, take the first element (assume uniform)
            dx_val = float(dx.flat[0]) if dx.size > 0 else 1.0
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float, np.float64, np.float32)):
            dy_val = float(dy)
        else:
            # If dy is an array, take the first element (assume uniform)
            dy_val = float(dy.flat[0]) if dy.size > 0 else 1.0
        
        # Create gradient arrays
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)
        
        # Interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Stencil: right, left, top, bottom (4 neighbors)
                dx_vals = np.array([dx_val, -dx_val, 0.0, 0.0])
                dy_vals = np.array([0.0, 0.0, dy_val, -dy_val])
                
                # Field differences at neighbors
                du_vals = np.array([
                    field[i+1, j] - field[i, j],  # right
                    field[i-1, j] - field[i, j],  # left
                    field[i, j+1] - field[i, j],  # top
                    field[i, j-1] - field[i, j]   # bottom
                ])
                
                # Build least-squares matrix: A * [grad_x, grad_y]^T = du
                A = np.column_stack([dx_vals, dy_vals])
                
                # Solve least-squares: g = (A^T A)^{-1} A^T du
                g = np.linalg.lstsq(A, du_vals, rcond=None)[0]
                
                grad_x = self.backend.set_item(grad_x, (i, j), g[0])
                grad_y = self.backend.set_item(grad_y, (i, j), g[1])
        
        # Boundaries: 2-point (1st order)
        if nx > 1:
            grad_x = self.backend.set_item(grad_x, (0, slice(None)), 
                                           (field[1, :] - field[0, :]) / dx_val)
            grad_x = self.backend.set_item(grad_x, (-1, slice(None)), 
                                           (field[-1, :] - field[-2, :]) / dx_val)
        
        if ny > 1:
            grad_y = self.backend.set_item(grad_y, (slice(None), 0), 
                                           (field[:, 1] - field[:, 0]) / dy_val)
            grad_y = self.backend.set_item(grad_y, (slice(None), -1), 
                                           (field[:, -1] - field[:, -2]) / dy_val)
        
        # Stack components into (nx, ny, 2) array
        result = self.backend.zeros((nx, ny, 2))
        result = self.backend.set_item(result, (slice(None), slice(None), 0), grad_x)
        result = self.backend.set_item(result, (slice(None), slice(None), 1), grad_y)
        
        return result







    def _five_point_standalone_3d(self, field, dx, dy, dz):
        """5-point 4th-order gradient for 3D stand-alone."""
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
        
        nx, ny, nz = field.shape
        
        # Reshape for broadcasting (3D)
        dx_arr_3d = dx_arr.reshape(-1, 1, 1)   # (nx, 1, 1)
        dy_arr_3d = dy_arr.reshape(1, -1, 1)   # (1, ny, 1)
        dz_arr_3d = dz_arr.reshape(1, 1, -1)   # (1, 1, nz)
        
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)
        grad_z = self.backend.zeros_like(field)
        
        # Interior x-gradient: 5-point
        if nx >= 5:
            dx_interior = dx_arr_3d[2:-2, :, :]  # (nx-4, 1, 1)
            interior_x = (-field[4:, :, :] + 8*field[3:-1, :, :] - 8*field[1:-3, :, :] + field[:-4, :, :]) / (12 * dx_interior)
            grad_x = self.backend.set_slice_3d(grad_x, 2, -2, 0, ny, 0, nz, interior_x)
        
        # Interior y-gradient: 5-point
        if ny >= 5:
            dy_interior = dy_arr_3d[:, 2:-2, :]  # (1, ny-4, 1)
            interior_y = (-field[:, 4:, :] + 8*field[:, 3:-1, :] - 8*field[:, 1:-3, :] + field[:, :-4, :]) / (12 * dy_interior)
            grad_y = self.backend.set_slice_3d(grad_y, 0, nx, 2, -2, 0, nz, interior_y)
        
        # Interior z-gradient: 5-point
        if nz >= 5:
            dz_interior = dz_arr_3d[:, :, 2:-2]  # (1, 1, nz-4)
            interior_z = (-field[:, :, 4:] + 8*field[:, :, 3:-1] - 8*field[:, :, 1:-3] + field[:, :, :-4]) / (12 * dz_interior)
            grad_z = self.backend.set_slice_3d(grad_z, 0, nx, 0, ny, 2, -2, interior_z)
        
        # Boundaries: 2-point (1st order)
        if nx > 1:
            grad_x = self.backend.set_item(grad_x, (0, slice(None), slice(None)), (field[1, :, :] - field[0, :, :]) / dx_arr[0])
            grad_x = self.backend.set_item(grad_x, (-1, slice(None), slice(None)), (field[-1, :, :] - field[-2, :, :]) / dx_arr[-1])
        
        if ny > 1:
            grad_y = self.backend.set_item(grad_y, (slice(None), 0, slice(None)), (field[:, 1, :] - field[:, 0, :]) / dy_arr[0])
            grad_y = self.backend.set_item(grad_y, (slice(None), -1, slice(None)), (field[:, -1, :] - field[:, -2, :]) / dy_arr[-1])
        
        if nz > 1:
            grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), 0), (field[:, :, 1] - field[:, :, 0]) / dz_arr[0])
            grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), -1), (field[:, :, -1] - field[:, :, -2]) / dz_arr[-1])
        
        # Stack to create (nx, ny, nz, 3) output
        result = self.backend.zeros((nx, ny, nz, 3))
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 0), grad_x)
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 1), grad_y)
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 2), grad_z)
        
        return result















    
    def _least_squares_standalone_3d(self, field, dx, dy, dz):
        """Least-squares gradient for 3D stand-alone."""
        # Handle dx
        if dx is None:
            dx_val = 1.0
        elif isinstance(dx, (int, float)):
            dx_val = dx
        else:
            dx_val = dx
        
        if dy is None:
            dy_val = 1.0
        elif isinstance(dy, (int, float)):
            dy_val = dy
        else:
            dy_val = dy
        
        if dz is None:
            dz_val = 1.0
        elif isinstance(dz, (int, float)):
            dz_val = dz
        else:
            dz_val = dz
        
        if isinstance(dx_val, (int, float)):
            dx_arr = np.full(field.shape[0], dx_val)
        else:
            dx_arr = np.array(dx_val)
        
        if isinstance(dy_val, (int, float)):
            dy_arr = np.full(field.shape[1], dy_val)
        else:
            dy_arr = np.array(dy_val)
        
        if isinstance(dz_val, (int, float)):
            dz_arr = np.full(field.shape[2], dz_val)
        else:
            dz_arr = np.array(dz_val)
        
        nx, ny, nz = field.shape
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)
        grad_z = self.backend.zeros_like(field)
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    if len(dx_arr.shape) == 3:
                        dx_local = dx_arr[i, j, k]
                    else:
                        dx_local = dx_arr[i]
                    
                    if len(dy_arr.shape) == 3:
                        dy_local = dy_arr[i, j, k]
                    else:
                        dy_local = dy_arr[j]
                    
                    if len(dz_arr.shape) == 3:
                        dz_local = dz_arr[i, j, k]
                    else:
                        dz_local = dz_arr[k]
                    
                    dx_vals = np.array([-dx_local, dx_local, 0, 0, 0, 0])
                    dy_vals = np.array([0, 0, -dy_local, dy_local, 0, 0])
                    dz_vals = np.array([0, 0, 0, 0, -dz_local, dz_local])
                    du_vals = np.array([
                        field[i+1, j, k] - field[i, j, k],
                        field[i-1, j, k] - field[i, j, k],
                        field[i, j+1, k] - field[i, j, k],
                        field[i, j-1, k] - field[i, j, k],
                        field[i, j, k+1] - field[i, j, k],
                        field[i, j, k-1] - field[i, j, k]
                    ])
                    A = np.column_stack([dx_vals, dy_vals, dz_vals])
                    g = np.linalg.lstsq(A, du_vals, rcond=None)[0]
                    grad_x = self.backend.set_item(grad_x, (i, j, k), g[0])
                    grad_y = self.backend.set_item(grad_y, (i, j, k), g[1])
                    grad_z = self.backend.set_item(grad_z, (i, j, k), g[2])
        
        if nx > 1:
            if len(dx_arr.shape) == 3:
                grad_x = self.backend.set_item(grad_x, (0, slice(None), slice(None)), (field[1, :, :] - field[0, :, :]) / dx_arr[0, :, :])
                grad_x = self.backend.set_item(grad_x, (-1, slice(None), slice(None)), (field[-1, :, :] - field[-2, :, :]) / dx_arr[-1, :, :])
            else:
                grad_x = self.backend.set_item(grad_x, (0, slice(None), slice(None)), (field[1, :, :] - field[0, :, :]) / dx_arr[0])
                grad_x = self.backend.set_item(grad_x, (-1, slice(None), slice(None)), (field[-1, :, :] - field[-2, :, :]) / dx_arr[-1])
        
        if ny > 1:
            if len(dy_arr.shape) == 3:
                grad_y = self.backend.set_item(grad_y, (slice(None), 0, slice(None)), (field[:, 1, :] - field[:, 0, :]) / dy_arr[:, 0, :])
                grad_y = self.backend.set_item(grad_y, (slice(None), -1, slice(None)), (field[:, -1, :] - field[:, -2, :]) / dy_arr[:, -1, :])
            else:
                grad_y = self.backend.set_item(grad_y, (slice(None), 0, slice(None)), (field[:, 1, :] - field[:, 0, :]) / dy_arr[0])
                grad_y = self.backend.set_item(grad_y, (slice(None), -1, slice(None)), (field[:, -1, :] - field[:, -2, :]) / dy_arr[-1])
        
        if nz > 1:
            if len(dz_arr.shape) == 3:
                grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), 0), (field[:, :, 1] - field[:, :, 0]) / dz_arr[:, :, 0])
                grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), -1), (field[:, :, -1] - field[:, :, -2]) / dz_arr[:, :, -1])
            else:
                grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), 0), (field[:, :, 1] - field[:, :, 0]) / dz_arr[0])
                grad_z = self.backend.set_item(grad_z, (slice(None), slice(None), -1), (field[:, :, -1] - field[:, :, -2]) / dz_arr[-1])
        
        result = self.backend.zeros((nx, ny, nz, 3))
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 0), grad_x)
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 1), grad_y)
        result = self.backend.set_item(result, (slice(None), slice(None), slice(None), 2), grad_z)
        
        return result
    
    # ============================================================
    # Mesh-based methods
    # ============================================================
    
    def _five_point(self, field):
        """5-point gradient on mesh."""
        if self.mesh is None:
            raise ValueError("Mesh is required for mesh-based gradient")
        return self._compute_standalone(field, self.mesh.dx, None, None, None, '5point', -1)
    
    def _least_squares(self, field):
        """Least-squares gradient on mesh."""
        if self.mesh is None:
            raise ValueError("Mesh is required for mesh-based gradient")
        return self._compute_standalone(field, self.mesh.dx, None, None, None, 'least_squares', -1)
    
    def _green_gauss_node(self, field):
        raise NotImplementedError("Green-Gauss node-based gradient not yet implemented")
    
    def _green_gauss_cell(self, field):
        raise NotImplementedError("Green-Gauss cell-based gradient not yet implemented")
    
    def _point_linear(self, field):
        raise NotImplementedError("Point linear gradient not yet implemented")
    
    def _spectral(self, field):
        raise NotImplementedError("Spectral gradient not yet implemented")
