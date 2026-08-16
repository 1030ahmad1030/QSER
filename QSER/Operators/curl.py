"""
Curl computation class for QSER.

Provides:
- Mesh-based Curl (2D scalar, 3D vector)
- Stand-alone Curl (2D scalar, 3D vector)
- Backend-agnostic (NumPy, PyTorch, JAX)
- Autograd support (PyTorch, JAX)

Methods:
    - gradient: Uses Gradient operator (Fluent-style, default)
    - gauss: Uses Gauss theorem (OpenFOAM-style, mesh-based only)
    - 3point: 2nd order finite difference (stand-alone only)
    - 5point: 4th order finite difference (stand-alone only)
    - least_squares: Least squares fit (stand-alone only)
"""

import numpy as np
from .base import Operator
from .gradient import Gradient


class Curl(Operator):
    """
    Curl computation engine.

    Parameters:
        mesh: Mesh object (optional)
        backend: str or Backend object (default: 'numpy')
        method: str, Curl method (default: 'gradient')
    """

    def __init__(self, mesh=None, backend='numpy', method='gradient'):
        super().__init__(mesh, backend, method)
        self._method = method
        if mesh is not None:
            self.backend = mesh.backend
            self.backend_name = mesh.backend_name
        else:
            from QSER.Backends import get_backend
            self.backend = get_backend(backend)
            self.backend_name = backend if isinstance(backend, str) else backend.name
        self._methods = {
            'gradient': self._curl_gradient,
            'gauss': self._curl_gauss,
            '3point': self._curl_3point,
            '5point': self._curl_5point,
            'least_squares': self._curl_least_squares,
        }

    def set_method(self, method):
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self._methods.keys())}")
        self._method = method

    def get_method(self):
        return self._method

    def available_methods(self):
        return list(self._methods.keys())

    def compute(self, Vx, Vy, Vz=None, dx=None, dy=None, dz=None, method=None):
        """
        Compute curl of vector field.

        Parameters:
            Vx: array, x-component of vector field
            Vy: array, y-component of vector field
            Vz: array, z-component of vector field (optional, 3D)
            dx: float or array, spacing in x (stand-alone)
            dy: float or array, spacing in y (stand-alone)
            dz: float or array, spacing in z (stand-alone)
            method: str, curl method override

        Returns:
            curl:
            - 2D: scalar array (curl_z)
            - 3D: tuple (curl_x, curl_y, curl_z)
        """
        method = method or self._method
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self._methods.keys())}")

        # Convert to backend
        Vx = self.to_backend(Vx)
        Vy = self.to_backend(Vy)
        if Vz is not None:
            Vz = self.to_backend(Vz)

        # Route to appropriate method
        if method in ['gradient', 'gauss']:
            # Literature methods
            if self.mesh is not None:
                return self._methods[method](Vx, Vy, Vz)
            else:
                # Gradient works stand-alone, Gauss doesn't
                if method == 'gauss':
                    raise ValueError("Gauss method requires mesh. Use 'gradient' or finite difference methods for stand-alone.")
                return self._methods[method](Vx, Vy, Vz, dx, dy, dz)
        else:
            # Finite difference methods (stand-alone only)
            if self.mesh is not None:
                # Fallback: use gradient method with warning
                import warnings
                warnings.warn(f"Method '{method}' is for stand-alone only. Falling back to 'gradient'.")
                return self._curl_gradient(Vx, Vy, Vz)
            else:
                return self._methods[method](Vx, Vy, Vz, dx, dy, dz)

    # ============================================================
    # METHOD 1: GRADIENT (Fluent-style, default)
    # ============================================================
    def _curl_gradient(self, Vx, Vy, Vz=None, dx=None, dy=None, dz=None):
        """Curl computed using Gradient operator (Fluent-style)."""
        if Vz is None:
            # 2D curl: curl_z = dVy/dx - dVx/dy
            if self.mesh is not None:
                # Mesh-based: use mesh's Gradient (5point)
                grad = Gradient(mesh=self.mesh, backend=self.backend_name)
                grad_Vy = grad.compute(Vy, method='5point')
                grad_Vx = grad.compute(Vx, method='5point')
                return grad_Vy[:, :, 0] - grad_Vx[:, :, 1]
            else:
                # Stand-alone: use 5point gradient directly (NO FALLBACK!)
                grad = Gradient(backend=self.backend_name)
                grad_Vy = grad.compute(Vy, dx=dx, dy=dy, method='5point')
                grad_Vx = grad.compute(Vx, dx=dx, dy=dy, method='5point')
                return grad_Vy[:, :, 0] - grad_Vx[:, :, 1]
        else:
            # 3D curl: (curl_x, curl_y, curl_z)
            if self.mesh is not None:
                grad = Gradient(mesh=self.mesh, backend=self.backend_name)
                grad_Vx = grad.compute(Vx, method='5point')
                grad_Vy = grad.compute(Vy, method='5point')
                grad_Vz = grad.compute(Vz, method='5point')
                
                curl_x = grad_Vz[:, :, :, 1] - grad_Vy[:, :, :, 2]
                curl_y = grad_Vx[:, :, :, 2] - grad_Vz[:, :, :, 0]
                curl_z = grad_Vy[:, :, :, 0] - grad_Vx[:, :, :, 1]
                return curl_x, curl_y, curl_z
            else:
                # Stand-alone 3D: use 5point gradient
                grad = Gradient(backend=self.backend_name)
                grad_Vx = grad.compute(Vx, dx=dx, dy=dy, dz=dz, method='5point')
                grad_Vy = grad.compute(Vy, dx=dx, dy=dy, dz=dz, method='5point')
                grad_Vz = grad.compute(Vz, dx=dx, dy=dy, dz=dz, method='5point')
                
                curl_x = grad_Vz[:, :, :, 1] - grad_Vy[:, :, :, 2]
                curl_y = grad_Vx[:, :, :, 2] - grad_Vz[:, :, :, 0]
                curl_z = grad_Vy[:, :, :, 0] - grad_Vx[:, :, :, 1]
                return curl_x, curl_y, curl_z














    # ============================================================
    # METHOD 2: GAUSS (OpenFOAM-style, mesh-based only)
    # ============================================================

    def _curl_gauss(self, Vx, Vy, Vz=None):
        """Curl computed using Gauss theorem (OpenFOAM-style)."""
        if self.mesh is None:
            raise ValueError("Gauss method requires a mesh.")
        
        if Vz is None:
            return self._curl_gauss_2d(Vx, Vy)
        else:
            return self._curl_gauss_3d(Vx, Vy, Vz)

    def _curl_gauss_2d(self, Vx, Vy):
        """2D Gauss curl."""
        nx = self.mesh.nx
        ny = self.mesh.ny
        # For now, fallback to gradient method with warning
        import warnings
        warnings.warn("Gauss method is not fully implemented. Falling back to gradient.")
        return self._curl_gradient(Vx, Vy)

    def _curl_gauss_3d(self, Vx, Vy, Vz):
        """3D Gauss curl."""
        import warnings
        warnings.warn("Gauss method is not fully implemented. Falling back to gradient.")
        return self._curl_gradient(Vx, Vy, Vz)

    # ============================================================
    # METHOD 3: 3POINT (Stand-alone only)
    # ============================================================

    def _curl_3point(self, Vx, Vy, Vz=None, dx=None, dy=None, dz=None):
        """2nd order finite difference curl (stand-alone only)."""
        if Vz is None:
            return self._curl_3point_2d(Vx, Vy, dx, dy)
        else:
            return self._curl_3point_3d(Vx, Vy, Vz, dx, dy, dz)

    def _curl_3point_2d(self, Vx, Vy, dx, dy):
        """2D 3point curl."""
        nx, ny = Vx.shape
        curl = self.backend.zeros((nx, ny))
        
        if np.isscalar(dx):
            dx_arr = np.full(nx, dx)
        else:
            dx_arr = dx
        
        if np.isscalar(dy):
            dy_arr = np.full(ny, dy)
        else:
            dy_arr = dy
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dVy_dx = (Vy[i+1, j] - Vy[i-1, j]) / (2 * dx_arr[i])
                dVx_dy = (Vx[i, j+1] - Vx[i, j-1]) / (2 * dy_arr[j])
                curl = self.backend.set_item(curl, (i, j), dVy_dx - dVx_dy)
        
        return curl

    def _curl_3point_3d(self, Vx, Vy, Vz, dx, dy, dz):
        """3D 3point curl."""
        nx, ny, nz = Vx.shape
        curl_x = self.backend.zeros((nx, ny, nz))
        curl_y = self.backend.zeros((nx, ny, nz))
        curl_z = self.backend.zeros((nx, ny, nz))
        
        if np.isscalar(dx):
            dx_arr = np.full(nx, dx)
        else:
            dx_arr = dx
        
        if np.isscalar(dy):
            dy_arr = np.full(ny, dy)
        else:
            dy_arr = dy
        
        if np.isscalar(dz):
            dz_arr = np.full(nz, dz)
        else:
            dz_arr = dz
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    dVz_dy = (Vz[i, j+1, k] - Vz[i, j-1, k]) / (2 * dy_arr[j])
                    dVy_dz = (Vy[i, j, k+1] - Vy[i, j, k-1]) / (2 * dz_arr[k])
                    curl_x = self.backend.set_item(curl_x, (i, j, k), dVz_dy - dVy_dz)
                    
                    dVx_dz = (Vx[i, j, k+1] - Vx[i, j, k-1]) / (2 * dz_arr[k])
                    dVz_dx = (Vz[i+1, j, k] - Vz[i-1, j, k]) / (2 * dx_arr[i])
                    curl_y = self.backend.set_item(curl_y, (i, j, k), dVx_dz - dVz_dx)
                    
                    dVy_dx = (Vy[i+1, j, k] - Vy[i-1, j, k]) / (2 * dx_arr[i])
                    dVx_dy = (Vx[i, j+1, k] - Vx[i, j-1, k]) / (2 * dy_arr[j])
                    curl_z = self.backend.set_item(curl_z, (i, j, k), dVy_dx - dVx_dy)
        
        return curl_x, curl_y, curl_z

    # ============================================================
    # METHOD 4: 5POINT (Stand-alone only)
    # ============================================================

    def _curl_5point(self, Vx, Vy, Vz=None, dx=None, dy=None, dz=None):
        """4th order finite difference curl (stand-alone only)."""
        if Vz is None:
            return self._curl_5point_2d(Vx, Vy, dx, dy)
        else:
            return self._curl_5point_3d(Vx, Vy, Vz, dx, dy, dz)

    def _curl_5point_2d(self, Vx, Vy, dx, dy):
        """2D 5point curl."""
        nx, ny = Vx.shape
        curl = self.backend.zeros((nx, ny))
        
        if np.isscalar(dx):
            dx_arr = np.full(nx, dx)
        else:
            dx_arr = dx
        
        if np.isscalar(dy):
            dy_arr = np.full(ny, dy)
        else:
            dy_arr = dy
        
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                dVy_dx = (-Vy[i+2, j] + 8*Vy[i+1, j] - 8*Vy[i-1, j] + Vy[i-2, j]) / (12 * dx_arr[i])
                dVx_dy = (-Vx[i, j+2] + 8*Vx[i, j+1] - 8*Vx[i, j-1] + Vx[i, j-2]) / (12 * dy_arr[j])
                curl = self.backend.set_item(curl, (i, j), dVy_dx - dVx_dy)
        
        return curl

    def _curl_5point_3d(self, Vx, Vy, Vz, dx, dy, dz):
        """3D 5point curl."""
        nx, ny, nz = Vx.shape
        curl_x = self.backend.zeros((nx, ny, nz))
        curl_y = self.backend.zeros((nx, ny, nz))
        curl_z = self.backend.zeros((nx, ny, nz))
        
        if np.isscalar(dx):
            dx_arr = np.full(nx, dx)
        else:
            dx_arr = dx
        
        if np.isscalar(dy):
            dy_arr = np.full(ny, dy)
        else:
            dy_arr = dy
        
        if np.isscalar(dz):
            dz_arr = np.full(nz, dz)
        else:
            dz_arr = dz
        
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                for k in range(2, nz - 2):
                    dVz_dy = (-Vz[i, j+2, k] + 8*Vz[i, j+1, k] - 8*Vz[i, j-1, k] + Vz[i, j-2, k]) / (12 * dy_arr[j])
                    dVy_dz = (-Vy[i, j, k+2] + 8*Vy[i, j, k+1] - 8*Vy[i, j, k-1] + Vy[i, j, k-2]) / (12 * dz_arr[k])
                    curl_x = self.backend.set_item(curl_x, (i, j, k), dVz_dy - dVy_dz)
                    
                    dVx_dz = (-Vx[i, j, k+2] + 8*Vx[i, j, k+1] - 8*Vx[i, j, k-1] + Vx[i, j, k-2]) / (12 * dz_arr[k])
                    dVz_dx = (-Vz[i+2, j, k] + 8*Vz[i+1, j, k] - 8*Vz[i-1, j, k] + Vz[i-2, j, k]) / (12 * dx_arr[i])
                    curl_y = self.backend.set_item(curl_y, (i, j, k), dVx_dz - dVz_dx)
                    
                    dVy_dx = (-Vy[i+2, j, k] + 8*Vy[i+1, j, k] - 8*Vy[i-1, j, k] + Vy[i-2, j, k]) / (12 * dx_arr[i])
                    dVx_dy = (-Vx[i, j+2, k] + 8*Vx[i, j+1, k] - 8*Vx[i, j-1, k] + Vx[i, j-2, k]) / (12 * dy_arr[j])
                    curl_z = self.backend.set_item(curl_z, (i, j, k), dVy_dx - dVx_dy)
        
        return curl_x, curl_y, curl_z

    # ============================================================
    # METHOD 5: LEAST_SQUARES (Stand-alone only)
    # ============================================================

    def _curl_least_squares(self, Vx, Vy, Vz=None, dx=None, dy=None, dz=None):
        """Least squares curl (stand-alone only)."""
        # For now, use 5point as fallback
        import warnings
        warnings.warn("Least squares curl not fully implemented. Falling back to 5point.")
        if Vz is None:
            return self._curl_5point_2d(Vx, Vy, dx, dy)
        else:
            return self._curl_5point_3d(Vx, Vy, Vz, dx, dy, dz)

    # ============================================================
    # DISPATCHERS
    # ============================================================

    def _standard(self, field):
        return field
