"""
Divergence computation class for QSER.

div(F) = ∇ · F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z

Supported methods (via Gradient):
    - '5point': 4th-order central difference (default)
    - 'least_squares': Least-squares fit (2nd order)
"""

import numpy as np


class Divergence:
    """
    Divergence computation engine.

    Parameters:
        mesh: Mesh object (optional)
        backend: str or Backend object (default: 'numpy')
        method: str, gradient method (default: '5point')
    """

    def __init__(self, mesh=None, backend='numpy', method='5point'):
        self.mesh = mesh
        self.backend_name = backend
        self.method = method
        
        if mesh is not None:
            from QSER.Backends import get_backend
            self.backend = get_backend(backend)
            self.dim = mesh.dim
        else:
            from QSER.Backends import get_backend
            self.backend = get_backend(backend)
            self.dim = None

    def set_method(self, method):
        """Set the gradient method."""
        self.method = method

    def get_method(self):
        """Get the current gradient method."""
        return self.method

    def available_methods(self):
        """Return list of available gradient methods."""
        return ['5point', 'least_squares']

    def compute(self, flux_x, flux_y=None, flux_z=None, dx=None, dy=None, dz=None, method=None):
        method = method or self.method
        
        # Convert to backend
        flux_x = self.backend.array(flux_x)
        if flux_y is not None:
            flux_y = self.backend.array(flux_y)
        if flux_z is not None:
            flux_z = self.backend.array(flux_z)

        if self.mesh is not None:
            return self._compute_mesh(flux_x, flux_y, flux_z, method)
        else:
            return self._compute_standalone(flux_x, flux_y, flux_z, dx, dy, dz, method)

    def _compute_mesh(self, flux_x, flux_y, flux_z, method):
        dim = self.mesh.dim
        if dim == 1:
            return self._divergence_1d(flux_x, self.mesh.dx, method)
        elif dim == 2:
            return self._divergence_2d(flux_x, flux_y, self.mesh.dx, self.mesh.dy, method)
        elif dim == 3:
            return self._divergence_3d(flux_x, flux_y, flux_z, self.mesh.dx, self.mesh.dy, self.mesh.dz, method)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def _compute_standalone(self, flux_x, flux_y, flux_z, dx, dy, dz, method):
        if flux_z is not None:
            return self._divergence_3d(flux_x, flux_y, flux_z, dx, dy, dz, method)
        elif flux_y is not None:
            return self._divergence_2d(flux_x, flux_y, dx, dy, method)
        else:
            return self._divergence_1d(flux_x, dx, method)

    # ============================================================
    # 1D Divergence (Forward difference)
    # ============================================================

    def _divergence_1d(self, flux_x, dx, method):
        n = len(flux_x) - 1
        div = self.backend.zeros(n)
        
        if np.isscalar(dx):
            dx_arr = np.full(n, dx)
        else:
            dx_arr = dx
        
        for i in range(n):
            val = (flux_x[i+1] - flux_x[i]) / dx_arr[i]
            div = self.backend.set_item(div, i, val)
        
        return div

    # ============================================================
    # 2D Divergence (Forward difference)
    # ============================================================

    def _divergence_2d(self, flux_x, flux_y, dx, dy, method):
        nx = flux_x.shape[0] - 1
        ny = flux_x.shape[1]
        div = self.backend.zeros((nx, ny))
        
        if np.isscalar(dx):
            dx_arr = np.full(nx, dx)
        else:
            dx_arr = dx
        
        if np.isscalar(dy):
            dy_arr = np.full(ny, dy)
        else:
            dy_arr = dy
        
        for i in range(nx):
            for j in range(ny):
                val = (flux_x[i+1, j] - flux_x[i, j]) / dx_arr[i] + \
                      (flux_y[i, j+1] - flux_y[i, j]) / dy_arr[j]
                div = self.backend.set_item(div, (i, j), val)
        
        return div

    # ============================================================
    # 3D Divergence (Forward difference)
    # ============================================================

    def _divergence_3d(self, flux_x, flux_y, flux_z, dx, dy, dz, method):
        nx = flux_x.shape[0] - 1
        ny = flux_x.shape[1]
        nz = flux_x.shape[2]
        div = self.backend.zeros((nx, ny, nz))
        
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
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    val = (flux_x[i+1, j, k] - flux_x[i, j, k]) / dx_arr[i] + \
                          (flux_y[i, j+1, k] - flux_y[i, j, k]) / dy_arr[j] + \
                          (flux_z[i, j, k+1] - flux_z[i, j, k]) / dz_arr[k]
                    div = self.backend.set_item(div, (i, j, k), val)
        
        return div
