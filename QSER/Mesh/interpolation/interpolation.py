"""
Interpolation class for QSER meshes.

Provides a unified interface for all interpolation methods.
The mesh class holds an instance of this class and delegates
all interpolation computations to it.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator, CubicSpline, RectBivariateSpline, RegularGridInterpolator
from QSER.Backends import get_backend


class Interpolation:
    """
    Interpolation class for QSER meshes.

    Parameters:
        mesh: Mesh object (Structured1D, Structured2D, Structured3D)
        method: str, interpolation method (default: 'linear')
        backend: str, 'numpy', 'jax', 'torch' (default: 'numpy')
    """

    def __init__(self, mesh, method='linear', backend='numpy'):
        self.mesh = mesh
        self.dim = mesh.dim
        self.backend = get_backend(backend)
        self._method = method

        self._methods = {
            'linear': self._linear,
            'nearest': self._nearest,
            'cubic': self._cubic,
            'pchip': self._pchip,
            'savgol': self._savgol,
        }

        self._available = self._get_available_methods()

    def _get_available_methods(self):
        # All methods are available for all dimensions
        # For 2D/3D, pchip and savgol fall back to cubic
        return ['linear', 'nearest', 'cubic', 'pchip', 'savgol']

    def set_method(self, method):
        if method not in self._available:
            raise ValueError(
                f"Unknown method: {method}. Available: {self._available}"
            )
        self._method = method

    def get_method(self):
        return self._method

    def available_methods(self):
        return self._available

    def to_faces(self, field, method=None):
        if method is None:
            method = self._method

        if method not in self._available:
            raise ValueError(
                f"Unknown method: {method}. Available: {self._available}"
            )

        if self.backend.name != 'numpy':
            field_np = self.backend.to_numpy(field)
        else:
            field_np = field

        result = self._methods[method](field_np)

        if self.backend.name != 'numpy':
            if isinstance(result, tuple):
                result = tuple(self.backend.array(r) for r in result)
            else:
                result = self.backend.array(result)

        return result

    # ============================================================
    # 1D INTERPOLATION METHODS
    # ============================================================

    def _linear_1d(self, field):
        n = len(field)
        face_field = np.zeros(self.mesh.n_faces)
        face_field[1:-1] = (field[:-1] + field[1:]) / 2
        face_field[0] = field[0]
        face_field[-1] = field[-1]
        return face_field

    def _nearest_1d(self, field):
        face_field = np.zeros(self.mesh.n_faces)
        face_field[0] = field[0]
        face_field[-1] = field[-1]
        for i in range(1, self.mesh.n_faces - 1):
            if (self.mesh.face_centers[i] - self.mesh.cell_centers[i-1]) < \
               (self.mesh.cell_centers[i] - self.mesh.face_centers[i]):
                face_field[i] = field[i-1]
            else:
                face_field[i] = field[i]
        return face_field

    def _cubic_1d(self, field):
        cs = CubicSpline(
            self.mesh.get_cell_centers(return_backend=False),
            field,
            bc_type='natural'
        )
        return cs(self.mesh.get_face_centers(return_backend=False))

    def _pchip_1d(self, field):
        pchip = PchipInterpolator(
            self.mesh.get_cell_centers(return_backend=False),
            field
        )
        return pchip(self.mesh.get_face_centers(return_backend=False))

    def _savgol_1d(self, field):
        n = len(field)
        window_length = min(n // 2 * 2 + 1, 11)
        window_length = max(window_length, 3)
        smoothed = savgol_filter(field, window_length, 3)
        face_field = np.zeros(self.mesh.n_faces)
        face_field[1:-1] = (smoothed[:-1] + smoothed[1:]) / 2
        face_field[0] = smoothed[0]
        face_field[-1] = smoothed[-1]
        return face_field

    # ============================================================
    # 2D INTERPOLATION METHODS
    # ============================================================

    def _get_2d_coords(self):
        """Get 1D coordinate arrays for 2D interpolation."""
        cell_centers = self.mesh.get_cell_centers(return_backend=False)
        x_cell = cell_centers[:, 0, 0]
        y_cell = cell_centers[0, :, 1]

        face_x_centers = self.mesh.get_face_centers_x(return_backend=False)
        x_face = face_x_centers[:, 0, 0]

        face_y_centers = self.mesh.get_face_centers_y(return_backend=False)
        y_face = face_y_centers[0, :, 1]

        return x_cell, y_cell, x_face, y_face

    def _linear_2d(self, field):
        nx, ny = self.mesh.nx, self.mesh.ny
        face_x = np.zeros((nx + 1, ny))
        face_y = np.zeros((nx, ny + 1))

        face_x[1:-1, :] = (field[:-1, :] + field[1:, :]) / 2
        face_x[0, :] = field[0, :]
        face_x[-1, :] = field[-1, :]

        face_y[:, 1:-1] = (field[:, :-1] + field[:, 1:]) / 2
        face_y[:, 0] = field[:, 0]
        face_y[:, -1] = field[:, -1]

        return face_x, face_y

    def _nearest_2d(self, field):
        nx, ny = self.mesh.nx, self.mesh.ny
        face_x = np.zeros((nx + 1, ny))
        face_y = np.zeros((nx, ny + 1))

        for i in range(nx + 1):
            for j in range(ny):
                if i == 0:
                    face_x[i, j] = field[0, j]
                elif i == nx:
                    face_x[i, j] = field[nx-1, j]
                else:
                    if (self.mesh.X_face_x[i, j] - self.mesh.X_cell[i-1, j]) < \
                       (self.mesh.X_cell[i, j] - self.mesh.X_face_x[i, j]):
                        face_x[i, j] = field[i-1, j]
                    else:
                        face_x[i, j] = field[i, j]

        for i in range(nx):
            for j in range(ny + 1):
                if j == 0:
                    face_y[i, j] = field[i, 0]
                elif j == ny:
                    face_y[i, j] = field[i, ny-1]
                else:
                    if (self.mesh.Y_face_y[i, j] - self.mesh.Y_cell[i, j-1]) < \
                       (self.mesh.Y_cell[i, j] - self.mesh.Y_face_y[i, j]):
                        face_y[i, j] = field[i, j-1]
                    else:
                        face_y[i, j] = field[i, j]

        return face_x, face_y

    def _cubic_2d(self, field):
        """Cubic spline interpolation for 2D."""
        x_cell, y_cell, x_face, y_face = self._get_2d_coords()

        spline = RectBivariateSpline(x_cell, y_cell, field, kx=3, ky=3)

        X_face_grid = np.meshgrid(x_face, y_cell, indexing='ij')
        face_x = spline.ev(X_face_grid[0], X_face_grid[1]).reshape(self.mesh.nx + 1, self.mesh.ny)

        Y_face_grid = np.meshgrid(x_cell, y_face, indexing='ij')
        face_y = spline.ev(Y_face_grid[0], Y_face_grid[1]).reshape(self.mesh.nx, self.mesh.ny + 1)

        return face_x, face_y

    def _pchip_2d(self, field):
        """PCHIP for 2D: falls back to cubic (RectBivariateSpline)."""
        return self._cubic_2d(field)

    def _savgol_2d(self, field):
        """Savitzky-Golay for 2D: falls back to cubic (RectBivariateSpline)."""
        return self._cubic_2d(field)

    # ============================================================
    # 3D INTERPOLATION METHODS
    # ============================================================

    def _get_3d_coords(self):
        """Get 1D coordinate arrays for 3D interpolation."""
        cell_centers = self.mesh.get_cell_centers(return_backend=False)
        x_cell = cell_centers[:, 0, 0, 0]
        y_cell = cell_centers[0, :, 0, 1]
        z_cell = cell_centers[0, 0, :, 2]

        face_x_centers = self.mesh.get_face_centers_x(return_backend=False)
        x_face = face_x_centers[:, 0, 0, 0]

        face_y_centers = self.mesh.get_face_centers_y(return_backend=False)
        y_face = face_y_centers[0, :, 0, 1]

        face_z_centers = self.mesh.get_face_centers_z(return_backend=False)
        z_face = face_z_centers[0, 0, :, 2]

        return x_cell, y_cell, z_cell, x_face, y_face, z_face

    def _linear_3d(self, field):
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        face_x = np.zeros((nx + 1, ny, nz))
        face_y = np.zeros((nx, ny + 1, nz))
        face_z = np.zeros((nx, ny, nz + 1))

        face_x[1:-1, :, :] = (field[:-1, :, :] + field[1:, :, :]) / 2
        face_x[0, :, :] = field[0, :, :]
        face_x[-1, :, :] = field[-1, :, :]

        face_y[:, 1:-1, :] = (field[:, :-1, :] + field[:, 1:, :]) / 2
        face_y[:, 0, :] = field[:, 0, :]
        face_y[:, -1, :] = field[:, -1, :]

        face_z[:, :, 1:-1] = (field[:, :, :-1] + field[:, :, 1:]) / 2
        face_z[:, :, 0] = field[:, :, 0]
        face_z[:, :, -1] = field[:, :, -1]

        return face_x, face_y, face_z

    def _nearest_3d(self, field):
        """Nearest neighbor for 3D."""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        face_x = np.zeros((nx + 1, ny, nz))
        face_y = np.zeros((nx, ny + 1, nz))
        face_z = np.zeros((nx, ny, nz + 1))

        # x-faces
        for i in range(nx + 1):
            for j in range(ny):
                for k in range(nz):
                    if i == 0:
                        face_x[i, j, k] = field[0, j, k]
                    elif i == nx:
                        face_x[i, j, k] = field[nx-1, j, k]
                    else:
                        if (self.mesh.X_face_x[i, j, k] - self.mesh.X_cell[i-1, j, k]) < \
                           (self.mesh.X_cell[i, j, k] - self.mesh.X_face_x[i, j, k]):
                            face_x[i, j, k] = field[i-1, j, k]
                        else:
                            face_x[i, j, k] = field[i, j, k]

        # y-faces
        for i in range(nx):
            for j in range(ny + 1):
                for k in range(nz):
                    if j == 0:
                        face_y[i, j, k] = field[i, 0, k]
                    elif j == ny:
                        face_y[i, j, k] = field[i, ny-1, k]
                    else:
                        if (self.mesh.Y_face_y[i, j, k] - self.mesh.Y_cell[i, j-1, k]) < \
                           (self.mesh.Y_cell[i, j, k] - self.mesh.Y_face_y[i, j, k]):
                            face_y[i, j, k] = field[i, j-1, k]
                        else:
                            face_y[i, j, k] = field[i, j, k]

        # z-faces
        for i in range(nx):
            for j in range(ny):
                for k in range(nz + 1):
                    if k == 0:
                        face_z[i, j, k] = field[i, j, 0]
                    elif k == nz:
                        face_z[i, j, k] = field[i, j, nz-1]
                    else:
                        if (self.mesh.Z_face_z[i, j, k] - self.mesh.Z_cell[i, j, k-1]) < \
                           (self.mesh.Z_cell[i, j, k] - self.mesh.Z_face_z[i, j, k]):
                            face_z[i, j, k] = field[i, j, k-1]
                        else:
                            face_z[i, j, k] = field[i, j, k]

        return face_x, face_y, face_z

    def _cubic_3d(self, field):
        """Cubic interpolation for 3D using RegularGridInterpolator."""
        x_cell, y_cell, z_cell, x_face, y_face, z_face = self._get_3d_coords()

        interpolator = RegularGridInterpolator(
            (x_cell, y_cell, z_cell),
            field,
            method='cubic'
        )

        # x-faces: (nx+1, ny, nz)
        X_face_grid = np.meshgrid(x_face, y_cell, z_cell, indexing='ij')
        points_x = np.stack([X_face_grid[0].flatten(), X_face_grid[1].flatten(), X_face_grid[2].flatten()], axis=1)
        face_x = interpolator(points_x).reshape(self.mesh.nx + 1, self.mesh.ny, self.mesh.nz)

        # y-faces: (nx, ny+1, nz)
        Y_face_grid = np.meshgrid(x_cell, y_face, z_cell, indexing='ij')
        points_y = np.stack([Y_face_grid[0].flatten(), Y_face_grid[1].flatten(), Y_face_grid[2].flatten()], axis=1)
        face_y = interpolator(points_y).reshape(self.mesh.nx, self.mesh.ny + 1, self.mesh.nz)

        # z-faces: (nx, ny, nz+1)
        Z_face_grid = np.meshgrid(x_cell, y_cell, z_face, indexing='ij')
        points_z = np.stack([Z_face_grid[0].flatten(), Z_face_grid[1].flatten(), Z_face_grid[2].flatten()], axis=1)
        face_z = interpolator(points_z).reshape(self.mesh.nx, self.mesh.ny, self.mesh.nz + 1)

        return face_x, face_y, face_z

    def _pchip_3d(self, field):
        """PCHIP for 3D: falls back to cubic."""
        return self._cubic_3d(field)

    def _savgol_3d(self, field):
        """Savitzky-Golay for 3D: falls back to cubic."""
        return self._cubic_3d(field)

    # ============================================================
    # MAIN DISPATCHERS
    # ============================================================

    def _linear(self, field):
        if self.dim == 1:
            return self._linear_1d(field)
        elif self.dim == 2:
            return self._linear_2d(field)
        elif self.dim == 3:
            return self._linear_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def _nearest(self, field):
        if self.dim == 1:
            return self._nearest_1d(field)
        elif self.dim == 2:
            return self._nearest_2d(field)
        elif self.dim == 3:
            return self._nearest_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def _cubic(self, field):
        if self.dim == 1:
            return self._cubic_1d(field)
        elif self.dim == 2:
            return self._cubic_2d(field)
        elif self.dim == 3:
            return self._cubic_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def _pchip(self, field):
        if self.dim == 1:
            return self._pchip_1d(field)
        elif self.dim == 2:
            return self._pchip_2d(field)
        elif self.dim == 3:
            return self._pchip_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def _savgol(self, field):
        if self.dim == 1:
            return self._savgol_1d(field)
        elif self.dim == 2:
            return self._savgol_2d(field)
        elif self.dim == 3:
            return self._savgol_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
