import numpy as np
import matplotlib.pyplot as plt

from QSER.Backends import get_backend
from QSER.Operators import Gradient
from QSER.Mesh.interpolation import Interpolation


class Structured2D:
    """
    2D Structured Mesh with 5-point (4th order) gradient.

    Operators:
        - Gradient: 5-point central difference (4th order)
        - Laplacian: 5-point central difference (4th order)
        - Divergence: Face-centered fluxes (2nd order)
        - Curl: Built from 5-point gradient (4th order)

    Parameters:
        nx, ny: int, number of cells in each direction (default: 20)
        Lx, Ly: float, domain lengths (default: 1.0)
        uniform: bool, uniform grid (default: True)
        x, y: array, custom coordinates (if uniform=False)
        backend: str, 'numpy', 'jax', 'torch' (default: 'numpy')
        gradient_method: str, gradient method (default: 'least_squares')
        interpolation_method: str, interpolation method (default: 'linear')
    """

    def __init__(self, nx=20, ny=20, Lx=1.0, Ly=1.0, uniform=True,
                 x=None, y=None, backend='numpy', gradient_method='least_squares',
                 interpolation_method='linear'):
        self.dim = 2
        self.nx = nx
        self.ny = ny
        self.n_cells = nx * ny
        self.n_nodes = (nx + 1) * (ny + 1)
        self.Lx = Lx
        self.Ly = Ly

        self.backend_name = backend
        self.backend = get_backend(backend)

        if x is not None and y is not None:
            self.x = self.backend.array(x)
            self.y = self.backend.array(y)
            self.nx = len(x) - 1
            self.ny = len(y) - 1
            self.Lx = x[-1] - x[0]
            self.Ly = y[-1] - y[0]
        elif uniform:
            self.x = self.backend.linspace(0, Lx, nx + 1)
            self.y = self.backend.linspace(0, Ly, ny + 1)
        else:
            self.x = self.backend.linspace(0, Lx, nx + 1)**2 / Lx
            self.y = self.backend.linspace(0, Ly, ny + 1)**2 / Ly

        self.dx = self.x[1:] - self.x[:-1]
        self.dy = self.y[1:] - self.y[:-1]

        self.X_cell, self.Y_cell = self.backend.meshgrid(
            (self.x[:-1] + self.x[1:]) / 2,
            (self.y[:-1] + self.y[1:]) / 2,
            indexing='ij'
        )

        self.cell_volumes = self.dx[:, None] * self.dy[None, :]

        self.X_face_x, self.Y_face_x = self.backend.meshgrid(
            self.x,
            (self.y[:-1] + self.y[1:]) / 2,
            indexing='ij'
        )
        self.X_face_y, self.Y_face_y = self.backend.meshgrid(
            (self.x[:-1] + self.x[1:]) / 2,
            self.y,
            indexing='ij'
        )

        self.face_areas_x = self.dy[None, :]
        self.face_areas_y = self.dx[:, None]

        self.n_faces_x = (self.nx + 1) * self.ny
        self.n_faces_y = self.nx * (self.ny + 1)
        self.n_faces = self.n_faces_x + self.n_faces_y

        self.gradient = Gradient(self, method=gradient_method, backend=backend)
        self.interpolation = Interpolation(self, method=interpolation_method, backend=backend)

        if self.backend.name == 'jax':
            self._compile_jax_functions()

    def _compile_jax_functions(self):
        import jax
        self._gradient_jax = jax.jit(self._compute_gradient_jax)
        self._laplacian_jax = jax.jit(self._compute_laplacian_jax)
        self._divergence_jax = jax.jit(self._compute_divergence_jax)
        self._curl_jax = jax.jit(self._compute_curl_jax)

    def _to_numpy(self, array):
        return self.backend.to_numpy(array)

    def _to_backend(self, array):
        if not self.backend.is_backend_array(array):
            return self.backend.array(array)
        return array

    # ============================================================
    # GRADIENT METHOD MANAGEMENT
    # ============================================================

    def set_gradient_method(self, method):
        self.gradient.set_method(method)

    def get_gradient_method(self):
        return self.gradient.get_method()

    def available_gradient_methods(self):
        return self.gradient.available_methods()

    # ============================================================
    # INTERPOLATION METHOD MANAGEMENT
    # ============================================================

    def set_interpolation_method(self, method):
        self.interpolation.set_method(method)

    def get_interpolation_method(self):
        return self.interpolation.get_method()

    def available_interpolation_methods(self):
        return self.interpolation.available_methods()

    # ============================================================
    # GEOMETRY ACCESS
    # ============================================================

    def get_cell_centers(self, return_backend=False):
        if return_backend:
            return self.backend.stack([self.X_cell, self.Y_cell], axis=-1)
        return self._to_numpy(self.backend.stack([self.X_cell, self.Y_cell], axis=-1))

    def get_cell_volumes(self, return_backend=False):
        if return_backend:
            return self.cell_volumes
        return self._to_numpy(self.cell_volumes)

    def get_face_centers_x(self, return_backend=False):
        if return_backend:
            return self.backend.stack([self.X_face_x, self.Y_face_x], axis=-1)
        return self._to_numpy(self.backend.stack([self.X_face_x, self.Y_face_x], axis=-1))

    def get_face_centers_y(self, return_backend=False):
        if return_backend:
            return self.backend.stack([self.X_face_y, self.Y_face_y], axis=-1)
        return self._to_numpy(self.backend.stack([self.X_face_y, self.Y_face_y], axis=-1))

    def get_face_areas_x(self, return_backend=False):
        if return_backend:
            return self.face_areas_x
        return self._to_numpy(self.face_areas_x)

    def get_face_areas_y(self, return_backend=False):
        if return_backend:
            return self.face_areas_y
        return self._to_numpy(self.face_areas_y)

    # ============================================================
    # OPERATORS
    # ============================================================

    def compute_gradient(self, field, method=None, return_backend=False):
        grad = self.gradient.compute(field, method=method)
        if return_backend:
            return grad
        return self._to_numpy(grad)

    def compute_laplacian(self, field, return_backend=False):
        field = self._to_backend(field)
        if self.backend.name == 'jax':
            lap = self._laplacian_jax(field, self.dx, self.dy)
        else:
            lap = self._compute_laplacian_generic(field)
        if return_backend:
            return lap
        return self._to_numpy(lap)

    def _compute_laplacian_generic(self, field):
        nx, ny = field.shape
        lap = self.backend.zeros_like(field)

        # Interior: 5-point Laplacian (4th order)
        lap = self.backend.set_item_slice(lap, 2, -2, 2, -2,
            (-field[4:, 2:-2] + 16*field[3:-1, 2:-2] - 30*field[2:-2, 2:-2] + 16*field[1:-3, 2:-2] - field[:-4, 2:-2]) / (12 * self.dx[2:-2, None]**2) +
            (-field[2:-2, 4:] + 16*field[2:-2, 3:-1] - 30*field[2:-2, 2:-2] + 16*field[2:-2, 1:-3] - field[2:-2, :-4]) / (12 * self.dy[None, 2:-2]**2)
        )

        # Boundaries: 3-point (2nd order) with proper broadcasting
        lap = self.backend.set_item_slice(lap, 0, 1, 0, ny, (field[1, :] - 2*field[0, :] + field[0, :]) / self.dx[0]**2)
        lap = self.backend.set_item_slice(lap, -1, -1, 0, ny, (field[-1, :] - 2*field[-1, :] + field[-2, :]) / self.dx[-1]**2)
        lap = self.backend.set_item_slice(lap, 0, nx, 0, 1, (field[:, 1] - 2*field[:, 0] + field[:, 0])[:, None] / self.dy[0]**2)
        lap = self.backend.set_item_slice(lap, 0, nx, -1, -1, (field[:, -1] - 2*field[:, -1] + field[:, -2])[:, None] / self.dy[-1]**2)

        return lap

    def compute_divergence(self, flux_x, flux_y, return_backend=False):
        flux_x = self._to_backend(flux_x)
        flux_y = self._to_backend(flux_y)

        if self.backend.name == 'jax':
            div = self._divergence_jax(flux_x, flux_y, self.dx, self.dy)
        else:
            div = self._compute_divergence_generic(flux_x, flux_y)

        if return_backend:
            return div
        return self._to_numpy(div)

    def _compute_divergence_generic(self, flux_x, flux_y):
        dx_reshaped = self.dx.reshape(-1, 1)
        dy_reshaped = self.dy.reshape(1, -1)
        div_x = (flux_x[1:, :] - flux_x[:-1, :]) / dx_reshaped
        div_y = (flux_y[:, 1:] - flux_y[:, :-1]) / dy_reshaped
        div = div_x + div_y
        return div

    def compute_curl(self, Vx, Vy, return_backend=False):
        Vx = self._to_backend(Vx)
        Vy = self._to_backend(Vy)

        if self.backend.name == 'jax':
            curl = self._curl_jax(Vx, Vy, self.dx, self.dy)
        else:
            dVy_dx, dVy_dy = self._compute_gradient_generic(Vy)
            dVx_dx, dVx_dy = self._compute_gradient_generic(Vx)
            curl = dVy_dx - dVx_dy

        if return_backend:
            return curl
        return self._to_numpy(curl)

    def _compute_gradient_generic(self, field):
        nx, ny = field.shape
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)

        # Interior: 5-point central difference (4th order)
        grad_x = self.backend.set_item_slice(grad_x, 2, -2, 0, ny,
            (-field[4:, :] + 8*field[3:-1, :] - 8*field[1:-3, :] + field[:-4, :]) / (12 * self.dx[2:-2, None])
        )
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 2, -2,
            (-field[:, 4:] + 8*field[:, 3:-1] - 8*field[:, 1:-3] + field[:, :-4]) / (12 * self.dy[None, 2:-2])
        )

        # Boundaries: 3-point (2nd order)
        grad_x = self.backend.set_slice(grad_x, 0, 1, (field[1, :] - field[0, :]) / self.dx[0])
        grad_x = self.backend.set_slice(grad_x, 1, 2, (field[2, :] - field[0, :]) / (2 * self.dx[1]))
        grad_x = self.backend.set_slice(grad_x, -1, -1, (field[-1, :] - field[-2, :]) / self.dx[-1])
        grad_x = self.backend.set_slice(grad_x, -2, -2, (field[-1, :] - field[-3, :]) / (2 * self.dx[-2]))

        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 0, 1, (field[:, 1] - field[:, 0])[:, None] / self.dy[0])
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 1, 2, (field[:, 2] - field[:, 0])[:, None] / (2 * self.dy[1]))
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, -1, -1, (field[:, -1] - field[:, -2])[:, None] / self.dy[-1])
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, -2, -2, (field[:, -1] - field[:, -3])[:, None] / (2 * self.dy[-2]))

        return grad_x, grad_y

    # ============================================================
    # INTERPOLATION
    # ============================================================

    def interpolate_to_faces(self, field, method=None, return_backend=False):
        result = self.interpolation.to_faces(field, method=method)
        if return_backend:
            return result
        if isinstance(result, tuple):
            return tuple(self._to_numpy(r) for r in result)
        return self._to_numpy(result)

    # ============================================================
    # BOUNDARY CONDITIONS
    # ============================================================

    def apply_bc(self, field, bc_type, bc_value, return_backend=False):
        field = self._to_backend(field)

        if bc_type == 'dirichlet':
            field = self.backend.set_item_slice(field, 0, 1, 0, self.ny, bc_value[0])
            field = self.backend.set_item_slice(field, -1, -1, 0, self.ny, bc_value[1])
            field = self.backend.set_item_slice(field, 0, self.nx, 0, 1, bc_value[2])
            field = self.backend.set_item_slice(field, 0, self.nx, -1, -1, bc_value[3])
        elif bc_type == 'neumann':
            left_val = field[1, :] - bc_value[0] * self.dx[0]
            right_val = field[-2, :] + bc_value[1] * self.dx[-1]
            bottom_val = (field[:, 1] - bc_value[2] * self.dy[0])[:, None]
            top_val = (field[:, -2] + bc_value[3] * self.dy[-1])[:, None]
            field = self.backend.set_item_slice(field, 0, 1, 0, self.ny, left_val)
            field = self.backend.set_item_slice(field, -1, -1, 0, self.ny, right_val)
            field = self.backend.set_item_slice(field, 0, self.nx, 0, 1, bottom_val)
            field = self.backend.set_item_slice(field, 0, self.nx, -1, -1, top_val)
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

        if return_backend:
            return field
        return self._to_numpy(field)

    # ============================================================
    # PLOTTING
    # ============================================================

    def plot(self, field=None, title="2D Mesh", figsize=(10, 8),
             plot_type='contourf', return_fig=False):
        fig, ax = plt.subplots(figsize=figsize)

        x_np = self._to_numpy(self.x)
        y_np = self._to_numpy(self.y)

        if plot_type == 'mesh':
            for i in range(len(x_np)):
                ax.axvline(x_np[i], color='gray', linestyle='-', alpha=0.3)
            for j in range(len(y_np)):
                ax.axhline(y_np[j], color='gray', linestyle='-', alpha=0.3)

        elif field is not None:
            field_np = self._to_numpy(field)
            X_cell_np = self._to_numpy(self.X_cell)
            Y_cell_np = self._to_numpy(self.Y_cell)

            if plot_type == 'contourf':
                cf = ax.contourf(X_cell_np, Y_cell_np, field_np.T, levels=20, cmap='viridis')
                plt.colorbar(cf, ax=ax, label='Field Value')
            elif plot_type == 'pcolormesh':
                im = ax.pcolormesh(x_np, y_np, field_np.T, shading='auto', cmap='viridis')
                plt.colorbar(im, ax=ax, label='Field Value')
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()

    # ============================================================
    # JAX BACKEND IMPLEMENTATIONS (JIT-compiled)
    # ============================================================

    def _compute_gradient_jax(self, field, dx, dy):
        nx, ny = field.shape
        grad_x = self.backend.zeros_like(field)
        grad_y = self.backend.zeros_like(field)

        grad_x = self.backend.set_item_slice(grad_x, 2, -2, 0, ny,
            (-field[4:, :] + 8*field[3:-1, :] - 8*field[1:-3, :] + field[:-4, :]) / (12 * dx[2:-2, None])
        )
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 2, -2,
            (-field[:, 4:] + 8*field[:, 3:-1] - 8*field[:, 1:-3] + field[:, :-4]) / (12 * dy[None, 2:-2])
        )

        grad_x = self.backend.set_slice(grad_x, 0, 1, (field[1, :] - field[0, :]) / dx[0])
        grad_x = self.backend.set_slice(grad_x, 1, 2, (field[2, :] - field[0, :]) / (2 * dx[1]))
        grad_x = self.backend.set_slice(grad_x, -1, -1, (field[-1, :] - field[-2, :]) / dx[-1])
        grad_x = self.backend.set_slice(grad_x, -2, -2, (field[-1, :] - field[-3, :]) / (2 * dx[-2]))

        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 0, 1, (field[:, 1] - field[:, 0])[:, None] / dy[0])
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, 1, 2, (field[:, 2] - field[:, 0])[:, None] / (2 * dy[1]))
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, -1, -1, (field[:, -1] - field[:, -2])[:, None] / dy[-1])
        grad_y = self.backend.set_item_slice(grad_y, 0, nx, -2, -2, (field[:, -1] - field[:, -3])[:, None] / (2 * dy[-2]))

        return grad_x, grad_y

    def _compute_laplacian_jax(self, field, dx, dy):
        nx, ny = field.shape
        lap = self.backend.zeros_like(field)
        
        # Interior: 5-point Laplacian (4th order) - only if grid is large enough
        if nx >= 5 and ny >= 5:
            lap = self.backend.set_item_slice(lap, 2, -2, 2, -2,
                (-field[4:, 2:-2] + 16*field[3:-1, 2:-2] - 30*field[2:-2, 2:-2] + 16*field[1:-3, 2:-2] - field[:-4, 2:-2]) / (12 * dx[2:-2, None]**2) +
                (-field[2:-2, 4:] + 16*field[2:-2, 3:-1] - 30*field[2:-2, 2:-2] + 16*field[2:-2, 1:-3] - field[2:-2, :-4]) / (12 * dy[None, 2:-2]**2)
            )
        else:
            # Fallback: 3-point Laplacian for small grids
            if nx >= 3 and ny >= 3:
                lap_x = (field[2:, :] - 2*field[1:-1, :] + field[:-2, :]) / dx[1:-1, None]**2
                lap = self.backend.set_item_slice(lap, 1, -1, 0, ny, lap_x)
                lap_y = (field[:, 2:] - 2*field[:, 1:-1] + field[:, :-2]) / dy[None, 1:-1]**2
                lap = self.backend.set_item_slice(lap, 0, nx, 1, -1, lap_y)
        
        # Boundaries: 3-point (2nd order)
        if nx >= 3:
            lap = self.backend.set_item_slice(lap, 0, 1, 0, ny, (field[1, :] - 2*field[0, :] + field[0, :]) / dx[0]**2)
            lap = self.backend.set_item_slice(lap, -1, -1, 0, ny, (field[-1, :] - 2*field[-1, :] + field[-2, :]) / dx[-1]**2)
        if ny >= 3:
            lap = self.backend.set_item_slice(lap, 0, nx, 0, 1, (field[:, 1] - 2*field[:, 0] + field[:, 0])[:, None] / dy[0]**2)
            lap = self.backend.set_item_slice(lap, 0, nx, -1, -1, (field[:, -1] - 2*field[:, -1] + field[:, -2])[:, None] / dy[-1]**2)
        
        return lap

    def _compute_divergence_jax(self, flux_x, flux_y, dx, dy):
        dx_reshaped = dx.reshape(-1, 1)
        dy_reshaped = dy.reshape(1, -1)
        div_x = (flux_x[1:, :] - flux_x[:-1, :]) / dx_reshaped
        div_y = (flux_y[:, 1:] - flux_y[:, :-1]) / dy_reshaped
        div = div_x + div_y
        return div

    def _compute_curl_jax(self, Vx, Vy, dx, dy):
        dVy_dx, dVy_dy = self._compute_gradient_jax(Vy, dx, dy)
        dVx_dx, dVx_dy = self._compute_gradient_jax(Vx, dx, dy)
        return dVy_dx - dVx_dy
