import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from QSER.Backends import get_backend
from QSER.Operators import Gradient
from QSER.Mesh.interpolation import Interpolation


class Structured1D:
    """
    1D Structured Mesh with 5-point (4th order) gradient.

    Operators:
        - Gradient: 5-point central difference (4th order)
        - Laplacian: 5-point central difference (4th order)
        - Divergence: Face-centered fluxes (2nd order)

    Parameters:
        nx: int, number of cells (default: 100)
        L: float, domain length (default: 1.0)
        uniform: bool, uniform grid (default: True)
        x: array, custom node coordinates (if uniform=False)
        backend: str, 'numpy', 'jax', 'torch' (default: 'numpy')
        gradient_method: str, gradient method (default: 'least_squares')
        interpolation_method: str, interpolation method (default: 'linear')
    """

    def __init__(self, nx=100, L=1.0, uniform=True, x=None,
                 backend='numpy', gradient_method='least_squares',
                 interpolation_method='linear'):
        self.dim = 1
        self.backend_name = backend

        self.backend = get_backend(backend)

        if x is not None:
            self.x = self.backend.array(x)
            self.n_nodes = len(x)
            self.n_cells = self.n_nodes - 1
            self.nx = self.n_cells
            self.L = x[-1] - x[0]
        elif uniform:
            self.x = self.backend.linspace(0, L, nx + 1)
            self.n_nodes = nx + 1
            self.n_cells = nx
            self.nx = nx
            self.L = L
        else:
            self.x = self.backend.linspace(0, L, nx + 1)**2 / L
            self.n_nodes = nx + 1
            self.n_cells = nx
            self.nx = nx
            self.L = L

        self.dx = self.x[1:] - self.x[:-1]
        self.cell_centers = (self.x[:-1] + self.x[1:]) / 2
        self.cell_volumes = self.dx
        self.face_centers = self.x
        self.n_faces = self.n_nodes

        self.gradient = Gradient(self, method=gradient_method, backend=backend)
        self.interpolation = Interpolation(self, method=interpolation_method, backend=backend)

        if self.backend.name == 'jax':
            self._compile_jax_functions()

    def _compile_jax_functions(self):
        import jax
        self._gradient_jax = jax.jit(self._compute_gradient_jax)
        self._laplacian_jax = jax.jit(self._compute_laplacian_jax)
        self._divergence_jax = jax.jit(self._compute_divergence_jax)

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
        """Set the gradient method."""
        self.gradient.set_method(method)

    def get_gradient_method(self):
        """Get the current gradient method."""
        return self.gradient.get_method()

    def available_gradient_methods(self):
        """Return list of available gradient methods."""
        return self.gradient.available_methods()

    # ============================================================
    # INTERPOLATION METHOD MANAGEMENT
    # ============================================================

    def set_interpolation_method(self, method):
        """Set the interpolation method."""
        self.interpolation.set_method(method)

    def get_interpolation_method(self):
        """Get the current interpolation method."""
        return self.interpolation.get_method()

    def available_interpolation_methods(self):
        """Return list of available interpolation methods."""
        return self.interpolation.available_methods()

    # ============================================================
    # GEOMETRY ACCESS
    # ============================================================

    def get_cell_centers(self, return_backend=False):
        if return_backend:
            return self.cell_centers
        return self._to_numpy(self.cell_centers)

    def get_cell_volumes(self, return_backend=False):
        if return_backend:
            return self.cell_volumes
        return self._to_numpy(self.cell_volumes)

    def get_face_centers(self, return_backend=False):
        if return_backend:
            return self.face_centers
        return self._to_numpy(self.face_centers)

    # ============================================================
    # OPERATORS
    # ============================================================

    def compute_gradient(self, field, method=None, return_backend=False):
        grad = self.gradient.compute(field, method=method)
        if return_backend:
            return grad
        return self._to_numpy(grad)

    def compute_laplacian(self, field, return_backend=False):
        if self.backend.name == 'jax' and not isinstance(field, jnp.ndarray):
            field = self.backend.array(field)

        if self.backend.name == 'jax':
            lap = self._laplacian_jax(field, self.dx)
        else:
            lap = self._compute_laplacian_generic(field)

        if return_backend:
            return lap
        return self._to_numpy(lap)

    def _compute_laplacian_generic(self, field):
        n = field.shape[0]
        lap = self.backend.zeros(n)

        lap = self.backend.set_slice(lap, 2, -2,
            (-field[4:] + 16*field[3:-1] - 30*field[2:-2] + 16*field[1:-3] - field[:-4]) / (12 * self.dx[2:-2]**2)
        )

        lap = self.backend.set_item(lap, 0, (field[2] - 2*field[1] + field[0]) / self.dx[0]**2)
        lap = self.backend.set_item(lap, 1, (field[3] - 2*field[2] + field[1]) / self.dx[1]**2)
        lap = self.backend.set_item(lap, -1, (field[-1] - 2*field[-2] + field[-3]) / self.dx[-1]**2)
        lap = self.backend.set_item(lap, -2, (field[-1] - 2*field[-2] + field[-3]) / self.dx[-2]**2)

        return lap

    def compute_divergence(self, flux, return_backend=False):
        if self.backend.name == 'jax' and not isinstance(flux, jnp.ndarray):
            flux = self.backend.array(flux)

        if self.backend.name == 'jax':
            div = self._divergence_jax(flux, self.dx)
        else:
            div = self._compute_divergence_generic(flux)

        if return_backend:
            return div
        return self._to_numpy(div)

    def _compute_divergence_generic(self, flux):
        div = self.backend.zeros(self.n_cells)
        div = self.backend.set_slice(div, 0, self.n_cells, (flux[1:] - flux[:-1]) / self.dx)
        return div

    # ============================================================
    # INTERPOLATION
    # ============================================================

    def interpolate_to_faces(self, field, method=None, return_backend=False):
        result = self.interpolation.to_faces(field, method=method)
        if return_backend:
            return result
        return self._to_numpy(result)

    # ============================================================
    # BOUNDARY CONDITIONS
    # ============================================================

    def apply_bc(self, field, bc_type, bc_value, return_backend=False):
        if self.backend.name == 'jax' and not isinstance(field, jnp.ndarray):
            field = self.backend.array(field)

        if bc_type == 'dirichlet':
            field = self.backend.set_item(field, 0, bc_value[0])
            field = self.backend.set_item(field, -1, bc_value[1])
        elif bc_type == 'neumann':
            left_val = field[1] - bc_value[0] * self.dx[0]
            right_val = field[-2] + bc_value[1] * self.dx[-1]
            field = self.backend.set_item(field, 0, left_val)
            field = self.backend.set_item(field, -1, right_val)
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

        if return_backend:
            return field
        return self._to_numpy(field)

    # ============================================================
    # PLOTTING
    # ============================================================

    def plot(self, field=None, title="1D Mesh", figsize=(10, 4), return_fig=False):
        fig, ax = plt.subplots(figsize=figsize)

        x_np = self.get_face_centers()
        centers_np = self.get_cell_centers()

        ax.plot(x_np, np.zeros_like(x_np), 'ko-', markersize=4, label='Nodes')
        ax.plot(centers_np, np.zeros_like(centers_np), 'r^', markersize=6, label='Cell Centers')

        if field is not None:
            field_np = self._to_numpy(field)
            ax.plot(centers_np, field_np, 'b-', linewidth=2, label='Field')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()

    # ============================================================
    # JAX BACKEND IMPLEMENTATIONS (JIT-compiled)
    # ============================================================

    def _compute_gradient_jax(self, field, cell_centers, dx):
        n = field.shape[0]
        grad = self.backend.zeros(n)

        grad = self.backend.set_slice(grad, 2, -2,
            (-field[4:] + 8*field[3:-1] - 8*field[1:-3] + field[:-4]) / (12 * dx[2:-2])
        )
        grad = self.backend.set_item(grad, 0, (field[1] - field[0]) / dx[0])
        grad = self.backend.set_item(grad, 1, (field[2] - field[0]) / (2 * dx[1]))
        grad = self.backend.set_item(grad, -1, (field[-1] - field[-2]) / dx[-1])
        grad = self.backend.set_item(grad, -2, (field[-1] - field[-3]) / (2 * dx[-2]))

        return grad

    def _compute_laplacian_jax(self, field, dx):
        n = field.shape[0]
        lap = self.backend.zeros(n)

        lap = self.backend.set_slice(lap, 2, -2,
            (-field[4:] + 16*field[3:-1] - 30*field[2:-2] + 16*field[1:-3] - field[:-4]) / (12 * dx[2:-2]**2)
        )
        lap = self.backend.set_item(lap, 0, (field[2] - 2*field[1] + field[0]) / dx[0]**2)
        lap = self.backend.set_item(lap, 1, (field[3] - 2*field[2] + field[1]) / dx[1]**2)
        lap = self.backend.set_item(lap, -1, (field[-1] - 2*field[-1] + field[-2]) / dx[-1]**2)
        lap = self.backend.set_item(lap, -2, (field[-1] - 2*field[-2] + field[-3]) / dx[-2]**2)

        return lap

    def _compute_divergence_jax(self, flux, dx):
        n = flux.shape[0] - 1
        div = self.backend.zeros(n)
        div = self.backend.set_slice(div, 0, n, (flux[1:] - flux[:-1]) / dx)
        return div
