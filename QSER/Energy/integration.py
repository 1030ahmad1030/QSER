"""
Integration Engine for QSER
===========================

Provides numerical integration methods with backend awareness.

Methods:
    - trapezoidal:  2nd order (O(h²))
    - simpson:      4th order (O(h⁴)) 
    - monte_carlo:  Crude Monte Carlo (O(1/√N))
    - quasi_monte_carlo: Sobol sequence (O(1/N))

Sampling Distributions:
    - uniform:     Uniform random (default)
    - normal:      Normal/Gaussian distribution
    - lognormal:   Log-normal distribution
    - custom:      User-provided distribution

Future methods:
    - gaussian:     Gaussian quadrature
    - midpoint:     Midpoint rule
    - adaptive:     Adaptive quadrature
"""

import numpy as np
from QSER.Backends import get_backend


class IntegrationEngine:
    """
    Backend-aware numerical integration engine.
    """

    def __init__(self, backend='numpy'):
        self.backend_name = backend
        self.backend = get_backend(backend)

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def integrate_function(self, f, a, b, n=100, method='trapezoidal'):
        """∫ f(x) dx from a to b."""
        if method == 'trapezoidal':
            return self._trapezoidal_function(f, a, b, n)
        elif method == 'simpson':
            return self._simpson_function(f, a, b, n)
        elif method == 'monte_carlo':
            return self._monte_carlo_function(f, a, b, n)
        elif method == 'quasi_monte_carlo':
            return self._quasi_monte_carlo_function(f, a, b, n)
        else:
            raise ValueError(f"Unknown method: {method}")

    def integrate_field(self, field, dx=None, dy=None, dz=None, dt=None,
                        method='trapezoidal', n_points=10000, distribution='uniform'):
        """∫ field dV over the domain."""
        # Automatic backend conversion (user provides NumPy)
        if isinstance(field, np.ndarray):
            if self.backend_name == 'torch':
                import torch
                field = torch.tensor(field, dtype=torch.float64)
            elif self.backend_name == 'jax':
                import jax.numpy as jnp
                field = jnp.array(field)

        ndim = len(field.shape)

        # Monte Carlo methods
        if method == 'monte_carlo':
            volume = self._compute_volume(field, dx, dy, dz, dt)
            return self._monte_carlo_field(field, n_points, volume, distribution)
        elif method == 'quasi_monte_carlo':
            volume = self._compute_volume(field, dx, dy, dz, dt)
            return self._quasi_monte_carlo_field(field, n_points, volume)

        # Deterministic methods
        if ndim == 1:
            if dx is None:
                raise ValueError("dx required for 1D integration")
            if method == 'trapezoidal':
                return self._trapezoidal_1d(field, dx)
            elif method == 'simpson':
                return self._simpson_1d(field, dx)

        elif ndim == 2:
            if dx is None or dy is None:
                raise ValueError("dx and dy required for 2D integration")
            if method == 'trapezoidal':
                return self._trapezoidal_2d(field, dx, dy)
            elif method == 'simpson':
                return self._simpson_2d(field, dx, dy)

        elif ndim == 3:
            if dx is None or dy is None or dz is None:
                raise ValueError("dx, dy, dz required for 3D integration")
            if method == 'trapezoidal':
                return self._trapezoidal_3d(field, dx, dy, dz)
            elif method == 'simpson':
                return self._simpson_3d(field, dx, dy, dz)

        elif ndim == 4:
            if dx is None or dy is None or dz is None or dt is None:
                raise ValueError("dx, dy, dz, dt required for 4D integration")
            if method == 'trapezoidal':
                return self._trapezoidal_4d(field, dx, dy, dz, dt)
            elif method == 'simpson':
                return self._simpson_4d(field, dx, dy, dz, dt)

        else:
            volume = self._compute_volume(field, dx, dy, dz, dt)
            return self._monte_carlo_field(field, n_points, volume, distribution)

    # ============================================================
    # FUNCTION INTEGRATION
    # ============================================================

    def _trapezoidal_function(self, f, a, b, n):
        h = (b - a) / n
        x = self.backend.linspace(a, b, n + 1)
        y = f(x)
        integral = h * (0.5 * y[0] + self._sum(y[1:-1]) + 0.5 * y[-1])
        return self._to_scalar(integral)

    def _simpson_function(self, f, a, b, n):
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        x = self.backend.linspace(a, b, n + 1)
        y = f(x)
        integral = (h / 3) * (
            y[0] + y[-1] +
            4 * self._sum(y[1:-1:2]) +
            2 * self._sum(y[2:-2:2])
        )
        return self._to_scalar(integral)

    def _monte_carlo_function(self, f, a, b, n):
        x = self._random(n) * (b - a) + a
        values = f(x)
        integral = (b - a) * self._mean(values)
        return self._to_scalar(integral)

    def _quasi_monte_carlo_function(self, f, a, b, n):
        x = self._sobol_sequence(n, 1)
        if len(x.shape) == 2:
            x = x[:, 0]
        values = f(x)
        integral = (b - a) * self._mean(values)
        return self._to_scalar(integral)

    # ============================================================
    # 1D FIELD INTEGRATION
    # ============================================================

    def _trapezoidal_1d(self, field, dx):
        integral = dx * (0.5 * field[0] + self._sum(field[1:-1]) + 0.5 * field[-1])
        return self._to_scalar(integral)

    def _simpson_1d(self, field, dx):
        n = len(field)
        if n % 2 == 0:
            integral = self._trapezoidal_1d(field, dx)
        else:
            integral = (dx / 3) * (
                field[0] + field[-1] +
                4 * self._sum(field[1:-1:2]) +
                2 * self._sum(field[2:-2:2])
            )
        return self._to_scalar(integral)

    # ============================================================
    # 2D FIELD INTEGRATION
    # ============================================================

    def _trapezoidal_2d(self, field, dx, dy):
        integral_x = dx * (0.5 * field[0, :] + self._sum(field[1:-1, :], axis=0) + 0.5 * field[-1, :])
        integral = dy * (0.5 * integral_x[0] + self._sum(integral_x[1:-1]) + 0.5 * integral_x[-1])
        return self._to_scalar(integral)

    def _simpson_2d(self, field, dx, dy):
        nx, ny = field.shape
        if nx % 2 == 0:
            integral_x = dx * (0.5 * field[0, :] + self._sum(field[1:-1, :], axis=0) + 0.5 * field[-1, :])
        else:
            integral_x = (dx / 3) * (
                field[0, :] + field[-1, :] +
                4 * self._sum(field[1:-1:2, :], axis=0) +
                2 * self._sum(field[2:-2:2, :], axis=0)
            )
        if ny % 2 == 0:
            integral = dy * (0.5 * integral_x[0] + self._sum(integral_x[1:-1]) + 0.5 * integral_x[-1])
        else:
            integral = (dy / 3) * (
                integral_x[0] + integral_x[-1] +
                4 * self._sum(integral_x[1:-1:2]) +
                2 * self._sum(integral_x[2:-2:2])
            )
        return self._to_scalar(integral)

    # ============================================================
    # 3D FIELD INTEGRATION
    # ============================================================

    def _trapezoidal_3d(self, field, dx, dy, dz):
        integral_x = dx * (0.5 * field[0, :, :] + self._sum(field[1:-1, :, :], axis=0) + 0.5 * field[-1, :, :])
        integral_xy = dy * (0.5 * integral_x[0, :] + self._sum(integral_x[1:-1, :], axis=0) + 0.5 * integral_x[-1, :])
        integral = dz * (0.5 * integral_xy[0] + self._sum(integral_xy[1:-1]) + 0.5 * integral_xy[-1])
        return self._to_scalar(integral)

    def _simpson_3d(self, field, dx, dy, dz):
        nx, ny, nz = field.shape
        if nx % 2 == 0:
            integral_x = dx * (0.5 * field[0, :, :] + self._sum(field[1:-1, :, :], axis=0) + 0.5 * field[-1, :, :])
        else:
            integral_x = (dx / 3) * (
                field[0, :, :] + field[-1, :, :] +
                4 * self._sum(field[1:-1:2, :, :], axis=0) +
                2 * self._sum(field[2:-2:2, :, :], axis=0)
            )
        if ny % 2 == 0:
            integral_xy = dy * (0.5 * integral_x[0, :] + self._sum(integral_x[1:-1, :], axis=0) + 0.5 * integral_x[-1, :])
        else:
            integral_xy = (dy / 3) * (
                integral_x[0, :] + integral_x[-1, :] +
                4 * self._sum(integral_x[1:-1:2, :], axis=0) +
                2 * self._sum(integral_x[2:-2:2, :], axis=0)
            )
        if nz % 2 == 0:
            integral = dz * (0.5 * integral_xy[0] + self._sum(integral_xy[1:-1]) + 0.5 * integral_xy[-1])
        else:
            integral = (dz / 3) * (
                integral_xy[0] + integral_xy[-1] +
                4 * self._sum(integral_xy[1:-1:2]) +
                2 * self._sum(integral_xy[2:-2:2])
            )
        return self._to_scalar(integral)

    # ============================================================
    # 4D FIELD INTEGRATION
    # ============================================================

    def _trapezoidal_4d(self, field, dx, dy, dz, dt):
        nx, ny, nz, nt = field.shape
        integral_x = dx * (0.5 * field[0, :, :, :] + self._sum(field[1:-1, :, :, :], axis=0) + 0.5 * field[-1, :, :, :])
        integral_xy = dy * (0.5 * integral_x[0, :, :] + self._sum(integral_x[1:-1, :, :], axis=0) + 0.5 * integral_x[-1, :, :])
        integral_xyz = dz * (0.5 * integral_xy[0, :] + self._sum(integral_xy[1:-1, :], axis=0) + 0.5 * integral_xy[-1, :])
        integral = dt * (0.5 * integral_xyz[0] + self._sum(integral_xyz[1:-1]) + 0.5 * integral_xyz[-1])
        return self._to_scalar(integral)

    def _simpson_4d(self, field, dx, dy, dz, dt):
        nx, ny, nz, nt = field.shape
        if nx % 2 == 0:
            integral_x = dx * (0.5 * field[0, :, :, :] + self._sum(field[1:-1, :, :, :], axis=0) + 0.5 * field[-1, :, :, :])
        else:
            integral_x = (dx / 3) * (
                field[0, :, :, :] + field[-1, :, :, :] +
                4 * self._sum(field[1:-1:2, :, :, :], axis=0) +
                2 * self._sum(field[2:-2:2, :, :, :], axis=0)
            )
        if ny % 2 == 0:
            integral_xy = dy * (0.5 * integral_x[0, :, :] + self._sum(integral_x[1:-1, :, :], axis=0) + 0.5 * integral_x[-1, :, :])
        else:
            integral_xy = (dy / 3) * (
                integral_x[0, :, :] + integral_x[-1, :, :] +
                4 * self._sum(integral_x[1:-1:2, :, :], axis=0) +
                2 * self._sum(integral_x[2:-2:2, :, :], axis=0)
            )
        if nz % 2 == 0:
            integral_xyz = dz * (0.5 * integral_xy[0, :] + self._sum(integral_xy[1:-1, :], axis=0) + 0.5 * integral_xy[-1, :])
        else:
            integral_xyz = (dz / 3) * (
                integral_xy[0, :] + integral_xy[-1, :] +
                4 * self._sum(integral_xy[1:-1:2, :], axis=0) +
                2 * self._sum(integral_xy[2:-2:2, :], axis=0)
            )
        if nt % 2 == 0:
            integral = dt * (0.5 * integral_xyz[0] + self._sum(integral_xyz[1:-1]) + 0.5 * integral_xyz[-1])
        else:
            integral = (dt / 3) * (
                integral_xyz[0] + integral_xyz[-1] +
                4 * self._sum(integral_xyz[1:-1:2]) +
                2 * self._sum(integral_xyz[2:-2:2])
            )
        return self._to_scalar(integral)

    # ============================================================
    # MONTE CARLO METHODS
    # ============================================================

    def _compute_volume(self, field, dx, dy, dz, dt):
        shape = field.shape
        volume = 1.0
        spacings = []
        if dx is not None:
            spacings.append(dx)
        if dy is not None:
            spacings.append(dy)
        if dz is not None:
            spacings.append(dz)
        if dt is not None:
            spacings.append(dt)

        if len(spacings) > 0:
            for i, s in enumerate(spacings):
                if i < len(shape):
                    volume *= shape[i] * s
        else:
            for s in shape:
                volume *= s
        return volume

    def _monte_carlo_field(self, field, n_points, volume, distribution='uniform'):
        flat_field = self._flatten(field)
        n_total = len(flat_field)

        if distribution == 'uniform':
            indices = self._randint(0, n_total, n_points)
        elif distribution == 'normal':
            samples = self._random(n_points, distribution='normal')
            indices = ((samples - samples.min()) / (samples.max() - samples.min()) * (n_total - 1)).astype(int)
        elif distribution == 'lognormal':
            samples = self._random(n_points, distribution='lognormal')
            indices = ((samples - samples.min()) / (samples.max() - samples.min()) * (n_total - 1)).astype(int)
        elif callable(distribution):
            samples = distribution(n_points)
            indices = ((samples - samples.min()) / (samples.max() - samples.min()) * (n_total - 1)).astype(int)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        sampled_values = flat_field[indices]
        integral = volume * self._mean(sampled_values)
        return self._to_scalar(integral)

    def _quasi_monte_carlo_field(self, field, n_points, volume):
        flat_field = self._flatten(field)
        n_total = len(flat_field)
        sobol_indices = self._sobol_sequence(n_points, 1)
        if self.backend_name == 'torch':
            import torch
            indices = (sobol_indices * (n_total - 1)).to(torch.int64)
        else:
            indices = (sobol_indices * (n_total - 1)).astype(int)
        if len(indices.shape) > 1:
            indices = indices[:, 0]
        sampled_values = flat_field[indices]
        integral = volume * self._mean(sampled_values)
        return self._to_scalar(integral)

    # ============================================================
    # BACKEND-AGNOSTIC HELPERS (NO RECURSION)
    # ============================================================

    def _sum(self, tensor, axis=None):
        """Sum with backend-agnostic interface."""
        if self.backend_name == 'torch':
            import torch
            if axis is not None:
                return torch.sum(tensor, dim=axis)
            else:
                return torch.sum(tensor)
        elif self.backend_name == 'jax':
            import jax.numpy as jnp
            if axis is not None:
                return jnp.sum(tensor, axis=axis)
            else:
                return jnp.sum(tensor)
        else:
            if axis is not None:
                return self.backend.sum(tensor, axis=axis)
            else:
                return self.backend.sum(tensor)

    def _mean(self, tensor, axis=None):
        """Mean with backend-agnostic interface."""
        if self.backend_name == 'torch':
            import torch
            if axis is not None:
                return torch.mean(tensor, dim=axis)
            else:
                return torch.mean(tensor)
        elif self.backend_name == 'jax':
            import jax.numpy as jnp
            if axis is not None:
                return jnp.mean(tensor, axis=axis)
            else:
                return jnp.mean(tensor)
        else:
            if axis is not None:
                return self.backend.mean(tensor, axis=axis)
            else:
                return self.backend.mean(tensor)

    # ============================================================
    # RANDOM NUMBER GENERATION
    # ============================================================

    def _random(self, n, distribution='uniform'):
        if self.backend_name == 'numpy':
            if distribution == 'uniform':
                return np.random.rand(n)
            elif distribution == 'normal':
                return np.random.randn(n)
            elif distribution == 'lognormal':
                return np.random.lognormal(size=n)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        elif self.backend_name == 'torch':
            import torch
            if distribution == 'uniform':
                return torch.rand(n)
            elif distribution == 'normal':
                return torch.randn(n)
            elif distribution == 'lognormal':
                return torch.exp(torch.randn(n))
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        elif self.backend_name == 'jax':
            import jax.numpy as jnp
            import jax.random as jr
            key = jr.PRNGKey(0)
            if distribution == 'uniform':
                return jr.uniform(key, shape=(n,))
            elif distribution == 'normal':
                return jr.normal(key, shape=(n,))
            elif distribution == 'lognormal':
                return jnp.exp(jr.normal(key, shape=(n,)))
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def _randint(self, low, high, n):
        if self.backend_name == 'numpy':
            return np.random.randint(low, high, n)
        elif self.backend_name == 'torch':
            import torch
            return torch.randint(low, high, (n,))
        elif self.backend_name == 'jax':
            import jax.numpy as jnp
            import jax.random as jr
            key = jr.PRNGKey(1)
            return jr.randint(key, (n,), low, high)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def _sobol_sequence(self, n, dim):
        try:
            from scipy.stats import qmc
            import math
            n_power2 = 2 ** int(math.ceil(math.log2(n)))
            if n_power2 != n:
                import warnings
                warnings.warn(f"Sobol sequence: rounding n={n} to nearest power of 2: {n_power2}")
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n_power2)
            if n_power2 > n:
                samples = samples[:n]
            if self.backend_name == 'torch':
                import torch
                return torch.tensor(samples, dtype=torch.float64)
            elif self.backend_name == 'jax':
                import jax.numpy as jnp
                return jnp.array(samples)
            else:
                return samples
        except ImportError:
            print("Warning: scipy not available. Falling back to random.")
            return self._random(n, distribution='uniform')

    def _flatten(self, field):
        if self.backend_name == 'numpy':
            return field.flatten()
        elif self.backend_name == 'torch':
            return field.flatten()
        elif self.backend_name == 'jax':
            return field.flatten()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def available_methods(self):
        return ['trapezoidal', 'simpson', 'monte_carlo', 'quasi_monte_carlo']

    def available_distributions(self):
        return ['uniform', 'normal', 'lognormal', 'custom']

    def future_methods(self):
        return ['gaussian', 'midpoint', 'adaptive']

    def _to_scalar(self, value):
        if self.backend_name == 'numpy':
            return float(value)
        elif self.backend_name == 'torch':
            return value.item() if hasattr(value, 'item') else float(value)
        elif self.backend_name == 'jax':
            return float(value)
        return value
