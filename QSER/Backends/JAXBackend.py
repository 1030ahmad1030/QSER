"""
JAX Backend Implementation.

Optional backend for QSER. Provides GPU acceleration, autograd, and XLA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from .Base import Backend

class JAXBackend(Backend):
    """JAX backend implementation."""

    def __init__(self):
        super().__init__()
        self.name = 'jax'
        self.version = jax.__version__
        self.is_gpu_available = jax.devices()[0].platform == 'gpu'
        self.is_production_ready = False
        self.np = jnp

    def array(self, data):
        return jnp.array(data)

    def zeros(self, shape):
        return jnp.zeros(shape)

    def ones(self, shape):
        return jnp.ones(shape)

    def linspace(self, start, stop, num):
        return jnp.linspace(start, stop, num)

    def meshgrid(self, x, y, indexing='ij'):
        return jnp.meshgrid(x, y, indexing=indexing)

    def meshgrid3(self, x, y, z, indexing='ij'):
        return jnp.meshgrid(x, y, z, indexing=indexing)

    def stack(self, arrays, axis=0):
        return jnp.stack(arrays, axis=axis)

    def zeros_like(self, array):
        return jnp.zeros_like(array)

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b

    def sum(self, a, axis=None):
        return jnp.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return jnp.mean(a, axis=axis)

    def matmul(self, a, b):
        return jnp.matmul(a, b)

    def solve(self, A, b):
        return jnp.linalg.solve(A, b)

    # ============================================================
    # BACKEND ABSTRACTION METHODS
    # ============================================================

    def set_item(self, array, idx, value):
        return array.at[idx].set(value)

    def set_slice(self, array, start, end, value):
        return array.at[start:end].set(value)

    def set_item_slice(self, array, row_start, row_end, col_start, col_end, value):
        return array.at[row_start:row_end, col_start:col_end].set(value)

    def set_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        return array.at[x_start:x_end, y_start:y_end, z_start:z_end].set(value)

    def add_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        return array.at[x_start:x_end, y_start:y_end, z_start:z_end].add(value)

    def add_slice(self, array, start, end, value):
        return array.at[start:end].add(value)

    def is_backend_array(self, obj):
        return isinstance(obj, jnp.ndarray)

    def to_numpy(self, array):
        if isinstance(array, jnp.ndarray):
            return np.array(array)
        return array
