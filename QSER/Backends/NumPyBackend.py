"""
NumPy Backend Implementation.

Default backend for QSER. Always available.
"""

import numpy as np
from .Base import Backend

class NumPyBackend(Backend):
    """NumPy backend implementation."""

    def __init__(self):
        super().__init__()
        self.name = 'numpy'
        self.version = np.__version__
        self.is_gpu_available = False
        self.is_production_ready = False

    def array(self, data):
        return np.array(data)

    def zeros(self, shape):
        return np.zeros(shape)

    def ones(self, shape):
        return np.ones(shape)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)

    def meshgrid(self, x, y, indexing='ij'):
        return np.meshgrid(x, y, indexing=indexing)

    def meshgrid3(self, x, y, z, indexing='ij'):
        return np.meshgrid(x, y, z, indexing=indexing)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def zeros_like(self, array):
        return np.zeros_like(array)

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    # ============================================================
    # BACKEND ABSTRACTION METHODS
    # ============================================================

    def set_item(self, array, idx, value):
        array[idx] = value
        return array

    def set_slice(self, array, start, end, value):
        array[start:end] = value
        return array

    def set_item_slice(self, array, row_start, row_end, col_start, col_end, value):
        array[row_start:row_end, col_start:col_end] = value
        return array

    def set_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        # Convert negative indices to positive
        shape = array.shape
        x_start = shape[0] + x_start if x_start < 0 else x_start
        x_end = shape[0] + x_end if x_end < 0 else x_end
        y_start = shape[1] + y_start if y_start < 0 else y_start
        y_end = shape[1] + y_end if y_end < 0 else y_end
        z_start = shape[2] + z_start if z_start < 0 else z_start
        z_end = shape[2] + z_end if z_end < 0 else z_end

        # For single-element slices (e.g., -1, -1), adjust end to be start+1
        if x_start == x_end:
            x_end = x_start + 1
        if y_start == y_end:
            y_end = y_start + 1
        if z_start == z_end:
            z_end = z_start + 1

        slice_shape = (x_end - x_start, y_end - y_start, z_end - z_start)

        # Ensure value matches slice shape
        if np.array(value).shape != slice_shape:
            try:
                value = np.broadcast_to(value, slice_shape)
            except ValueError:
                # Try to expand dims
                val_shape = np.array(value).shape
                if len(val_shape) == 2 and slice_shape[1] == 1:
                    value = np.expand_dims(value, axis=1)
                elif len(val_shape) == 2 and slice_shape[2] == 1:
                    value = np.expand_dims(value, axis=2)
                elif len(val_shape) == 1 and slice_shape[0] == 1:
                    value = np.expand_dims(value, axis=0)
                else:
                    raise ValueError(f"Cannot broadcast shape {np.array(value).shape} to {slice_shape}")
                value = np.broadcast_to(value, slice_shape)

        array[x_start:x_end, y_start:y_end, z_start:z_end] = value
        return array

    def add_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        # Convert negative indices to positive
        shape = array.shape
        x_start = shape[0] + x_start if x_start < 0 else x_start
        x_end = shape[0] + x_end if x_end < 0 else x_end
        y_start = shape[1] + y_start if y_start < 0 else y_start
        y_end = shape[1] + y_end if y_end < 0 else y_end
        z_start = shape[2] + z_start if z_start < 0 else z_start
        z_end = shape[2] + z_end if z_end < 0 else z_end

        if x_start == x_end:
            x_end = x_start + 1
        if y_start == y_end:
            y_end = y_start + 1
        if z_start == z_end:
            z_end = z_start + 1

        slice_shape = (x_end - x_start, y_end - y_start, z_end - z_start)

        if np.array(value).shape != slice_shape:
            try:
                value = np.broadcast_to(value, slice_shape)
            except ValueError:
                val_shape = np.array(value).shape
                if len(val_shape) == 2 and slice_shape[1] == 1:
                    value = np.expand_dims(value, axis=1)
                elif len(val_shape) == 2 and slice_shape[2] == 1:
                    value = np.expand_dims(value, axis=2)
                elif len(val_shape) == 1 and slice_shape[0] == 1:
                    value = np.expand_dims(value, axis=0)
                else:
                    raise ValueError(f"Cannot broadcast shape {np.array(value).shape} to {slice_shape}")
                value = np.broadcast_to(value, slice_shape)

        array[x_start:x_end, y_start:y_end, z_start:z_end] += value
        return array

    def add_slice(self, array, start, end, value):
        array[start:end] += value
        return array

    def is_backend_array(self, obj):
        return isinstance(obj, np.ndarray)

    def to_numpy(self, array):
        return array
