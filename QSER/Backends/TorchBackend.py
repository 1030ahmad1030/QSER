"""
PyTorch Backend Implementation.

Optional backend for QSER. Provides GPU acceleration and autograd.
"""

import torch
from .Base import Backend

class TorchBackend(Backend):
    """PyTorch backend implementation."""

    def __init__(self, device='auto'):
        super().__init__()
        self.name = 'torch'
        self.version = torch.__version__
        self.device = self._get_device(device)
        self.is_gpu_available = torch.cuda.is_available()
        self.is_production_ready = False

    def _get_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def array(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=torch.float64)
        return torch.tensor(data, device=self.device, dtype=torch.float64)

    def zeros(self, shape):
        return torch.zeros(shape, device=self.device, dtype=torch.float64)

    def ones(self, shape):
        return torch.ones(shape, device=self.device, dtype=torch.float64)

    def linspace(self, start, stop, num):
        return torch.linspace(start, stop, num, device=self.device, dtype=torch.float64)

    def meshgrid(self, x, y, indexing='ij'):
        return torch.meshgrid(x, y, indexing=indexing)

    def meshgrid3(self, x, y, z, indexing='ij'):
        return torch.meshgrid(x, y, z, indexing=indexing)

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def zeros_like(self, array):
        return torch.zeros_like(array)

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b

    def sum(self, a, axis=None):
        return torch.sum(a, dim=axis)

    def mean(self, a, axis=None):
        return torch.mean(a, dim=axis)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def solve(self, A, b):
        return torch.linalg.solve(A, b)

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
        array[x_start:x_end, y_start:y_end, z_start:z_end] = value
        return array

    def add_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        array[x_start:x_end, y_start:y_end, z_start:z_end] += value
        return array

    def add_slice(self, array, start, end, value):
        array[start:end] += value
        return array

    def is_backend_array(self, obj):
        return isinstance(obj, torch.Tensor)

    def to_numpy(self, array):
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return array
