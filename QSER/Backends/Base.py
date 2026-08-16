"""
Abstract Backend Interface.

All backends (NumPy, PyTorch, JAX, OpenFOAM) must implement this interface.
"""

from abc import ABC, abstractmethod

class Backend(ABC):
    """Abstract Backend Interface."""

    def __init__(self):
        self.name = 'base'
        self.is_gpu_available = False
        self.is_production_ready = False
        self.version = '0.0.0'

    @abstractmethod
    def array(self, data):
        """Convert data to backend array."""
        pass

    @abstractmethod
    def zeros(self, shape):
        """Create zeros array."""
        pass

    @abstractmethod
    def ones(self, shape):
        """Create ones array."""
        pass

    @abstractmethod
    def linspace(self, start, stop, num):
        """Create linearly spaced array."""
        pass

    @abstractmethod
    def meshgrid(self, x, y, indexing='ij'):
        """Create 2D meshgrid."""
        pass

    @abstractmethod
    def meshgrid3(self, x, y, z, indexing='ij'):
        """Create 3D meshgrid."""
        pass

    @abstractmethod
    def stack(self, arrays, axis=0):
        """Stack arrays."""
        pass

    @abstractmethod
    def zeros_like(self, array):
        """Create zeros array with same shape as input."""
        pass

    @abstractmethod
    def add(self, a, b):
        """Element-wise addition."""
        pass

    @abstractmethod
    def subtract(self, a, b):
        """Element-wise subtraction."""
        pass

    @abstractmethod
    def multiply(self, a, b):
        """Element-wise multiplication."""
        pass

    @abstractmethod
    def divide(self, a, b):
        """Element-wise division."""
        pass

    @abstractmethod
    def sum(self, a, axis=None):
        """Sum of array elements."""
        pass

    @abstractmethod
    def mean(self, a, axis=None):
        """Mean of array elements."""
        pass

    @abstractmethod
    def matmul(self, a, b):
        """Matrix multiplication."""
        pass

    @abstractmethod
    def solve(self, A, b):
        """Solve linear system Ax = b."""
        pass

    # ============================================================
    # BACKEND ABSTRACTION METHODS
    # ============================================================

    @abstractmethod
    def set_item(self, array, idx, value):
        """Set item in array."""
        pass

    @abstractmethod
    def set_slice(self, array, start, end, value):
        """Set 1D slice in array."""
        pass

    @abstractmethod
    def set_item_slice(self, array, row_start, row_end, col_start, col_end, value):
        """Set 2D slice in array."""
        pass

    @abstractmethod
    def set_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        """Set 3D slice in array."""
        pass

    @abstractmethod
    def add_slice_3d(self, array, x_start, x_end, y_start, y_end, z_start, z_end, value):
        """Add to 3D slice in array."""
        pass

    @abstractmethod
    def add_slice(self, array, start, end, value):
        """Add to 1D slice in array."""
        pass

    @abstractmethod
    def is_backend_array(self, obj):
        """Check if object is a backend array."""
        pass

    @abstractmethod
    def to_numpy(self, array):
        """Convert backend array to NumPy."""
        pass
