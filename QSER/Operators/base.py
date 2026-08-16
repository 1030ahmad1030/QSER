"""
Base operator class for QSER.

All operators inherit from this class and have access to:
- mesh: Mesh object (optional)
- backend: Backend object
- gradient_method: str
"""

from QSER.Backends import get_backend


class Operator:
    """
    Base class for all QSER operators.

    Parameters:
        mesh: Mesh object (optional)
        backend: str or Backend object (default: 'numpy')
        gradient_method: str, gradient method (default: '5point')
    """

    def __init__(self, mesh=None, backend='numpy', gradient_method='5point'):
        self.mesh = mesh
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.gradient_method = gradient_method
        self.dim = mesh.dim if mesh else None

    def set_gradient_method(self, method):
        """Set the gradient method."""
        self.gradient_method = method

    def get_gradient_method(self):
        """Get the current gradient method."""
        return self.gradient_method

    def to_numpy(self, array):
        """Convert backend array to NumPy."""
        return self.backend.to_numpy(array)

    def to_backend(self, array):
        """Convert to backend array if needed."""
        if not self.backend.is_backend_array(array):
            return self.backend.array(array)
        return array

    def compute(self, *args, **kwargs):
        """Compute the operator."""
        raise NotImplementedError("Subclasses must implement compute()")
