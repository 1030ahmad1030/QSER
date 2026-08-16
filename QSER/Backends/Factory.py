"""
Backend Factory.

Creates and returns the requested backend.
"""

from .NumPyBackend import NumPyBackend
from .TorchBackend import TorchBackend
from .JAXBackend import JAXBackend
from .OpenFOAMBackend import OpenFOAMBackend

def get_backend(name='numpy', **kwargs):
    """
    Get a backend instance.

    Parameters:
        name: str, backend name ('numpy', 'torch', 'jax', 'openfoam')
        **kwargs: Additional arguments for the backend

    Returns:
        Backend instance
    """
    if name == 'numpy':
        return NumPyBackend()
    elif name == 'torch':
        return TorchBackend(**kwargs)
    elif name == 'jax':
        return JAXBackend()
    elif name == 'openfoam':
        return OpenFOAMBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")
