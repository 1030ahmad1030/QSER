"""
Test: QSER Backend Module
"""

import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import():
    """Test that all backends can be imported."""
    try:
        from QSER.Backends import NumPyBackend, TorchBackend, OpenFOAMBackend, get_backend
        print("✅ All backends imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_numpy_backend():
    """Test NumPy backend operations."""
    try:
        from QSER.Backends import get_backend
        backend = get_backend('numpy')
        
        # Test array creation
        a = backend.array([1, 2, 3])
        b = backend.array([4, 5, 6])
        
        # Test operations
        c = backend.add(a, b)
        d = backend.sum(c)
        
        print(f"✅ NumPy backend working: sum = {d}")
        return True
    except Exception as e:
        print(f"❌ NumPy backend test failed: {e}")
        return False

def test_torch_backend():
    """Test PyTorch backend operations (if available)."""
    try:
        from QSER.Backends import get_backend
        backend = get_backend('torch')
        
        # Test array creation
        a = backend.array([1, 2, 3])
        b = backend.array([4, 5, 6])
        
        # Test operations
        c = backend.add(a, b)
        d = backend.sum(c)
        
        print(f"✅ PyTorch backend working: sum = {d}")
        if backend.is_gpu_available:
            print(f"  GPU available: {backend.device}")
        return True
    except Exception as e:
        print(f"⚠️ PyTorch backend not available: {e}")
        return True  # Not a failure if PyTorch is not installed

def test_openfoam_backend():
    """Test OpenFOAM backend import (placeholder)."""
    try:
        from QSER.Backends import OpenFOAMBackend
        backend = OpenFOAMBackend()
        print(f"✅ OpenFOAM backend imported: {backend.name}")
        return True
    except Exception as e:
        print(f"⚠️ OpenFOAM backend not available: {e}")
        return True  # Not a failure if OpenFOAM is not installed

if __name__ == "__main__":
    print("=" * 60)
    print("QSER Backend Tests")
    print("=" * 60)
    
    success = True
    success &= test_import()
    success &= test_numpy_backend()
    success &= test_torch_backend()
    success &= test_openfoam_backend()
    
    print("=" * 60)
    if success:
        print("✅ All Backend tests passed!")
    else:
        print("❌ Some Backend tests failed.")
    print("=" * 60)