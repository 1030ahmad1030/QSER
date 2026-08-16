"""
Test: QSER Import and Basic Structure
"""

def test_import():
    """Test that QSER can be imported."""
    try:
        import QSER
        print(f"✅ QSER version {QSER.__version__} imported successfully")
        print(f"   Author: {QSER.__author__}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_qser_class():
    """Test that the QSER class can be instantiated."""
    try:
        from QSER import QSER
        # Create a dummy mesh (placeholder)
        class DummyMesh:
            n_cells = 100
            def __class__(self):
                return type('DummyMesh', (), {})
        
        mesh = DummyMesh()
        solver = QSER(mesh, method='FVM', backend='NumPy')
        print("✅ QSER class instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ QSER class test failed: {e}")
        return False

def test_backend_import():
    """Test that backends can be imported."""
    try:
        from QSER.Backends import NumPyBackend
        backend = NumPyBackend()
        print(f"✅ NumPy backend imported: {backend.name} v{backend.version}")
        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_torch_backend():
    """Test PyTorch backend if available."""
    try:
        from QSER.Backends import TorchBackend
        backend = TorchBackend()
        print(f"✅ Torch backend imported: {backend.name} v{backend.version}")
        if backend.is_gpu_available:
            print(f"   GPU available: {backend.device}")
        return True
    except Exception as e:
        print(f"⚠️ Torch backend not available: {e}")
        return True  # Not a failure if PyTorch is not installed

def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("QSER Phase 0 Tests")
    print("=" * 60)
    
    success = True
    success &= test_import()
    success &= test_qser_class()
    success &= test_backend_import()
    success &= test_torch_backend()
    
    print("=" * 60)
    if success:
        print("✅ All Phase 0 tests passed!")
    else:
        print("❌ Some Phase 0 tests failed.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    run_all_tests()
