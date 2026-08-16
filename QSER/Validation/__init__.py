"""
QSER Validation Module.

Provides:
    1. Energy conservation in S
    2. Zero initial condition of E
    3. Exact reconstruction (R = S - E)
    4. Infinite memory of S (tau_u scaling)
"""

from .qser_tests import validate_qser

__all__ = ['validate_qser']
