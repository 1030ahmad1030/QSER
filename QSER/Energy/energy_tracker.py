"""
Energy Tracker for QSER Framework
=================================

Provides energy tracking with backend inheritance and flexible integration.

Quantities:
    - epsilon_S: 0.5 * ∫ S² dV
    - epsilon_E: 0.5 * ∫ E² dV
    - epsilon_R: 0.5 * ∫ R² dV
    - depsilon_S_dt: ∫ S * ∂S/∂t dV
    - depsilon_E_dt: ∫ E * ∂E/∂t dV
    - depsilon_R_dt: ∫ R * ∂R/∂t dV
    - J_SE: ∫ ∂E/∂t * (Ld S) dV
    - D_E: ∫ ∂E/∂t * (Ld E) dV
    - Phi_boundary: Boundary flux (user provides)

Energy Continuity:
    dε_E/dt + D_E + Phi_boundary = J_SE

Backend Inheritance:
    - If mesh is provided → inherits backend from mesh
    - If mesh is None → uses user-specified backend

dt Inheritance:
    - If mesh is provided and has dt → inherits dt
    - User can override by passing explicit dt
    - Stand-alone mode → user must provide dt
"""

import numpy as np
from QSER.Backends import get_backend
from .integration import IntegrationEngine


class EnergyTracker:
    """
    Energy tracker for QSER framework.
    
    Parameters:
        mesh: Mesh object (optional) — if provided, backend and dt are inherited
        backend: str (optional) — only used if mesh is None
        integration_method: str, 'trapezoidal', 'simpson', 'monte_carlo' (default: 'trapezoidal')
        n_samples: int, number of samples for Monte Carlo (default: 10000)
    
    Backend Inheritance:
        - If mesh is provided → inherits backend from mesh
        - If mesh is None → uses user-specified backend
    
    dt Inheritance:
        - If mesh is provided and has dt/time_step → inherits dt
        - User can override by passing explicit dt
        - Stand-alone mode → user must provide dt
    
    Usage:
        # Mesh-based (inherits backend and dt)
        mesh = Structured1D(nx=100, L=10.0, dt=0.001)
        tracker = EnergyTracker(mesh=mesh)
        passed, q = tracker.verify_energy_continuity(E, E_dot, Ld_E, Ld_S)
        
        # Mesh-based with override
        passed, q = tracker.verify_energy_continuity(E, E_dot, Ld_E, Ld_S, dt=0.002)
        
        # Stand-alone (user provides everything)
        tracker = EnergyTracker(backend='numpy')
        passed, q = tracker.verify_energy_continuity(E, E_dot, Ld_E, Ld_S, dt=0.001)
    """
    
    def __init__(self, mesh=None, backend='numpy', 
                 integration_method='trapezoidal', n_samples=10000):
        
        # ============================================================
        # 1. BACKEND INHERITANCE
        # ============================================================
        
        if mesh is not None:
            self.mesh = mesh
            self.backend = mesh.backend
            self.backend_name = mesh.backend_name
            self.dim = mesh.dim
            
            # Inherit dt from mesh if available
            self.dt = getattr(mesh, 'dt', None)
            self.time_step = getattr(mesh, 'time_step', None)
        else:
            self.mesh = None
            self.backend = get_backend(backend)
            self.backend_name = backend if isinstance(backend, str) else backend.name
            self.dim = None
            self.dt = None
            self.time_step = None
        
        # ============================================================
        # 2. INTEGRATION SETTINGS
        # ============================================================
        
        self.integration_method = integration_method
        self.n_samples = n_samples
        
        # Create IntegrationEngine with the same backend
        self.integrator = IntegrationEngine(backend=self.backend_name)
    
    # ============================================================
    # 3. INTERNAL INTEGRATION PIPELINE
    # ============================================================
    
    def _get_dt(self, dt=None):
        """Get dt from user, mesh, or raise error."""
        if dt is not None:
            return dt
        if self.dt is not None:
            return self.dt
        if self.time_step is not None:
            return self.time_step
        raise ValueError(
            "dt must be provided or mesh must have dt or time_step attribute"
        )
    
    def _get_spacings(self, dx=None, dy=None, dz=None):
        """Get spacings from user or mesh."""
        if self.mesh is not None:
            if dx is None and hasattr(self.mesh, 'dx'):
                dx = self.mesh.dx[0] if isinstance(self.mesh.dx, np.ndarray) else self.mesh.dx
            if dy is None and hasattr(self.mesh, 'dy'):
                dy = self.mesh.dy[0] if isinstance(self.mesh.dy, np.ndarray) else self.mesh.dy
            if dz is None and hasattr(self.mesh, 'dz'):
                dz = self.mesh.dz[0] if isinstance(self.mesh.dz, np.ndarray) else self.mesh.dz
        return dx, dy, dz
    
    def _integrate(self, field, volume=None, dx=None, dy=None, dz=None):
        """
        Internal integration pipeline.
        
        Steps:
            1. Convert field to backend (if needed)
            2. Get spacings (from mesh or user)
            3. Get volume (from mesh or user)
            4. Call IntegrationEngine
            5. Multiply by volume
            6. Return scalar
        """
        # Handle scalar fields (stand-alone mode with scalars)
        if np.isscalar(field):
            # For scalars, integration is just field * volume
            if volume is None:
                volume = 1.0
            return self._to_scalar(field * volume)
        
        # Step 1: Convert field to backend if it's NumPy
        if isinstance(field, np.ndarray) and self.backend_name != 'numpy':
            if self.backend_name == 'torch':
                import torch
                field = torch.tensor(field, dtype=torch.float64)
            elif self.backend_name == 'jax':
                import jax.numpy as jnp
                field = jnp.array(field)
        
        # Step 2: Get spacings (from mesh or user)
        dx, dy, dz = self._get_spacings(dx, dy, dz)
        
        # Step 3: Get volume
        if volume is None and self.mesh is not None:
            volume = self._get_volume()
        elif volume is None:
            volume = 1.0
        
        # Step 4: Integrate using IntegrationEngine
        if self.mesh is not None:
            # Mesh-based integration
            result = self.integrator.integrate_field(
                field,
                dx=dx,
                dy=dy,
                dz=dz,
                method=self.integration_method,
                n_points=self.n_samples
            )
        else:
            # Stand-alone integration
            result = self.integrator.integrate_field(
                field,
                dx=dx,
                dy=dy,
                dz=dz,
                method=self.integration_method,
                n_points=self.n_samples
            )
        
        # Step 5: Multiply by volume
        result = result * volume
        
        # Step 6: Return scalar
        return self._to_scalar(result)
    
    def _get_volume(self):
        """Get total volume from mesh."""
        if self.mesh is None:
            return None
        if self.dim == 1:
            return np.sum(self.mesh.cell_volumes)
        elif self.dim == 2:
            return np.sum(self.mesh.cell_volumes)
        elif self.dim == 3:
            return np.sum(self.mesh.cell_volumes)
        else:
            return 1.0
    
    def _to_scalar(self, value):
        """Convert backend array to scalar."""
        if self.backend_name == 'numpy':
            return float(value)
        elif self.backend_name == 'torch':
            return value.item() if hasattr(value, 'item') else float(value)
        elif self.backend_name == 'jax':
            return float(value)
        return value
    
    # ============================================================
    # 4. ENERGY QUANTITIES
    # ============================================================
    
    def epsilon(self, field, volume=None, dx=None, dy=None, dz=None):
        """ε = 0.5 * ∫ f² dV"""
        return 0.5 * self._integrate(field**2, volume, dx, dy, dz)
    
    def epsilon_S(self, S, volume=None, dx=None, dy=None, dz=None):
        """ε_S = 0.5 * ∫ S² dV"""
        return self.epsilon(S, volume, dx, dy, dz)
    
    def epsilon_E(self, E, volume=None, dx=None, dy=None, dz=None):
        """ε_E = 0.5 * ∫ E² dV"""
        return self.epsilon(E, volume, dx, dy, dz)
    
    def epsilon_R(self, R, volume=None, dx=None, dy=None, dz=None):
        """ε_R = 0.5 * ∫ R² dV"""
        return self.epsilon(R, volume, dx, dy, dz)
    
    # ============================================================
    # 5. ENERGY RATES
    # ============================================================
    
    def depsilon_dt(self, field, field_dot, volume=None, dx=None, dy=None, dz=None):
        """dε/dt = ∫ f * ∂f/∂t dV"""
        return self._integrate(field * field_dot, volume, dx, dy, dz)
    
    def depsilon_S_dt(self, S, S_dot, volume=None, dx=None, dy=None, dz=None):
        """dε_S/dt = ∫ S * ∂S/∂t dV"""
        return self.depsilon_dt(S, S_dot, volume, dx, dy, dz)
    
    def depsilon_E_dt(self, E, E_dot, volume=None, dx=None, dy=None, dz=None):
        """dε_E/dt = ∫ E * ∂E/∂t dV"""
        return self.depsilon_dt(E, E_dot, volume, dx, dy, dz)
    
    def depsilon_R_dt(self, R, R_dot, volume=None, dx=None, dy=None, dz=None):
        """dε_R/dt = ∫ R * ∂R/∂t dV"""
        return self.depsilon_dt(R, R_dot, volume, dx, dy, dz)
    
    # ============================================================
    # 6. ENERGY TRANSFER AND DISSIPATION
    # ============================================================
    
    def J_SE(self, E_dot, Ld_S, volume=None, dx=None, dy=None, dz=None):
        """J_SE = ∫ ∂E/∂t · (Ld S) dV"""
        return self._integrate(E_dot * Ld_S, volume, dx, dy, dz)
    
    def D_E(self, E_dot, Ld_E, volume=None, dx=None, dy=None, dz=None):
        """D_E = ∫ ∂E/∂t · (Ld E) dV"""
        return self._integrate(E_dot * Ld_E, volume, dx, dy, dz)
    
    # ============================================================
    # 7. ENERGY CONTINUITY VERIFICATION
    # ============================================================
    
    def verify_energy_continuity(self, E, E_dot, Ld_E, Ld_S, 
                                  dt=None, volume=None, Phi_boundary=0.0,
                                  dx=None, dy=None, dz=None, tol=1e-6):
        """
        Verify energy continuity: dε_E/dt + D_E + Phi_boundary = J_SE
        
        Args:
            E: Environment field
            E_dot: Time derivative of E
            Ld_E: Dissipative operator applied to E
            Ld_S: Dissipative operator applied to S
            dt: Time step (inherited from mesh if not provided)
            volume: Volume (optional)
            Phi_boundary: Boundary flux (user provides, default 0)
            dx, dy, dz: Spacings (inherited from mesh if not provided)
            tol: Tolerance for verification
            
        Returns:
            passed: bool
            quantities: dict with all terms
        """
        # Inherit dt from mesh if not provided
        dt = self._get_dt(dt)
        
        # Get spacings (from mesh or user)
        dx, dy, dz = self._get_spacings(dx, dy, dz)
        
        depsilon_E = self.depsilon_E_dt(E, E_dot, volume, dx, dy, dz)
        J_SE = self.J_SE(E_dot, Ld_S, volume, dx, dy, dz)
        D_E = self.D_E(E_dot, Ld_E, volume, dx, dy, dz)
        
        residual = depsilon_E + D_E + Phi_boundary - J_SE
        passed = abs(residual) < tol
        
        quantities = {
            'depsilon_E_dt': depsilon_E,
            'D_E': D_E,
            'Phi_boundary': Phi_boundary,
            'J_SE': J_SE,
            'residual': residual,
            'passed': passed
        }
        
        return passed, quantities
    
    # ============================================================
    # 8. UTILITY
    # ============================================================
    
    def available_methods(self):
        """Return available integration methods."""
        return self.integrator.available_methods()
    
    def set_method(self, method):
        """Set integration method."""
        if method not in self.available_methods():
            raise ValueError(f"Unknown method: {method}")
        self.integration_method = method
    
    def set_samples(self, n_samples):
        """Set number of Monte Carlo samples."""
        self.n_samples = n_samples
