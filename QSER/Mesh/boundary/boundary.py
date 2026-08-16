"""
Boundary condition engine for QSER meshes.

Supports 1D, 2D, and 3D meshes with patch-based BC assignment.
"""

import numpy as np
from QSER.Backends import get_backend


class Boundary:
    """
    Boundary condition engine with patch-based assignment.

    Supports:
        - Dirichlet: Fixed value
        - Neumann: Fixed gradient
        - Robin: Mixed condition
        - Periodic: Periodic
        - Wall: Wall condition (no-slip, free-slip, moving wall)
        - Source: Source term
        - Custom: User-defined condition

    Parameters:
        mesh: Mesh object
        backend: str, 'numpy', 'jax', 'torch' (default: 'numpy')
    """

    def __init__(self, mesh, backend='numpy'):
        self.mesh = mesh
        self.dim = mesh.dim
        self.backend = get_backend(backend)
        self.patches = {}
        self.conditions = {}
        self._condition_registry = {}

    # ============================================================
    # PATCH MANAGEMENT
    # ============================================================

    def add_patch(self, name, indices):
        self.patches[name] = indices
        return self

    def remove_patch(self, name):
        if name in self.patches:
            del self.patches[name]
        if name in self.conditions:
            del self.conditions[name]
        return self

    def get_patches(self):
        return self.patches

    def get_conditions(self):
        return self.conditions

    def clear(self):
        self.patches = {}
        self.conditions = {}
        return self

    # ============================================================
    # BOUNDARY CONDITION SETTERS
    # ============================================================

    def set_dirichlet(self, patch, value):
        self.conditions[patch] = {'type': 'dirichlet', 'value': value}
        return self

    def set_neumann(self, patch, value):
        self.conditions[patch] = {'type': 'neumann', 'value': value}
        return self

    def set_robin(self, patch, a, b, c):
        self.conditions[patch] = {'type': 'robin', 'a': a, 'b': b, 'c': c}
        return self

    def set_periodic(self, patch, match_patch=None):
        self.conditions[patch] = {'type': 'periodic', 'match': match_patch}
        return self

    def set_wall(self, patch, wall_type='no-slip'):
        self.conditions[patch] = {'type': 'wall', 'wall_type': wall_type}
        return self

    def set_source(self, patch, value):
        self.conditions[patch] = {'type': 'source', 'value': value}
        return self

    def set_custom(self, patch, func):
        self.conditions[patch] = {'type': 'custom', 'func': func}
        return self

    def register_bc_type(self, name, handler_class):
        self._condition_registry[name] = handler_class
        return self

    # ============================================================
    # APPLICATION
    # ============================================================

    def apply(self, field, time=0.0):
        field_copy = self.backend.array(field)
        if not self.backend.is_backend_array(field_copy):
            field_copy = self.backend.array(field_copy)

        for patch_name, indices in self.patches.items():
            if patch_name in self.conditions:
                cond = self.conditions[patch_name]
                field_copy = self._apply_condition(field_copy, indices, cond, time)

        return field_copy

    def _apply_condition(self, field, indices, condition, time):
        cond_type = condition['type']

        if cond_type == 'dirichlet':
            return self._apply_dirichlet(field, indices, condition['value'], time)
        elif cond_type == 'neumann':
            return self._apply_neumann(field, indices, condition['value'], time)
        elif cond_type == 'robin':
            return self._apply_robin(field, indices, condition['a'], condition['b'], condition['c'], time)
        elif cond_type == 'periodic':
            return self._apply_periodic(field, indices, condition.get('match'), time)
        elif cond_type == 'wall':
            return self._apply_wall(field, indices, condition['wall_type'], time)
        elif cond_type == 'source':
            return self._apply_source(field, indices, condition['value'], time)
        elif cond_type == 'custom':
            return condition['func'](field, self.mesh, indices, time)
        else:
            if cond_type in self._condition_registry:
                handler = self._condition_registry[cond_type]
                return handler().apply(field, indices, condition)
            else:
                raise ValueError(f"Unknown boundary condition type: {cond_type}")

    # ============================================================
    # 1D IMPLEMENTATIONS
    # ============================================================

    def _apply_dirichlet(self, field, indices, value, time):
        if callable(value):
            value = value(time, self.mesh)
        if not self.backend.is_backend_array(value):
            value = self.backend.array(value)

        for idx in indices:
            if self.dim == 1:
                field = self.backend.set_item(field, idx, value)
            else:
                # 2D/3D: idx is tuple
                field = self.backend.set_item(field, idx, value)
        return field

    def _apply_neumann_1d(self, field, idx, value):
        """Neumann for 1D."""
        if idx == 0:
            neighbor = 1
            dx = self.mesh.dx[0]
            return self.backend.set_item(field, idx, field[neighbor] - value * dx)
        elif idx == self.mesh.n_cells - 1:
            neighbor = self.mesh.n_cells - 2
            dx = self.mesh.dx[-1]
            return self.backend.set_item(field, idx, field[neighbor] + value * dx)
        else:
            left = idx - 1
            right = idx + 1
            dx = (self.mesh.dx[left] + self.mesh.dx[idx]) / 2
            return self.backend.set_item(field, idx, (field[left] + field[right]) / 2 - value * dx / 2)

    def _apply_neumann_2d(self, field, idx, value, axis):
        """Neumann for 2D boundary."""
        i, j = idx
        nx, ny = self.mesh.nx, self.mesh.ny
        
        if axis == 'x':
            # Left or right boundary
            if i == 0:
                neighbor = (1, j)
                dx = self.mesh.dx[0]
                val = field[1, j] - value * dx
            elif i == nx - 1:
                neighbor = (nx - 2, j)
                dx = self.mesh.dx[-1]
                val = field[nx - 2, j] + value * dx
            else:
                # Should not happen for boundary
                return field
        elif axis == 'y':
            # Bottom or top boundary
            if j == 0:
                neighbor = (i, 1)
                dy = self.mesh.dy[0]
                val = field[i, 1] - value * dy
            elif j == ny - 1:
                neighbor = (i, ny - 2)
                dy = self.mesh.dy[-1]
                val = field[i, ny - 2] + value * dy
            else:
                return field
        else:
            return field
        
        return self.backend.set_item(field, idx, val)

    def _apply_neumann(self, field, indices, value, time):
        if callable(value):
            value = value(time, self.mesh)

        for idx in indices:
            if self.dim == 1:
                field = self._apply_neumann_1d(field, idx, value)
            elif self.dim == 2:
                # Determine if this is an x-boundary or y-boundary
                i, j = idx
                if i == 0 or i == self.mesh.nx - 1:
                    field = self._apply_neumann_2d(field, idx, value, 'x')
                elif j == 0 or j == self.mesh.ny - 1:
                    field = self._apply_neumann_2d(field, idx, value, 'y')
                else:
                    # Interior point (should not happen for boundary)
                    pass
            else:
                # 3D not yet implemented
                raise NotImplementedError("3D Neumann BC not yet implemented")

        return field

    def _apply_robin_1d(self, field, idx, a, b, c):
        """Robin for 1D."""
        if idx == 0:
            neighbor = 1
            dx = self.mesh.dx[0]
            f_new = (c * dx + b * field[neighbor]) / (a * dx + b)
        elif idx == self.mesh.n_cells - 1:
            neighbor = self.mesh.n_cells - 2
            dx = self.mesh.dx[-1]
            f_new = (c * dx + b * field[neighbor]) / (a * dx + b)
        else:
            left = idx - 1
            right = idx + 1
            dx = (self.mesh.dx[left] + self.mesh.dx[idx]) / 2
            f_new = (c * dx + b * (field[left] + field[right]) / 2) / (a * dx + b)
        return self.backend.set_item(field, idx, f_new)

    def _apply_robin(self, field, indices, a, b, c, time):
        if callable(a):
            a = a(time, self.mesh)
        if callable(b):
            b = b(time, self.mesh)
        if callable(c):
            c = c(time, self.mesh)

        for idx in indices:
            if self.dim == 1:
                field = self._apply_robin_1d(field, idx, a, b, c)
            else:
                # 2D Robin not yet fully implemented
                # For now, apply to each index as Dirichlet-like
                if callable(a):
                    a_val = a
                else:
                    a_val = a
                # Simple fallback: set to c/a
                field = self.backend.set_item(field, idx, c / a_val)
        return field

    def _apply_periodic(self, field, indices, match_patch, time):
        """Apply Periodic BC."""
        if self.dim == 1:
            left_val = field[0]
            right_val = field[-1]
            avg_val = (left_val + right_val) / 2.0
            field = self.backend.set_item(field, 0, avg_val)
            field = self.backend.set_item(field, -1, avg_val)
        else:
            # 2D Periodic: match opposite boundaries
            # Left ↔ Right
            for j in range(self.mesh.ny):
                avg_val = (field[0, j] + field[-1, j]) / 2.0
                field = self.backend.set_item(field, (0, j), avg_val)
                field = self.backend.set_item(field, (-1, j), avg_val)
            # Bottom ↔ Top
            for i in range(self.mesh.nx):
                avg_val = (field[i, 0] + field[i, -1]) / 2.0
                field = self.backend.set_item(field, (i, 0), avg_val)
                field = self.backend.set_item(field, (i, -1), avg_val)
        return field

    def _apply_wall(self, field, indices, wall_type, time):
        for idx in indices:
            if wall_type == 'no-slip':
                field = self.backend.set_item(field, idx, 0.0)
            elif wall_type == 'free-slip':
                if self.dim == 1:
                    if idx == 0:
                        neighbor = 1
                        field = self.backend.set_item(field, idx, field[neighbor])
                    elif idx == self.mesh.n_cells - 1:
                        neighbor = self.mesh.n_cells - 2
                        field = self.backend.set_item(field, idx, field[neighbor])
                    else:
                        left = idx - 1
                        right = idx + 1
                        field = self.backend.set_item(field, idx, (field[left] + field[right]) / 2)
                else:
                    # 2D free-slip: copy neighbor value
                    i, j = idx
                    if i == 0:
                        field = self.backend.set_item(field, idx, field[1, j])
                    elif i == self.mesh.nx - 1:
                        field = self.backend.set_item(field, idx, field[self.mesh.nx - 2, j])
                    elif j == 0:
                        field = self.backend.set_item(field, idx, field[i, 1])
                    elif j == self.mesh.ny - 1:
                        field = self.backend.set_item(field, idx, field[i, self.mesh.ny - 2])
            elif wall_type == 'moving_wall':
                field = self.backend.set_item(field, idx, 0.0)
            else:
                raise ValueError(f"Unknown wall type: {wall_type}")
        return field

    def _apply_source(self, field, indices, value, time):
        if callable(value):
            value = value(time, self.mesh)

        for idx in indices:
            current = field[idx]
            field = self.backend.set_item(field, idx, current + value)
        return field
