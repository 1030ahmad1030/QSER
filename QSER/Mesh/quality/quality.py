"""
Mesh quality metrics for QSER meshes.

Provides industry-standard mesh quality metrics:
- Orthogonality: Face orthogonality (Fluent/OpenFOAM)
- Skewness: Cell skewness (Fluent)
- Aspect Ratio: Cell aspect ratio (Fluent)
- Cell Volume: Cell volumes (OpenFOAM checkMesh)
"""

import numpy as np
from QSER.Backends import get_backend


class MeshQuality:
    """
    Mesh quality computation engine.

    Parameters:
        mesh: Mesh object (Structured1D, Structured2D, Structured3D)
        backend: str, 'numpy', 'jax', 'torch' (default: 'numpy')
    """

    def __init__(self, mesh, backend='numpy'):
        self.mesh = mesh
        self.dim = mesh.dim
        self.backend = get_backend(backend)
        self._metrics_cache = {}

    # ============================================================
    # 1D QUALITY METRICS
    # ============================================================

    def _orthogonality_1d(self):
        """Orthogonality for 1D: always 1.0 (faces are perpendicular)."""
        n = self.mesh.n_cells
        return np.ones(n)

    def _skewness_1d(self):
        """Skewness for 1D: always 0.0 (uniform grids) or small for non-uniform."""
        dx = self.mesh.dx
        dx_np = self.backend.to_numpy(dx)
        skewness = np.zeros_like(dx_np)
        # For non-uniform grids, compute skewness
        if len(dx_np) > 1:
            for i in range(1, len(dx_np) - 1):
                ideal = (dx_np[i-1] + dx_np[i]) / 2
                actual = dx_np[i]
                if ideal > 0:
                    skewness[i] = abs(actual - ideal) / ideal
        return skewness

    def _aspect_ratio_1d(self):
        """Aspect ratio for 1D: always 1.0 (no stretching)."""
        n = self.mesh.n_cells
        return np.ones(n)

    def _cell_volumes_1d(self):
        """Cell volumes for 1D: cell lengths."""
        return self.backend.to_numpy(self.mesh.cell_volumes)

    # ============================================================
    # 2D QUALITY METRICS
    # ============================================================

    def _orthogonality_2d(self):
        """Orthogonality for 2D: cosine of angle between face normal and cell-center line."""
        nx, ny = self.mesh.nx, self.mesh.ny
        orthogonality = np.ones((nx, ny))

        # Compute for x-faces
        X_cell = self.backend.to_numpy(self.mesh.X_cell)
        Y_cell = self.backend.to_numpy(self.mesh.Y_cell)
        dx = self.backend.to_numpy(self.mesh.dx)
        dy = self.backend.to_numpy(self.mesh.dy)

        # For structured grids, orthogonality is 1.0 (faces are aligned)
        # This is a placeholder for future non-orthogonal grids
        return orthogonality

    def _skewness_2d(self):
        """Skewness for 2D: deviation from ideal cell shape."""
        nx, ny = self.mesh.nx, self.mesh.ny
        skewness = np.zeros((nx, ny))

        dx = self.backend.to_numpy(self.mesh.dx)
        dy = self.backend.to_numpy(self.mesh.dy)

        for i in range(nx):
            for j in range(ny):
                ideal_area = dx[i] * dy[j]
                actual_area = dx[i] * dy[j]  # For structured grids, actual = ideal
                if ideal_area > 0:
                    skewness[i, j] = abs(actual_area - ideal_area) / ideal_area

        return skewness

    def _aspect_ratio_2d(self):
        """Aspect ratio for 2D: max edge / min edge."""
        nx, ny = self.mesh.nx, self.mesh.ny
        aspect_ratio = np.zeros((nx, ny))

        dx = self.backend.to_numpy(self.mesh.dx)
        dy = self.backend.to_numpy(self.mesh.dy)

        for i in range(nx):
            for j in range(ny):
                max_edge = max(dx[i], dy[j])
                min_edge = min(dx[i], dy[j])
                if min_edge > 0:
                    aspect_ratio[i, j] = max_edge / min_edge

        return aspect_ratio

    def _cell_volumes_2d(self):
        """Cell volumes for 2D: cell areas."""
        return self.backend.to_numpy(self.mesh.cell_volumes)

    # ============================================================
    # 3D QUALITY METRICS
    # ============================================================

    def _orthogonality_3d(self):
        """Orthogonality for 3D: always 1.0 for structured grids."""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        return np.ones((nx, ny, nz))

    def _skewness_3d(self):
        """Skewness for 3D: deviation from ideal cell shape."""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        skewness = np.zeros((nx, ny, nz))

        dx = self.backend.to_numpy(self.mesh.dx)
        dy = self.backend.to_numpy(self.mesh.dy)
        dz = self.backend.to_numpy(self.mesh.dz)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ideal_volume = dx[i] * dy[j] * dz[k]
                    actual_volume = dx[i] * dy[j] * dz[k]
                    if ideal_volume > 0:
                        skewness[i, j, k] = abs(actual_volume - ideal_volume) / ideal_volume

        return skewness

    def _aspect_ratio_3d(self):
        """Aspect ratio for 3D: max edge / min edge."""
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        aspect_ratio = np.zeros((nx, ny, nz))

        dx = self.backend.to_numpy(self.mesh.dx)
        dy = self.backend.to_numpy(self.mesh.dy)
        dz = self.backend.to_numpy(self.mesh.dz)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    max_edge = max(dx[i], dy[j], dz[k])
                    min_edge = min(dx[i], dy[j], dz[k])
                    if min_edge > 0:
                        aspect_ratio[i, j, k] = max_edge / min_edge

        return aspect_ratio

    def _cell_volumes_3d(self):
        """Cell volumes for 3D: cell volumes."""
        return self.backend.to_numpy(self.mesh.cell_volumes)

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def compute_orthogonality(self):
        """Compute orthogonality metric."""
        if self.dim == 1:
            return self._orthogonality_1d()
        elif self.dim == 2:
            return self._orthogonality_2d()
        elif self.dim == 3:
            return self._orthogonality_3d()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def compute_skewness(self):
        """Compute skewness metric."""
        if self.dim == 1:
            return self._skewness_1d()
        elif self.dim == 2:
            return self._skewness_2d()
        elif self.dim == 3:
            return self._skewness_3d()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def compute_aspect_ratio(self):
        """Compute aspect ratio metric."""
        if self.dim == 1:
            return self._aspect_ratio_1d()
        elif self.dim == 2:
            return self._aspect_ratio_2d()
        elif self.dim == 3:
            return self._aspect_ratio_3d()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def compute_cell_volumes(self):
        """Compute cell volumes."""
        if self.dim == 1:
            return self._cell_volumes_1d()
        elif self.dim == 2:
            return self._cell_volumes_2d()
        elif self.dim == 3:
            return self._cell_volumes_3d()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

    def report(self):
        """
        Generate a full quality report.

        Returns:
            dict: Quality metrics with statistics and recommendations.
        """
        # Compute all metrics
        orthogonality = self.compute_orthogonality()
        skewness = self.compute_skewness()
        aspect_ratio = self.compute_aspect_ratio()
        volumes = self.compute_cell_volumes()

        # Convert to NumPy for statistics
        o_np = np.array(orthogonality).flatten()
        s_np = np.array(skewness).flatten()
        a_np = np.array(aspect_ratio).flatten()
        v_np = np.array(volumes).flatten()

        # Check for negative volumes
        has_negative_volume = np.any(v_np < 0)
        is_valid = not has_negative_volume

        # Recommendations
        max_skewness = np.max(s_np)
        max_aspect_ratio = np.max(a_np)
        min_orthogonality = np.min(o_np)

        if max_skewness < 0.1 and max_aspect_ratio < 2 and min_orthogonality > 0.9:
            recommendation = "Excellent quality"
        elif max_skewness < 0.5 and max_aspect_ratio < 10 and min_orthogonality > 0.5:
            recommendation = "Good quality"
        elif max_skewness < 0.85 and max_aspect_ratio < 100 and min_orthogonality > 0.1:
            recommendation = "Acceptable quality"
        else:
            recommendation = "Poor quality - consider refining the mesh"

        report = {
            'orthogonality': {
                'min': float(np.min(o_np)),
                'max': float(np.max(o_np)),
                'avg': float(np.mean(o_np)),
                'std': float(np.std(o_np))
            },
            'skewness': {
                'min': float(np.min(s_np)),
                'max': float(np.max(s_np)),
                'avg': float(np.mean(s_np)),
                'std': float(np.std(s_np))
            },
            'aspect_ratio': {
                'min': float(np.min(a_np)),
                'max': float(np.max(a_np)),
                'avg': float(np.mean(a_np)),
                'std': float(np.std(a_np))
            },
            'cell_volume': {
                'min': float(np.min(v_np)),
                'max': float(np.max(v_np)),
                'avg': float(np.mean(v_np)),
                'std': float(np.std(v_np))
            },
            'has_negative_volume': bool(has_negative_volume),
            'is_valid': bool(is_valid),
            'recommendation': recommendation
        }

        return report
