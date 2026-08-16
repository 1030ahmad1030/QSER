"""
Geometry Module for QSER
========================

Provides mesh-independent geometry import and point sampling for PINNs.

Supported formats:
- STL (.stl)
- OBJ (.obj)
- PLY (.ply)
- OFF (.off)
- CSV (.csv)
- Excel (.xlsx, .xls)
- Direct points (from arrays)
- Built-in geometries (sphere, cube, cylinder)
- Mathematical domains (user-defined functions)

Usage:
    from qser.Geometry import Geometry
    
    # Import from file
    geo = Geometry.from_stl('model.stl')
    geo = Geometry.from_csv('points.csv')
    
    # Built-in
    geo = Geometry.sphere(radius=1.0)
    
    # Mathematical
    def inside(x, y, z):
        return (x/2)**2 + (y/3)**2 + (z/1.5)**2 < 1
    geo = Geometry.define(inside=inside)
    
    # Sample points
    interior = geo.sample_points(n_points=10000)
    boundary = geo.sample_boundary(n_points=2000)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable, Tuple, List, Any
import warnings

from .sampling import PointSampler
from .utils import (
    bounding_box,
    estimate_volume,
    contains_points,
    mesh_grid_from_geometry,
)


class Geometry:
    """
    Mesh-independent geometry for PINN training.
    
    Attributes:
        vertices (np.ndarray): (N, 3) array of vertex coordinates
        faces (np.ndarray): (M, 3) array of face indices (for surface meshes)
        normals (np.ndarray): (M, 3) array of face normals (optional)
        is_surface (bool): True if geometry has surface mesh
        bounds (tuple): (xmin, xmax, ymin, ymax, zmin, zmax)
        _inside_func (callable): For mathematical domains
    """
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.normals = None
        self.is_surface = False
        self.bounds = None
        self._inside_func = None
        self._source_file = None
        self._metadata = {}
        self._boundary_indices = None
    
    # ============================================================
    # CLASS METHODS: IMPORT FROM FILES
    # ============================================================
    
    @classmethod
    def from_stl(cls, filepath: Union[str, Path], **kwargs) -> "Geometry":
        """Import geometry from STL file."""
        try:
            from .handlers import STLHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        geo.vertices, geo.faces, geo.normals = STLHandler.read(filepath, **kwargs)
        geo.is_surface = True
        geo._source_file = str(filepath)
        geo._update_bounds()
        geo._metadata['format'] = 'stl'
        return geo
    
    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: str = 'z',
        boundary_col: Optional[str] = None,
        delimiter: str = ',',
        header: int = 0,
        **kwargs
    ) -> "Geometry":
        """Import point cloud from CSV file."""
        try:
            from .handlers import CSVHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        result = CSVHandler.read(
            filepath,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            boundary_col=boundary_col,
            delimiter=delimiter,
            header=header,
            **kwargs
        )
        geo.vertices = result[0]
        geo.faces = result[1]
        geo.normals = result[2]
        geo.is_surface = False
        geo._source_file = str(filepath)
        geo._update_bounds()
        geo._metadata['format'] = 'csv'
        if boundary_col is not None:
            geo._metadata['has_boundary_labels'] = True
        return geo
    
    @classmethod
    def from_excel(
        cls,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: str = 'z',
        boundary_col: Optional[str] = None,
        **kwargs
    ) -> "Geometry":
        """Import point cloud from Excel file (.xlsx, .xls)."""
        try:
            from .handlers import ExcelHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        result = ExcelHandler.read(
            filepath,
            sheet_name=sheet_name,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            boundary_col=boundary_col,
            **kwargs
        )
        geo.vertices = result[0]
        geo.faces = result[1]
        geo.normals = result[2]
        geo.is_surface = False
        geo._source_file = str(filepath)
        geo._update_bounds()
        geo._metadata['format'] = 'excel'
        if boundary_col is not None:
            geo._metadata['has_boundary_labels'] = True
        return geo
    
    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        boundary_indices: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Geometry":
        """Create geometry from point cloud."""
        try:
            from .handlers import PointsHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        result = PointsHandler.read(points, boundary_indices=boundary_indices, **kwargs)
        geo.vertices = result[0]
        geo.faces = result[1]
        geo.normals = result[2]
        geo.is_surface = False
        geo._update_bounds()
        geo._metadata['format'] = 'points'
        if boundary_indices is not None:
            geo._metadata['has_boundary_labels'] = True
            geo._boundary_indices = boundary_indices
        return geo
    
    @classmethod
    def from_dataframe(cls, df, x_col: str = 'x', y_col: str = 'y', z_col: str = 'z', **kwargs) -> "Geometry":
        """Import point cloud from pandas DataFrame."""
        points = df[[x_col, y_col, z_col]].values
        return cls.from_points(points, **kwargs)
    
    # ============================================================
    # BUILT-IN GEOMETRIES
    # ============================================================
    
    @classmethod
    def sphere(cls, center: Tuple[float, float, float] = (0, 0, 0), radius: float = 1.0, **kwargs) -> "Geometry":
        """Create a sphere geometry."""
        try:
            from .handlers import BuiltinHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        geo.vertices, geo.faces, geo.normals = BuiltinHandler.sphere(center, radius, **kwargs)
        geo.is_surface = True
        geo._update_bounds()
        geo._metadata['format'] = 'builtin'
        geo._metadata['shape'] = 'sphere'
        geo._metadata['center'] = center
        geo._metadata['radius'] = radius
        return geo
    
    @classmethod
    def cube(cls, center: Tuple[float, float, float] = (0, 0, 0), size: float = 2.0, **kwargs) -> "Geometry":
        """Create a cube geometry."""
        try:
            from .handlers import BuiltinHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        geo.vertices, geo.faces, geo.normals = BuiltinHandler.cube(center, size, **kwargs)
        geo.is_surface = True
        geo._update_bounds()
        geo._metadata['format'] = 'builtin'
        geo._metadata['shape'] = 'cube'
        geo._metadata['center'] = center
        geo._metadata['size'] = size
        return geo
    
    @classmethod
    def cylinder(cls, center: Tuple[float, float, float] = (0, 0, 0), radius: float = 0.5, height: float = 2.0, **kwargs) -> "Geometry":
        """Create a cylinder geometry."""
        try:
            from .handlers import BuiltinHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        geo.vertices, geo.faces, geo.normals = BuiltinHandler.cylinder(center, radius, height, **kwargs)
        geo.is_surface = True
        geo._update_bounds()
        geo._metadata['format'] = 'builtin'
        geo._metadata['shape'] = 'cylinder'
        return geo
    
    @classmethod
    def define(cls, inside_func: Callable, bounds: Optional[Tuple] = None, **kwargs) -> "Geometry":
        """
        Define geometry from mathematical function.
        
        Args:
            inside_func: Function returning True for points inside domain
            bounds: Optional (xmin, xmax, ymin, ymax, zmin, zmax)
            
        Returns:
            Geometry object
        """
        geo = cls()
        geo._inside_func = inside_func
        geo.is_surface = False
        if bounds is not None:
            geo.bounds = tuple(bounds)
        geo._metadata['format'] = 'mathematical'
        return geo
    
    # ============================================================
    # POINT SAMPLING
    # ============================================================
    
    def sample_points(
        self,
        n_points: int,
        method: str = 'random',
        seed: Optional[int] = None,
        bounds: Optional[Tuple] = None,
        max_attempts: int = 10000,
        **kwargs
    ) -> np.ndarray:
        """
        Sample interior points from the geometry.
        
        Args:
            n_points: Number of points to sample
            method: 'random', 'latin_hypercube', 'sobol', 'adaptive'
            seed: Random seed for reproducibility
            bounds: Optional bounds for sampling
            max_attempts: Maximum attempts for rejection sampling
            
        Returns:
            (n_points, 3) array of point coordinates
        """
        # Determine bounds
        if bounds is None:
            if self.bounds is not None:
                bounds = self.bounds
            else:
                raise ValueError("Bounds not set. Provide bounds or use geometry with bounds.")
        
        # For mathematical domains, use rejection sampling
        if self._inside_func is not None:
            points = PointSampler.sample_inside_function(
                self._inside_func,
                n_points,
                bounds=bounds,
                method=method,
                seed=seed,
                max_attempts=max_attempts,
                **kwargs
            )
            return points
        
        # For geometry with vertices, use sampling within bounds
        if self.vertices is not None:
            return PointSampler.sample_within_bounds(
                n_points,
                bounds=bounds,
                method=method,
                seed=seed,
                **kwargs
            )
        
        raise ValueError("Cannot sample: no geometry data or inside function.")
    
    def sample_boundary(
        self,
        n_points: int,
        method: str = 'uniform',
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample boundary points from the geometry.
        
        Args:
            n_points: Number of boundary points to sample
            method: 'uniform', 'random', 'adaptive'
            seed: Random seed
            
        Returns:
            (n_points, 3) array of point coordinates
        """
        if self.faces is not None and self.vertices is not None:
            return PointSampler.sample_boundary_mesh(
                self.vertices,
                self.faces,
                n_points,
                method=method,
                seed=seed,
                **kwargs
            )
        elif self.vertices is not None and not self.is_surface:
            # Point cloud: use existing points or edge detection
            warnings.warn("Point cloud boundary sampling may be approximate.")
            return PointSampler.sample_boundary_points(
                self.vertices,
                n_points,
                method=method,
                seed=seed,
                **kwargs
            )
        else:
            raise ValueError("Cannot sample boundary: no surface mesh or boundary points.")
    
    # ============================================================
    # GEOMETRY INFORMATION
    # ============================================================
    
    def _update_bounds(self):
        """Update bounding box from vertices."""
        if self.vertices is not None and len(self.vertices) > 0:
            self.bounds = bounding_box(self.vertices)
    
    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """Return bounding box: (xmin, xmax, ymin, ymax, zmin, zmax)."""
        if self.bounds is not None:
            return tuple(self.bounds)
        if self.vertices is not None:
            self._update_bounds()
            return tuple(self.bounds)
        raise ValueError("Bounding box not available.")
    
    def volume(self) -> float:
        """Estimate volume of the geometry."""
        if self.vertices is not None and self.faces is not None:
            return estimate_volume(self.vertices, self.faces)
        if self.bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        raise ValueError("Volume cannot be estimated.")
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside the geometry."""
        if self._inside_func is not None:
            return bool(self._inside_func(point[0], point[1], point[2]))
        if self.vertices is not None and self.faces is not None:
            return contains_points(point, self.vertices, self.faces)
        raise ValueError("Contains point check not available.")
    
    def get_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vertices, faces) for visualization."""
        return self.vertices, self.faces
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices) if self.vertices is not None else 0
    
    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces) if self.faces is not None else 0
    
    # ============================================================
    # EXPORT
    # ============================================================
    
    def export_points(self, filepath: Union[str, Path], points: np.ndarray, header: str = "x,y,z") -> None:
        """Export sampled points to CSV."""
        np.savetxt(filepath, points, delimiter=',', header=header, comments='')
    
    # ============================================================
    # VISUALIZATION (Optional)
    # ============================================================
    
    def plot(self, ax=None, show_vertices=True, show_faces=True, **kwargs):
        """Plot the geometry (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            warnings.warn("matplotlib not installed. Install for plotting.")
            return
        
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        if self.vertices is not None and show_vertices:
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], s=1, alpha=0.5, **kwargs)
        
        if self.faces is not None and show_faces:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh = Poly3DCollection(self.vertices[self.faces], alpha=0.2, edgecolor='k', linewidths=0.1)
            ax.add_collection3d(mesh)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Geometry ({self._metadata.get("format", "unknown")})')
        return ax


    @classmethod
    def from_cad(cls, filepath: Union[str, Path], **kwargs) -> "Geometry":
        """Import geometry from CAD file (STEP, IGES)."""
        try:
            from .handlers import CADHandler
        except ImportError:
            raise ImportError("handlers.py not found. Please ensure Geometry/handlers.py exists.")
        
        geo = cls()
        geo.vertices, geo.faces, geo.normals = CADHandler.read(filepath, **kwargs)
        geo.is_surface = True
        geo._source_file = str(filepath)
        geo._update_bounds()
        geo._metadata['format'] = 'cad'
        return geo
