"""
Format Handlers for Geometry Module
===================================

Each handler reads a specific file format and returns:
    - vertices: (N, 3) array
    - faces: (M, 3) array (optional, None for point clouds)
    - normals: (M, 3) array (optional)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
import warnings


class STLHandler:
    """Handler for STL files."""
    
    @staticmethod
    def read(filepath: Union[str, Path], **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read STL file and return vertices, faces, normals."""
        try:
            from stl import mesh
        except ImportError:
            raise ImportError("numpy-stl required for STL import. Install: pip install numpy-stl")
        
        stl_mesh = mesh.Mesh.from_file(str(filepath))
        vertices = stl_mesh.vectors.reshape(-1, 3)
        # Remove duplicate vertices
        vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
        faces = indices.reshape(-1, 3)
        normals = stl_mesh.normals
        return vertices, faces, normals


class OBJHandler:
    """Handler for OBJ files."""
    
    @staticmethod
    def read(filepath: Union[str, Path], **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read OBJ file and return vertices, faces, normals."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required for OBJ import. Install: pip install trimesh")
        
        mesh = trimesh.load(str(filepath))
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
        return vertices, faces, normals


class PLYHandler:
    """Handler for PLY files."""
    
    @staticmethod
    def read(filepath: Union[str, Path], **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read PLY file and return vertices, faces, normals."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required for PLY import. Install: pip install trimesh")
        
        mesh = trimesh.load(str(filepath))
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
        return vertices, faces, normals


class OFFHandler:
    """Handler for OFF files."""
    
    @staticmethod
    def read(filepath: Union[str, Path], **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read OFF file and return vertices, faces."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        idx = 0
        while idx < len(lines) and lines[idx].strip().upper() not in ['OFF', 'OFF BINARY', 'OFF ASCII']:
            idx += 1
        idx += 1
        
        # Read vertex and face counts
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        n_vertices, n_faces, _ = map(int, lines[idx].strip().split())
        idx += 1
        
        # Read vertices
        vertices = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            vertices[i] = list(map(float, lines[idx].strip().split()[:3]))
            idx += 1
        
        # Read faces
        faces = []
        for i in range(n_faces):
            while idx < len(lines) and lines[idx].strip() == '':
                idx += 1
            parts = list(map(int, lines[idx].strip().split()))
            n = parts[0]
            faces.append(parts[1:1+n])
            idx += 1
        
        faces = np.array(faces)
        return vertices, faces, None


class CSVHandler:
    """Handler for CSV point cloud files."""
    
    @staticmethod
    def read(
        filepath: Union[str, Path],
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: str = 'z',
        boundary_col: Optional[str] = None,
        delimiter: str = ',',
        header: int = 0,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read CSV file and return points."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV import. Install: pip install pandas")
        
        df = pd.read_csv(filepath, delimiter=delimiter, header=header)
        points = df[[x_col, y_col, z_col]].values
        
        # Store boundary info in normals if boundary_col provided
        boundary_labels = None
        if boundary_col is not None and boundary_col in df.columns:
            boundary_labels = df[boundary_col].values
        
        return points, None, boundary_labels


class ExcelHandler:
    """Handler for Excel point cloud files."""
    
    @staticmethod
    def read(
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        x_col: str = 'x',
        y_col: str = 'y',
        z_col: str = 'z',
        boundary_col: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read Excel file and return points."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Excel import. Install: pip install pandas openpyxl")
        
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        points = df[[x_col, y_col, z_col]].values
        
        boundary_labels = None
        if boundary_col is not None and boundary_col in df.columns:
            boundary_labels = df[boundary_col].values
        
        return points, None, boundary_labels


class PointsHandler:
    """Handler for direct point input."""
    
    @staticmethod
    def read(
        points: np.ndarray,
        boundary_indices: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return points directly."""
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got {points.shape}")
        
        boundary_labels = None
        if boundary_indices is not None:
            boundary_labels = np.zeros(len(points))
            boundary_labels[boundary_indices] = 1
        
        return points, None, boundary_labels


class BuiltinHandler:
    """Handler for built-in geometries."""
    
    @staticmethod
    def sphere(center=(0, 0, 0), radius=1.0, n_subdivisions=3, **kwargs):
        """Create a sphere mesh."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required for built-in geometries. Install: pip install trimesh")
        
        mesh = trimesh.creation.icosphere(subdivisions=n_subdivisions, radius=radius)
        vertices = np.array(mesh.vertices) + np.array(center)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
        return vertices, faces, normals
    
    @staticmethod
    def cube(center=(0, 0, 0), size=2.0, **kwargs):
        """Create a cube mesh."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required for built-in geometries. Install: pip install trimesh")
        
        mesh = trimesh.creation.box(extents=[size, size, size])
        vertices = np.array(mesh.vertices) + np.array(center)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
        return vertices, faces, normals
    
    @staticmethod
    def cylinder(center=(0, 0, 0), radius=0.5, height=2.0, segments=32, **kwargs):
        """Create a cylinder mesh."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh required for built-in geometries. Install: pip install trimesh")
        
        mesh = trimesh.creation.cylinder(radius=radius, height=height, segments=segments)
        vertices = np.array(mesh.vertices) + np.array(center)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
        return vertices, faces, normals


class CADHandler:
    """Handler for CAD files (STEP, IGES) using trimesh."""
    
    @staticmethod
    def read(filepath: Union[str, Path], **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read CAD file and return vertices, faces, normals."""
        filepath = str(filepath)
        
        # Try trimesh first (it has good STEP support via OCC)
        try:
            import trimesh
            
            # Load the CAD file
            mesh = trimesh.load(filepath)
            
            # Check if we got a valid mesh
            if mesh is not None:
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    vertices = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                    normals = np.array(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else None
                    
                    # Ensure faces are integers
                    faces = faces.astype(np.int64)
                    
                    return vertices, faces, normals
                elif hasattr(mesh, 'geometry'):
                    # Might be a scene with multiple geometries
                    vertices_list = []
                    faces_list = []
                    for geom in mesh.geometry.values():
                        vertices_list.append(np.array(geom.vertices))
                        faces_list.append(np.array(geom.faces))
                    
                    if vertices_list:
                        vertices = np.vstack(vertices_list)
                        # Adjust face indices
                        offset = 0
                        all_faces = []
                        for v, f in zip(vertices_list, faces_list):
                            all_faces.append(f + offset)
                            offset += len(v)
                        faces = np.vstack(all_faces)
                        return vertices, faces, None
            
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"trimesh import failed: {e}")
        
        # Fallback: try cadquery
        try:
            import cadquery as cq
            import tempfile
            import os
            
            # Read the CAD file
            if filepath.endswith('.step') or filepath.endswith('.stp'):
                part = cq.importers.importStep(filepath)
            elif filepath.endswith('.iges') or filepath.endswith('.igs'):
                part = cq.importers.importIges(filepath)
            else:
                raise ValueError(f"Unsupported CAD format: {filepath}")
            
            # Export to STL
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                cq.exporters.export(part, tmp_path, 'STL')
                # Read the STL using STLHandler
                return STLHandler.read(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except ImportError:
            raise ImportError(
                "CAD import requires trimesh or cadquery. "
                "Install: pip install trimesh  OR  pip install cadquery"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to import CAD file: {e}")
