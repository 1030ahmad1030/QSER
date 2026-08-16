"""
Geometry Utilities
===================

Helper functions for geometry operations:
- Bounding box
- Volume estimation
- Point-in-geometry test
- Mesh grid generation
"""

import numpy as np
from typing import Tuple, Optional


def bounding_box(vertices: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute bounding box of vertices.
    
    Args:
        vertices: (N, 3) array
        
    Returns:
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, ymin, zmin = vertices.min(axis=0)
    xmax, ymax, zmax = vertices.max(axis=0)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


def estimate_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Estimate volume of closed surface mesh.
    
    Args:
        vertices: (N, 3) array
        faces: (M, 3) face indices
        
    Returns:
        Volume (float)
    """
    # Check if mesh is closed
    face_vertices = vertices[faces]
    v0 = face_vertices[:, 0]
    v1 = face_vertices[:, 1]
    v2 = face_vertices[:, 2]
    
    # Volume of tetrahedron from origin
    volume = np.sum(
        np.abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)))
    ) / 6.0
    
    return volume


def contains_points(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Check if points are inside a closed mesh.
    
    Args:
        points: (N, 3) points to test
        vertices: (M, 3) mesh vertices
        faces: (K, 3) face indices
        
    Returns:
        Boolean array of length N
    """
    try:
        from trimesh import Trimesh
    except ImportError:
        raise ImportError("trimesh required for point-in-mesh tests. Install: pip install trimesh")
    
    mesh = Trimesh(vertices=vertices, faces=faces)
    return mesh.contains(points)


def mesh_grid_from_geometry(
    geometry,
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
    method: str = 'uniform'
) -> np.ndarray:
    """
    Generate a structured grid from geometry bounds.
    
    Args:
        geometry: Geometry object
        nx, ny, nz: Number of points in each direction
        method: 'uniform', 'adaptive'
        
    Returns:
        (nx*ny*nz, 3) array of grid points
    """
    bounds = geometry.bounding_box()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    if method == 'uniform':
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return points
    else:
        raise ValueError(f"Unknown grid method: {method}")
