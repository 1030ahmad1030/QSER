"""
Point Sampling Module
======================

Provides methods for sampling points within geometries:
- Random sampling
- Latin Hypercube sampling
- Sobol sequence sampling
- Adaptive sampling
- Boundary sampling from mesh
"""

import numpy as np
from typing import Optional, Tuple, Callable
import warnings


class PointSampler:
    """Static class for point sampling methods."""
    
    @staticmethod
    def sample_within_bounds(
        n_points: int,
        bounds: Tuple[float, float, float, float, float, float],
        method: str = 'random',
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample points within bounding box.
        
        Args:
            n_points: Number of points
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            method: 'random', 'latin_hypercube', 'sobol'
            seed: Random seed
            
        Returns:
            (n_points, 3) array
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        origin = np.array([xmin, ymin, zmin])
        
        if method == 'random':
            if seed is not None:
                np.random.seed(seed)
            samples = np.random.rand(n_points, 3)
            
        elif method == 'latin_hypercube':
            samples = PointSampler._latin_hypercube(n_points, 3, seed)
            
        elif method == 'sobol':
            samples = PointSampler._sobol_sequence(n_points, 3, seed)
            
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return origin + samples * dims
    
    @staticmethod
    def sample_inside_function(
        inside_func: Callable,
        n_points: int,
        bounds: Tuple[float, float, float, float, float, float],
        method: str = 'random',
        seed: Optional[int] = None,
        max_attempts: int = 10000,
        **kwargs
    ) -> np.ndarray:
        """
        Sample points inside a mathematical domain using rejection sampling.
        
        Args:
            inside_func: Function returning True for points inside
            n_points: Number of points to sample
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            method: 'random' only (rejection sampling)
            seed: Random seed
            max_attempts: Maximum attempts per point
            
        Returns:
            (n_points, 3) array of points inside the domain
        """
        if seed is not None:
            np.random.seed(seed)
        
        points = []
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts * n_points:
            # Sample candidate point
            candidate = PointSampler.sample_within_bounds(
                1, bounds, method='random'
            )[0]
            
            # Check if inside
            if inside_func(candidate[0], candidate[1], candidate[2]):
                points.append(candidate)
            
            attempts += 1
        
        if len(points) < n_points:
            warnings.warn(
                f"Only {len(points)}/{n_points} points sampled. "
                f"Increase max_attempts or check domain size."
            )
        
        return np.array(points)
    
    @staticmethod
    def sample_boundary_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        n_points: int,
        method: str = 'uniform',
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample points on boundary of a mesh.
        
        Args:
            vertices: (N, 3) vertex array
            faces: (M, 3) face indices
            n_points: Number of boundary points
            method: 'uniform', 'random'
            seed: Random seed
            
        Returns:
            (n_points, 3) array
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute face areas for weighted sampling
        face_vertices = vertices[faces]
        v0 = face_vertices[:, 0]
        v1 = face_vertices[:, 1]
        v2 = face_vertices[:, 2]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        areas = areas / areas.sum()  # Normalize
        
        if method == 'uniform':
            # Sample faces proportionally to area
            face_indices = np.random.choice(len(faces), size=n_points, p=areas)
        else:
            # Random sample (biased)
            face_indices = np.random.choice(len(faces), size=n_points)
        
        # Sample points on each face
        points = []
        for idx in face_indices:
            face = faces[idx]
            v0, v1, v2 = vertices[face]
            
            # Barycentric coordinates
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            
            point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
            points.append(point)
        
        return np.array(points)
    
    @staticmethod
    def sample_boundary_points(
        points: np.ndarray,
        n_points: int,
        method: str = 'random',
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample boundary points from point cloud.
        
        For point clouds, this is approximate - samples from convex hull.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Simple: sample from existing points
        if len(points) <= n_points:
            return points
        
        indices = np.random.choice(len(points), size=n_points, replace=False)
        return points[indices]
    
    @staticmethod
    def _latin_hypercube(n_points: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        if seed is not None:
            np.random.seed(seed)
        
        samples = np.zeros((n_points, n_dims))
        for i in range(n_dims):
            segments = np.linspace(0, 1, n_points + 1)
            points = segments[:-1] + np.random.rand(n_points) * (1 / n_points)
            np.random.shuffle(points)
            samples[:, i] = points
        
        return samples
    
    @staticmethod
    def _sobol_sequence(n_points: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate Sobol sequence samples."""
        try:
            from scipy.stats import qmc
        except ImportError:
            warnings.warn("scipy not available. Falling back to random.")
            if seed is not None:
                np.random.seed(seed)
            return np.random.rand(n_points, n_dims)
        
        sampler = qmc.Sobol(d=n_dims, seed=seed)
        return sampler.random(n_points)
