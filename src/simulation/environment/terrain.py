"""
Terrain Rendering and Collision Detection

Provides a simple terrain model with height mapping and collision detection
for flight simulation.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from dataclasses import dataclass

from src.core.utils.logging_framework import get_logger

logger = get_logger("terrain")


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    
    # Terrain dimensions
    width: float = 10000.0  # meters
    length: float = 10000.0  # meters
    
    # Resolution (points per side)
    resolution: int = 100
    
    # Height range
    min_height: float = 0.0  # meters
    max_height: float = 1000.0  # meters
    
    # Roughness (0-1)
    roughness: float = 0.5
    
    # Seed for reproducibility
    seed: int = 42


class TerrainModel:
    """
    Simple terrain model with height mapping and collision detection.
    """
    
    def __init__(self, config: Optional[TerrainConfig] = None):
        """Initialize the terrain model."""
        self.config = config if config is not None else TerrainConfig()
        
        # Initialize random generator
        self.rng = np.random.RandomState(self.config.seed)
        
        # Generate height map
        self.height_map = self._generate_height_map()
        
        # Calculate cell size
        self.cell_width = self.config.width / (self.config.resolution - 1)
        self.cell_length = self.config.length / (self.config.resolution - 1)
        
        # Origin coordinates (center of terrain)
        self.origin_x = -self.config.width / 2
        self.origin_y = -self.config.length / 2
        
        logger.info(f"Terrain model initialized with {self.config.resolution}x{self.config.resolution} grid")
    
    def _generate_height_map(self) -> np.ndarray:
        """
        Generate a height map using diamond-square algorithm (simplified).
        
        Returns:
            np.ndarray: 2D height map
        """
        # Ensure resolution is 2^n + 1
        n = self.config.resolution
        
        # Initialize height map
        height_map = np.zeros((n, n))
        
        # Set corner values
        height_map[0, 0] = self.rng.uniform(0, 1)
        height_map[0, n-1] = self.rng.uniform(0, 1)
        height_map[n-1, 0] = self.rng.uniform(0, 1)
        height_map[n-1, n-1] = self.rng.uniform(0, 1)
        
        # Simple implementation of diamond-square
        step = n - 1
        scale = 1.0
        
        while step > 1:
            half_step = step // 2
            
            # Diamond step
            for i in range(half_step, n, step):
                for j in range(half_step, n, step):
                    avg = (height_map[i-half_step, j-half_step] +
                           height_map[i-half_step, j+half_step] +
                           height_map[i+half_step, j-half_step] +
                           height_map[i+half_step, j+half_step]) / 4.0
                    
                    height_map[i, j] = avg + self.rng.uniform(-scale, scale) * self.config.roughness
            
            # Square step
            for i in range(0, n, half_step):
                for j in range((i + half_step) % step, n, step):
                    count = 0
                    avg = 0.0
                    
                    if i >= half_step:
                        avg += height_map[i-half_step, j]
                        count += 1
                    if i + half_step < n:
                        avg += height_map[i+half_step, j]
                        count += 1
                    if j >= half_step:
                        avg += height_map[i, j-half_step]
                        count += 1
                    if j + half_step < n:
                        avg += height_map[i, j+half_step]
                        count += 1
                    
                    avg /= count
                    height_map[i, j] = avg + self.rng.uniform(-scale, scale) * self.config.roughness
            
            step = half_step
            scale *= 0.5
        
        # Normalize and scale to desired height range
        height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        height_map = self.config.min_height + height_map * (self.config.max_height - self.config.min_height)
        
        return height_map
    
    def get_height(self, x: float, y: float) -> float:
        """
        Get terrain height at the specified position.
        
        Args:
            x: X coordinate (m)
            y: Y coordinate (m)
            
        Returns:
            float: Terrain height (m)
        """
        # Convert to local coordinates
        local_x = x - self.origin_x
        local_y = y - self.origin_y
        
        # Check if within bounds
        if (local_x < 0 or local_x >= self.config.width or
            local_y < 0 or local_y >= self.config.length):
            return 0.0  # Default height for out-of-bounds
        
        # Convert to grid indices
        grid_x = local_x / self.cell_width
        grid_y = local_y / self.cell_length
        
        # Get grid cell indices
        i0 = int(grid_x)
        j0 = int(grid_y)
        i1 = min(i0 + 1, self.config.resolution - 1)
        j1 = min(j0 + 1, self.config.resolution - 1)
        
        # Get fractional parts for interpolation
        fx = grid_x - i0
        fy = grid_y - j0
        
        # Bilinear interpolation
        h00 = self.height_map[j0, i0]
        h01 = self.height_map[j0, i1]
        h10 = self.height_map[j1, i0]
        h11 = self.height_map[j1, i1]
        
        h0 = h00 * (1 - fx) + h01 * fx
        h1 = h10 * (1 - fx) + h11 * fx
        
        return h0 * (1 - fy) + h1 * fy
    
    def get_normal(self, x: float, y: float) -> np.ndarray:
        """
        Get terrain normal vector at the specified position.
        
        Args:
            x: X coordinate (m)
            y: Y coordinate (m)
            
        Returns:
            np.ndarray: Normal vector [nx, ny, nz]
        """
        # Sample heights at nearby points
        eps = 1.0  # Small distance for gradient calculation
        h_center = self.get_height(x, y)
        h_x = self.get_height(x + eps, y)
        h_y = self.get_height(x, y + eps)
        
        # Calculate tangent vectors
        tangent_x = np.array([eps, 0, h_x - h_center])
        tangent_y = np.array([0, eps, h_y - h_center])
        
        # Cross product to get normal
        normal = np.cross(tangent_x, tangent_y)
        
        # Normalize
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length
        else:
            normal = np.array([0, 0, 1])  # Default normal
        
        return normal
    
    def check_collision(self, position: np.ndarray, radius: float = 0.0) -> Tuple[bool, float, np.ndarray]:
        """
        Check if an object collides with the terrain.
        
        Args:
            position: Object position [x, y, z] (m)
            radius: Object collision radius (m)
            
        Returns:
            Tuple[bool, float, np.ndarray]: (collision, penetration depth, normal)
        """
        x, y, z = position
        terrain_height = self.get_height(x, y)
        
        # Check if below terrain (z is up)
        if z - radius <= terrain_height:
            penetration = terrain_height - (z - radius)
            normal = self.get_normal(x, y)
            return True, penetration, normal
        
        return False, 0.0, np.array([0, 0, 1])
    
    def get_rendering_data(self) -> Dict[str, Any]:
        """
        Get data needed for rendering the terrain.
        
        Returns:
            Dict[str, Any]: Rendering data
        """
        # Create vertex positions
        vertices = []
        normals = []
        
        for j in range(self.config.resolution):
            for i in range(self.config.resolution):
                # Calculate position
                x = self.origin_x + i * self.cell_width
                y = self.origin_y + j * self.cell_length
                z = self.height_map[j, i]
                
                vertices.append([x, y, z])
                normals.append(self.get_normal(x, y).tolist())
        
        # Create triangle indices
        indices = []
        for j in range(self.config.resolution - 1):
            for i in range(self.config.resolution - 1):
                # Calculate vertex indices
                v0 = j * self.config.resolution + i
                v1 = j * self.config.resolution + i + 1
                v2 = (j + 1) * self.config.resolution + i
                v3 = (j + 1) * self.config.resolution + i + 1
                
                # Add two triangles for each grid cell
                indices.append([v0, v1, v2])
                indices.append([v2, v1, v3])
        
        return {
            'vertices': vertices,
            'normals': normals,
            'indices': indices,
            'width': self.config.width,
            'length': self.config.length,
            'min_height': self.config.min_height,
            'max_height': self.config.max_height
        }


def create_default_terrain() -> TerrainModel:
    """
    Create a default terrain model with moderate hills.
    
    Returns:
        TerrainModel: Default terrain model
    """
    config = TerrainConfig(
        width=20000.0,
        length=20000.0,
        resolution=129,  # 2^7 + 1
        min_height=0.0,
        max_height=800.0,
        roughness=0.6,
        seed=42
    )
    
    return TerrainModel(config)