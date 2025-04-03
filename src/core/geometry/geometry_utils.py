import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import math
from src.core.geometry.complex_geometry import (
    ComplexGeometry, PrimitiveGeometry, MeshGeometry, 
    PrimitiveType, Transform, Material
)

class GeometryUtils:
    """Utility functions for working with complex geometries."""
    
    @staticmethod
    def calculate_volume(geometry: Union[ComplexGeometry, PrimitiveGeometry, MeshGeometry]) -> float:
        """Calculate approximate volume of a geometry."""
        if isinstance(geometry, PrimitiveGeometry):
            return GeometryUtils._calculate_primitive_volume(geometry)
        elif isinstance(geometry, MeshGeometry):
            return GeometryUtils._calculate_mesh_volume(geometry)
        elif isinstance(geometry, ComplexGeometry):
            total_volume = 0.0
            for component in geometry.components:
                total_volume += GeometryUtils.calculate_volume(component)
            return total_volume
        return 0.0
    
    @staticmethod
    def _calculate_primitive_volume(primitive: PrimitiveGeometry) -> float:
        """Calculate volume of a primitive geometry."""
        if primitive.type == PrimitiveType.BOX:
            width = primitive.dimensions.get("width", 0)
            height = primitive.dimensions.get("height", 0)
            depth = primitive.dimensions.get("depth", 0)
            return width * height * depth
            
        elif primitive.type == PrimitiveType.SPHERE:
            radius = primitive.dimensions.get("radius", 0)
            return (4/3) * math.pi * radius**3
            
        elif primitive.type == PrimitiveType.CYLINDER:
            radius = primitive.dimensions.get("radius", 0)
            height = primitive.dimensions.get("height", 0)
            return math.pi * radius**2 * height
            
        elif primitive.type == PrimitiveType.CONE:
            radius = primitive.dimensions.get("radius", 0)
            height = primitive.dimensions.get("height", 0)
            return (1/3) * math.pi * radius**2 * height
            
        elif primitive.type == PrimitiveType.ELLIPSOID:
            a = primitive.dimensions.get("length", 0) / 2
            b = primitive.dimensions.get("width", 0) / 2
            c = primitive.dimensions.get("height", 0) / 2
            return (4/3) * math.pi * a * b * c
            
        return 0.0
    
    @staticmethod
    def _calculate_mesh_volume(mesh: MeshGeometry) -> float:
        """Calculate volume of a mesh geometry using signed tetrahedron method."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return 0.0
            
        # Transform vertices
        vertices = mesh.transform_vertices()
        
        # Calculate volume using signed tetrahedron method
        volume = 0.0
        for face in mesh.faces:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Form tetrahedron with origin
            volume += np.dot(np.cross(v1, v2), v3) / 6.0
            
        return abs(volume)
    
    @staticmethod
    def calculate_surface_area(geometry: Union[ComplexGeometry, PrimitiveGeometry, MeshGeometry]) -> float:
        """Calculate approximate surface area of a geometry."""
        if isinstance(geometry, PrimitiveGeometry):
            return GeometryUtils._calculate_primitive_surface_area(geometry)
        elif isinstance(geometry, MeshGeometry):
            return GeometryUtils._calculate_mesh_surface_area(geometry)
        elif isinstance(geometry, ComplexGeometry):
            total_area = 0.0
            for component in geometry.components:
                total_area += GeometryUtils.calculate_surface_area(component)
            return total_area
        return 0.0
    
    @staticmethod
    def _calculate_primitive_surface_area(primitive: PrimitiveGeometry) -> float:
        """Calculate surface area of a primitive geometry."""
        if primitive.type == PrimitiveType.BOX:
            width = primitive.dimensions.get("width", 0)
            height = primitive.dimensions.get("height", 0)
            depth = primitive.dimensions.get("depth", 0)
            return 2 * (width * height + width * depth + height * depth)
            
        elif primitive.type == PrimitiveType.SPHERE:
            radius = primitive.dimensions.get("radius", 0)
            return 4 * math.pi * radius**2
            
        # Add other primitive types as needed
            
        return 0.0
    
    @staticmethod
    def _calculate_mesh_surface_area(mesh: MeshGeometry) -> float:
        """Calculate surface area of a mesh geometry."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return 0.0
            
        # Transform vertices
        vertices = mesh.transform_vertices()
        
        # Calculate area
        area = 0.0
        for face in mesh.faces:
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Calculate triangle area using cross product
            area += 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            
        return float(area)
