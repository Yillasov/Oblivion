from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from enum import Enum

class GeometryType(Enum):
    """Types of supported geometries."""
    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    MESH = "mesh"
    PARAMETRIC = "parametric"

class PrimitiveType(Enum):
    """Types of primitive geometries."""
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CONE = "cone"
    WEDGE = "wedge"
    ELLIPSOID = "ellipsoid"

@dataclass
class Transform:
    """3D transformation data."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles in radians
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    def apply_to_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transformation to a point."""
        # Create rotation matrix from Euler angles
        rx, ry, rz = self.rotation
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        # Apply scale, rotation, and translation
        transformed = np.multiply(point, self.scale)
        transformed = R @ transformed
        transformed = transformed + self.position
        
        return transformed

@dataclass
class Material:
    """Material properties for geometry."""
    name: str
    radar_reflectivity: float = 0.5  # 0.0 to 1.0
    ir_emissivity: float = 0.5       # 0.0 to 1.0
    visual_properties: Dict[str, Any] = field(default_factory=dict)
    physical_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrimitiveGeometry:
    """Primitive geometry representation."""
    type: PrimitiveType
    dimensions: Dict[str, float]
    transform: Transform = field(default_factory=Transform)
    material: Optional[Material] = None
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max)."""
        if self.type == PrimitiveType.BOX:
            half_size = np.array([
                self.dimensions.get("width", 0) / 2,
                self.dimensions.get("height", 0) / 2,
                self.dimensions.get("depth", 0) / 2
            ])
            return -half_size, half_size
            
        elif self.type == PrimitiveType.SPHERE:
            radius = self.dimensions.get("radius", 0)
            return np.array([-radius, -radius, -radius]), np.array([radius, radius, radius])
            
        # Add other primitive types as needed
        
        # Default fallback
        return np.zeros(3), np.zeros(3)

@dataclass
class MeshGeometry:
    """Mesh-based geometry representation."""
    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: np.ndarray     # Mx3 array of vertex indices
    transform: Transform = field(default_factory=Transform)
    material: Optional[Material] = None
    normals: Optional[np.ndarray] = None  # Nx3 array of vertex normals
    uvs: Optional[np.ndarray] = None      # Nx2 array of texture coordinates
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max)."""
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
            
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds
    
    def transform_vertices(self) -> np.ndarray:
        """Get transformed vertices."""
        transformed = np.zeros_like(self.vertices)
        for i, vertex in enumerate(self.vertices):
            transformed[i] = self.transform.apply_to_point(vertex)
        return transformed

class ComplexGeometry:
    """Complex geometry composed of multiple primitive or mesh geometries."""
    
    def __init__(self, name: str):
        self.name = name
        self.components: List[Union[PrimitiveGeometry, MeshGeometry, 'ComplexGeometry']] = []
        self.transform = Transform()
        
    def add_component(self, component: Union[PrimitiveGeometry, MeshGeometry, 'ComplexGeometry']) -> None:
        """Add a component to the complex geometry."""
        self.components.append(component)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max) for the entire complex geometry."""
        if not self.components:
            return np.zeros(3), np.zeros(3)
            
        # Initialize with the first component
        min_bounds, max_bounds = self.components[0].get_bounding_box()
        
        # Expand bounds to include all components
        for component in self.components[1:]:
            comp_min, comp_max = component.get_bounding_box()
            min_bounds = np.minimum(min_bounds, comp_min)
            max_bounds = np.maximum(max_bounds, comp_max)
            
        return min_bounds, max_bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        components_data = []
        
        for component in self.components:
            if isinstance(component, PrimitiveGeometry):
                components_data.append({
                    "type": "primitive",
                    "primitive_type": component.type.value,
                    "dimensions": component.dimensions,
                    "transform": {
                        "position": component.transform.position.tolist(),
                        "rotation": component.transform.rotation.tolist(),
                        "scale": component.transform.scale.tolist()
                    },
                    "material": component.material.__dict__ if component.material else None
                })
            elif isinstance(component, MeshGeometry):
                components_data.append({
                    "type": "mesh",
                    "vertices_count": len(component.vertices),
                    "faces_count": len(component.faces),
                    "transform": {
                        "position": component.transform.position.tolist(),
                        "rotation": component.transform.rotation.tolist(),
                        "scale": component.transform.scale.tolist()
                    },
                    "material": component.material.__dict__ if component.material else None
                })
            elif isinstance(component, ComplexGeometry):
                components_data.append({
                    "type": "complex",
                    "name": component.name,
                    "components": component.to_dict()["components"]
                })
                
        return {
            "name": self.name,
            "components": components_data,
            "transform": {
                "position": self.transform.position.tolist(),
                "rotation": self.transform.rotation.tolist(),
                "scale": self.transform.scale.tolist()
            }
        }