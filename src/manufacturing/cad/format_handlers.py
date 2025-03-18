from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class CADFormatHandler(ABC):
    """Base class for CAD format handlers."""
    
    @abstractmethod
    def read(self, filepath: str) -> Dict[str, Any]:
        """Read CAD file and return data."""
        pass
    
    @abstractmethod
    def write(self, data: Dict[str, Any], filepath: str) -> bool:
        """Write CAD data to file."""
        pass

class STLHandler(CADFormatHandler):
    def read(self, filepath: str) -> Dict[str, Any]:
        # Basic STL ASCII reader
        vertices = []
        normals = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if 'vertex' in line:
                        coords = [float(x) for x in line.split()[1:4]]
                        vertices.append(coords)
                    elif 'normal' in line:
                        normal = [float(x) for x in line.split()[2:5]]
                        normals.append(normal)
        except Exception as e:
            raise ValueError(f"Failed to read STL file: {str(e)}")
            
        return {
            "format": "STL",
            "vertices": np.array(vertices),
            "normals": np.array(normals)
        }
    
    def write(self, data: Dict[str, Any], filepath: str) -> bool:
        try:
            with open(filepath, 'w') as f:
                f.write("solid model\n")
                vertices = data.get("vertices", [])
                normals = data.get("normals", [])
                
                for i in range(0, len(vertices), 3):
                    normal = normals[i//3] if i//3 < len(normals) else [0, 0, 1]
                    f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                    f.write("  outer loop\n")
                    for j in range(3):
                        v = vertices[i+j]
                        f.write(f"    vertex {v[0]} {v[1]} {v[2]}\n")
                    f.write("  endloop\n")
                    f.write("endfacet\n")
                f.write("endsolid model\n")
            return True
        except Exception:
            return False

class OBJHandler(CADFormatHandler):
    def read(self, filepath: str) -> Dict[str, Any]:
        vertices = []
        faces = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        coords = [float(x) for x in line.split()[1:4]]
                        vertices.append(coords)
                    elif line.startswith('f '):
                        face = [int(x.split('/')[0]) for x in line.split()[1:4]]
                        faces.append(face)
        except Exception as e:
            raise ValueError(f"Failed to read OBJ file: {str(e)}")
            
        return {
            "format": "OBJ",
            "vertices": np.array(vertices),
            "faces": np.array(faces)
        }
    
    def write(self, data: Dict[str, Any], filepath: str) -> bool:
        try:
            with open(filepath, 'w') as f:
                vertices = data.get("vertices", [])
                faces = data.get("faces", [])
                
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            return True
        except Exception:
            return False