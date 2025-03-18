from enum import Enum
from typing import Dict, Any, Optional
import os
from .format_handlers import STLHandler, OBJHandler

class CADFormat(Enum):
    STEP = "step"
    STL = "stl"
    OBJ = "obj"

class CADHandler:
    """Base handler for CAD file operations."""
    
    def __init__(self):
        self.format_handlers = {
            ".stl": STLHandler(),
            ".obj": OBJHandler()
        }
        
        self.supported_formats = {
            CADFormat.STEP: [".step", ".stp"],
            CADFormat.STL: [".stl"],
            CADFormat.OBJ: [".obj"]
        }
    
    def get_handler(self, filepath: str):
        """Get appropriate handler for file format."""
        ext = os.path.splitext(filepath)[1].lower()
        return self.format_handlers.get(ext)
    
    def get_format(self, filepath: str) -> Optional[CADFormat]:
        """Determine CAD format from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        for format_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return format_type
        return None