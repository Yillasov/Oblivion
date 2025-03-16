"""
Manufacturing integration for payload systems.

Provides interfaces for integrating payload designs with manufacturing processes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os
import time

from src.payload.base import PayloadSpecs


@dataclass
class ManufacturingSpec:
    """Manufacturing specifications for a payload component."""
    component_id: str
    materials: List[str]
    tolerances: Dict[str, float]
    assembly_steps: List[Dict[str, Any]]
    quality_checks: List[str]
    estimated_time: float  # Manufacturing time in hours


class ManufacturingIntegration:
    """
    Integrates payload designs with manufacturing processes.
    """
    
    def __init__(self, manufacturing_system_api=None):
        """Initialize the manufacturing integration system."""
        self.manufacturing_api = manufacturing_system_api
        self.specs_registry: Dict[str, ManufacturingSpec] = {}
        self.payload_mappings: Dict[str, List[str]] = {}  # Payload ID to component IDs
    
    def register_component_spec(self, spec: ManufacturingSpec) -> bool:
        """Register manufacturing specifications for a component."""
        if spec.component_id in self.specs_registry:
            return False
        
        self.specs_registry[spec.component_id] = spec
        return True
    
    def map_payload_to_components(self, payload_id: str, component_ids: List[str]) -> bool:
        """Map a payload to its manufacturing components."""
        # Verify all components exist
        for cid in component_ids:
            if cid not in self.specs_registry:
                return False
        
        self.payload_mappings[payload_id] = component_ids
        return True
    
    def generate_manufacturing_order(self, payload_id: str) -> Dict[str, Any]:
        """Generate a manufacturing order for a payload."""
        if payload_id not in self.payload_mappings:
            return {"error": "Payload not mapped to components"}
        
        component_ids = self.payload_mappings[payload_id]
        components = [self.specs_registry[cid] for cid in component_ids]
        
        # Calculate manufacturing metrics
        total_time = sum(comp.estimated_time for comp in components)
        materials = set()
        for comp in components:
            materials.update(comp.materials)
        
        # Create manufacturing order
        order = {
            "payload_id": payload_id,
            "components": component_ids,
            "total_estimated_time": total_time,
            "materials_required": list(materials),
            "assembly_sequence": self._generate_assembly_sequence(components),
            "quality_checks": self._compile_quality_checks(components),
            "timestamp": time.time()
        }
        
        # Send to manufacturing API if available
        if self.manufacturing_api:
            try:
                order_id = self.manufacturing_api.submit_order(order)
                order["order_id"] = order_id
            except Exception as e:
                order["submission_error"] = str(e)
        
        return order
    
    def _generate_assembly_sequence(self, components: List[ManufacturingSpec]) -> List[Dict[str, Any]]:
        """Generate an optimized assembly sequence from component specs."""
        # Simple implementation - in reality would use more sophisticated algorithms
        sequence = []
        for comp in components:
            sequence.extend(comp.assembly_steps)
        
        # Sort by any dependencies (simplified)
        return sorted(sequence, key=lambda x: x.get("step_number", 0))
    
    def _compile_quality_checks(self, components: List[ManufacturingSpec]) -> Dict[str, List[str]]:
        """Compile quality checks from all components."""
        checks = {}
        for comp in components:
            checks[comp.component_id] = comp.quality_checks
        
        return checks
    
    def export_manufacturing_data(self, payload_id: str, output_dir: str) -> str:
        """Export manufacturing data to files."""
        order = self.generate_manufacturing_order(payload_id)
        
        if "error" in order:
            return f"Error: {order['error']}"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write manufacturing order to JSON file
        output_file = os.path.join(output_dir, f"{payload_id}_manufacturing.json")
        with open(output_file, 'w') as f:
            json.dump(order, f, indent=2)
        
        return output_file