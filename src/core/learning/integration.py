"""
Centralized neuromorphic learning integration module.

Provides a unified interface for applying learning algorithms across
different subsystems and hardware platforms.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Callable, Type, Union
import logging
import time
import threading
import numpy as np

from src.core.integration.neuromorphic_system import NeuromorphicSystem
from src.core.optimization.hardware_switching import HardwareSwitchingOptimizer

logger = logging.getLogger(__name__)


class LearningStrategy:
    """Learning strategy types."""
    HEBBIAN = "hebbian"
    STDP = "stdp"
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    TRANSFER = "transfer"


class LearningIntegration:
    """
    Centralized learning integration for neuromorphic systems.
    
    Provides:
    - Unified learning interface across subsystems
    - Knowledge transfer between components
    - Adaptive learning based on hardware capabilities
    """
    
    def __init__(self, 
                 system: Optional[NeuromorphicSystem] = None,
                 hardware_optimizer: Optional[HardwareSwitchingOptimizer] = None):
        """
        Initialize learning integration.
        
        Args:
            system: Neuromorphic system
            hardware_optimizer: Hardware switching optimizer
        """
        self.system = system
        self.hardware_optimizer = hardware_optimizer
        self.learning_modules = {}
        self.shared_knowledge = {}
        self.active_strategies = set()
        self.learning_lock = threading.RLock()
        
    def register_learning_module(self, 
                               module_id: str, 
                               module: Any, 
                               strategy: str = LearningStrategy.HEBBIAN) -> bool:
        """
        Register a learning module.
        
        Args:
            module_id: Unique identifier for the module
            module: Learning module instance
            strategy: Learning strategy
            
        Returns:
            bool: Success status
        """
        with self.learning_lock:
            if module_id in self.learning_modules:
                logger.warning(f"Learning module '{module_id}' already registered")
                return False
                
            self.learning_modules[module_id] = {
                "module": module,
                "strategy": strategy,
                "active": True,
                "last_update": time.time(),
                "performance": {}
            }
            
            self.active_strategies.add(strategy)
            logger.info(f"Registered learning module '{module_id}' with {strategy} strategy")
            return True
    
    def apply_learning(self, 
                     module_id: str, 
                     inputs: Dict[str, Any], 
                     targets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply learning for a specific module.
        
        Args:
            module_id: Module identifier
            inputs: Input data
            targets: Target data (for supervised learning)
            
        Returns:
            Dict[str, Any]: Learning results
        """
        with self.learning_lock:
            if module_id not in self.learning_modules:
                logger.warning(f"Learning module '{module_id}' not found")
                return {"error": "module_not_found"}
                
            module_info = self.learning_modules[module_id]
            module = module_info["module"]
            strategy = module_info["strategy"]
            
            # Check if module has the right method based on strategy
            if strategy == LearningStrategy.SUPERVISED and hasattr(module, "learn_supervised"):
                if targets is None:
                    logger.warning(f"Targets required for supervised learning in '{module_id}'")
                    return {"error": "targets_required"}
                results = module.learn_supervised(inputs, targets)
            elif strategy == LearningStrategy.REINFORCEMENT and hasattr(module, "learn_reinforcement"):
                results = module.learn_reinforcement(inputs)
            elif hasattr(module, "learn"):
                # Generic learn method
                results = module.learn(inputs, targets)
            else:
                logger.warning(f"No compatible learning method found for '{module_id}'")
                return {"error": "no_learning_method"}
            
            # Update module info
            module_info["last_update"] = time.time()
            module_info["performance"] = results.get("performance", {})
            
            # Update shared knowledge
            self._update_shared_knowledge(module_id, results)
            
            return results
    
    def _update_shared_knowledge(self, module_id: str, results: Dict[str, Any]) -> None:
        """
        Update shared knowledge base with learning results.
        
        Args:
            module_id: Module identifier
            results: Learning results
        """
        # Extract knowledge to share
        if "shared_knowledge" in results:
            knowledge = results["shared_knowledge"]
            
            # Add to shared knowledge base
            for key, value in knowledge.items():
                if key not in self.shared_knowledge:
                    self.shared_knowledge[key] = {}
                
                self.shared_knowledge[key][module_id] = {
                    "value": value,
                    "timestamp": time.time(),
                    "confidence": results.get("confidence", 0.5)
                }
    
    def get_shared_knowledge(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get shared knowledge.
        
        Args:
            key: Optional specific knowledge key
            
        Returns:
            Dict[str, Any]: Shared knowledge
        """
        with self.learning_lock:
            if key is not None:
                return self.shared_knowledge.get(key, {})
            return self.shared_knowledge
    
    def optimize_learning(self) -> Dict[str, Any]:
        """
        Optimize learning across all modules.
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        with self.learning_lock:
            results = {}
            
            # Check hardware capabilities if optimizer is available
            hardware_capabilities = {}
            if self.hardware_optimizer:
                current_hardware = self.hardware_optimizer.current_hardware
                hardware_info = self.hardware_optimizer.get_current_performance()
                hardware_capabilities = {
                    "hardware_type": current_hardware,
                    "performance": hardware_info
                }
            
            # Adjust learning parameters based on hardware
            for module_id, module_info in self.learning_modules.items():
                module = module_info["module"]
                strategy = module_info["strategy"]
                
                # Skip inactive modules
                if not module_info["active"]:
                    continue
                
                # Adjust learning parameters if method exists
                if hasattr(module, "adjust_parameters") and hardware_capabilities:
                    try:
                        adjusted = module.adjust_parameters(hardware_capabilities)
                        results[module_id] = {
                            "adjusted": adjusted,
                            "hardware": hardware_capabilities.get("hardware_type")
                        }
                    except Exception as e:
                        logger.error(f"Error adjusting parameters for '{module_id}': {str(e)}")
                        results[module_id] = {"error": str(e)}
            
            return results
    
    def transfer_knowledge(self, 
                         source_id: str, 
                         target_id: str, 
                         knowledge_keys: Optional[List[str]] = None) -> bool:
        """
        Transfer knowledge between learning modules.
        
        Args:
            source_id: Source module ID
            target_id: Target module ID
            knowledge_keys: Optional specific knowledge keys to transfer
            
        Returns:
            bool: Success status
        """
        with self.learning_lock:
            if source_id not in self.learning_modules:
                logger.warning(f"Source module '{source_id}' not found")
                return False
                
            if target_id not in self.learning_modules:
                logger.warning(f"Target module '{target_id}' not found")
                return False
            
            source_module = self.learning_modules[source_id]["module"]
            target_module = self.learning_modules[target_id]["module"]
            
            # Check if modules support knowledge transfer
            if not (hasattr(source_module, "export_knowledge") and 
                   hasattr(target_module, "import_knowledge")):
                logger.warning(f"Knowledge transfer not supported between '{source_id}' and '{target_id}'")
                return False
            
            try:
                # Export knowledge from source
                knowledge = source_module.export_knowledge(knowledge_keys)
                
                # Import knowledge to target
                success = target_module.import_knowledge(knowledge)
                
                if success:
                    logger.info(f"Successfully transferred knowledge from '{source_id}' to '{target_id}'")
                    return True
                else:
                    logger.warning(f"Failed to import knowledge to '{target_id}'")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during knowledge transfer: {str(e)}")
                return False