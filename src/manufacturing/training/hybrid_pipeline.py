from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np

from .specialized_pipeline import SpecializedTrainingPipeline, TrainingDomain
from src.core.utils.logging_framework import get_logger

logger = get_logger("hybrid_training")

class HybridMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class HybridTrainingPipeline:
    """Handles hybrid training approaches combining multiple specialized pipelines."""
    
    def __init__(self, domains: List[TrainingDomain], mode: HybridMode = HybridMode.SEQUENTIAL):
        self.pipelines = {
            domain: SpecializedTrainingPipeline(domain)
            for domain in domains
        }
        self.mode = mode
        self.results: Dict[str, Any] = {}
    
    def train(self, training_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Execute hybrid training across multiple domains."""
        if self.mode == HybridMode.SEQUENTIAL:
            return self._train_sequential(training_data)
        elif self.mode == HybridMode.PARALLEL:
            return self._train_parallel(training_data)
        else:
            return self._train_adaptive(training_data)
    
    def _train_sequential(self, training_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train domains sequentially, using results from previous domains."""
        results = {}
        combined_data = training_data.copy()
        
        for domain, pipeline in self.pipelines.items():
            domain_results = pipeline.train(combined_data)
            results[domain.value] = domain_results
            
            # Update training data with results for next pipeline
            if domain_results["status"] == "success":
                combined_data.update({
                    f"{domain.value}_features": domain_results["results"]
                })
        
        return {
            "mode": "sequential",
            "status": "success",
            "domain_results": results
        }
    
    def _train_parallel(self, training_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train all domains in parallel."""
        results = {}
        
        for domain, pipeline in self.pipelines.items():
            results[domain.value] = pipeline.train(training_data)
        
        return {
            "mode": "parallel",
            "status": "success",
            "domain_results": results
        }
    
    def _train_adaptive(self, training_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Adaptively choose between sequential and parallel based on data characteristics."""
        # Simple adaptive logic: use parallel for small datasets, sequential for large
        if len(training_data["features"]) < 5000:
            return self._train_parallel(training_data)
        else:
            return self._train_sequential(training_data)