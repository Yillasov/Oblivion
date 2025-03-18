from typing import Dict, Any, Optional, List
import numpy as np
from enum import Enum

from .specialized_pipeline import TrainingDomain, SpecializedTrainingPipeline
from src.core.utils.logging_framework import get_logger

logger = get_logger("transfer_learning")

class TransferStrategy(Enum):
    FULL = "full"          # Transfer all learned features
    SELECTIVE = "selective"  # Transfer only specific features
    ADAPTIVE = "adaptive"   # Adaptively choose features to transfer

class TransferLearningPipeline:
    def __init__(self, 
                 source_domain: TrainingDomain,
                 target_domain: TrainingDomain,
                 strategy: TransferStrategy = TransferStrategy.ADAPTIVE):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.strategy = strategy
        self.source_pipeline = SpecializedTrainingPipeline(source_domain)
        self.target_pipeline = SpecializedTrainingPipeline(target_domain)
        
    def transfer_and_train(self, 
                          source_data: Dict[str, np.ndarray],
                          target_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Execute transfer learning from source to target domain."""
        
        # Train source domain
        source_results = self.source_pipeline.train(source_data)
        if source_results["status"] != "success":
            return {"status": "failed", "error": "Source training failed"}
            
        # Extract transferable features
        transferred_features = self._extract_transferable_features(
            source_results, source_data, target_data
        )
        
        # Adapt target data with transferred features
        adapted_target_data = self._adapt_target_data(target_data, transferred_features)
        
        # Train target domain with transferred knowledge
        target_results = self.target_pipeline.train(adapted_target_data)
        
        return {
            "status": "success",
            "source_results": source_results,
            "target_results": target_results,
            "transfer_stats": {
                "features_transferred": len(transferred_features),
                "strategy_used": self.strategy.value
            }
        }
    
    def _extract_transferable_features(self,
                                     source_results: Dict[str, Any],
                                     source_data: Dict[str, np.ndarray],
                                     target_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Extract features that can be transferred to target domain."""
        if self.strategy == TransferStrategy.FULL:
            return source_results.get("results", {}).get("features", [])
        
        elif self.strategy == TransferStrategy.SELECTIVE:
            # Select features based on correlation with target domain
            features = source_results.get("results", {}).get("features", [])
            return [f for f in features if self._is_feature_relevant(f, target_data)]
        
        else:  # ADAPTIVE
            return self._adaptive_feature_selection(source_results, target_data)
    
    def _adapt_target_data(self,
                          target_data: Dict[str, np.ndarray],
                          transferred_features: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Adapt target data with transferred features."""
        adapted_data = target_data.copy()
        
        if transferred_features:
            adapted_data["transferred_features"] = np.array(transferred_features)
            
        return adapted_data
    
    def _is_feature_relevant(self, feature: np.ndarray, target_data: Dict[str, np.ndarray]) -> bool:
        """Check if a feature is relevant for the target domain."""
        # Simple relevance check based on correlation
        target_features = target_data.get("features", np.array([]))
        if len(target_features) > 0:
            correlation = np.corrcoef(feature, target_features.mean(axis=1))[0, 1]
            return abs(correlation) > 0.5
        return False
    
    def _adaptive_feature_selection(self,
                                  source_results: Dict[str, Any],
                                  target_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Adaptively select features based on target domain characteristics."""
        features = source_results.get("results", {}).get("features", [])
        selected_features = []
        
        for feature in features:
            if self._is_feature_relevant(feature, target_data):
                selected_features.append(feature)
                
        return selected_features