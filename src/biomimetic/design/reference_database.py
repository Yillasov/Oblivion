#!/usr/bin/env python3
"""
Biological Reference Models Database

This module provides a database system for storing, retrieving, and searching
biological reference models for biomimetic UCAV design.
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import asdict
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils.logging_framework import get_logger
from src.biomimetic.design.principles import BiologicalReference, BiologicalInspiration

logger = get_logger("biomimetic_database")


class BiologicalReferenceDatabase:
    """Database for storing and retrieving biological reference models."""
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the biological reference database.
        
        Args:
            database_path: Path to the database file
        """
        self.references: Dict[str, BiologicalReference] = {}
        self.database_path = database_path or os.path.join(
            project_root, 
            "data", 
            "biomimetic", 
            "reference_models.json"
        )
        self.feature_index: Dict[str, Set[str]] = {}  # Index for fast feature-based lookup
        self.inspiration_index: Dict[BiologicalInspiration, Set[str]] = {
            inspiration: set() for inspiration in BiologicalInspiration
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        # Load database if it exists
        self.load_database()
        logger.info(f"Initialized biological reference database with {len(self.references)} models")
    
    def load_database(self) -> bool:
        """
        Load reference models from database file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.database_path):
            logger.info(f"Database file not found at {self.database_path}, starting with empty database")
            return False
            
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                
            for ref_id, ref_data in data.items():
                # Convert inspiration type string to enum
                inspiration_str = ref_data.get("inspiration_type")
                if inspiration_str:
                    try:
                        ref_data["inspiration_type"] = BiologicalInspiration(inspiration_str)
                    except ValueError:
                        logger.warning(f"Invalid inspiration type: {inspiration_str}")
                        continue
                
                # Create reference object
                reference = BiologicalReference(**ref_data)
                self.references[ref_id] = reference
                
                # Update indexes
                self._index_reference(ref_id, reference)
                
            logger.info(f"Loaded {len(self.references)} reference models from database")
            return True
            
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            return False
    
    def save_database(self) -> bool:
        """
        Save reference models to database file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert references to serializable format
            serializable_data = {}
            for ref_id, reference in self.references.items():
                ref_dict = asdict(reference)
                # Convert enum to string for serialization
                if "inspiration_type" in ref_dict and isinstance(ref_dict["inspiration_type"], BiologicalInspiration):
                    ref_dict["inspiration_type"] = ref_dict["inspiration_type"].value
                serializable_data[ref_id] = ref_dict
            
            with open(self.database_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logger.info(f"Saved {len(self.references)} reference models to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
            return False
    
    def add_reference(self, ref_id: str, reference: BiologicalReference) -> bool:
        """
        Add a reference model to the database.
        
        Args:
            ref_id: Unique identifier for the reference
            reference: The biological reference model
            
        Returns:
            True if added successfully, False otherwise
        """
        if ref_id in self.references:
            logger.warning(f"Reference ID {ref_id} already exists")
            return False
            
        self.references[ref_id] = reference
        self._index_reference(ref_id, reference)
        
        # Auto-save after adding
        self.save_database()
        return True
    
    def get_reference(self, ref_id: str) -> Optional[BiologicalReference]:
        """
        Get a reference model by ID.
        
        Args:
            ref_id: ID of the reference model
            
        Returns:
            BiologicalReference if found, None otherwise
        """
        return self.references.get(ref_id)
    
    def remove_reference(self, ref_id: str) -> bool:
        """
        Remove a reference model from the database.
        
        Args:
            ref_id: ID of the reference model
            
        Returns:
            True if removed successfully, False otherwise
        """
        if ref_id not in self.references:
            logger.warning(f"Reference ID {ref_id} not found")
            return False
            
        # Remove from indexes
        reference = self.references[ref_id]
        self._remove_from_indexes(ref_id, reference)
        
        # Remove from references
        del self.references[ref_id]
        
        # Auto-save after removing
        self.save_database()
        return True
    
    def find_by_feature(self, feature: str) -> List[Tuple[str, BiologicalReference]]:
        """
        Find reference models by feature.
        
        Args:
            feature: Feature to search for
            
        Returns:
            List of (ref_id, reference) tuples
        """
        if feature not in self.feature_index:
            return []
            
        return [(ref_id, self.references[ref_id]) for ref_id in self.feature_index[feature]]
    
    def find_by_inspiration(self, inspiration: BiologicalInspiration) -> List[Tuple[str, BiologicalReference]]:
        """
        Find reference models by inspiration type.
        
        Args:
            inspiration: Inspiration type to search for
            
        Returns:
            List of (ref_id, reference) tuples
        """
        if inspiration not in self.inspiration_index:
            return []
            
        return [(ref_id, self.references[ref_id]) for ref_id in self.inspiration_index[inspiration]]
    
    def find_by_performance(self, metric: str, min_value: float, max_value: float) -> List[Tuple[str, BiologicalReference]]:
        """
        Find reference models by performance metric range.
        
        Args:
            metric: Performance metric to search for
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            List of (ref_id, reference) tuples
        """
        results = []
        for ref_id, reference in self.references.items():
            if metric in reference.performance_metrics:
                value = reference.performance_metrics[metric]
                if min_value <= value <= max_value:
                    results.append((ref_id, reference))
        return results
    
    def _index_reference(self, ref_id: str, reference: BiologicalReference) -> None:
        """
        Index a reference model for fast lookup.
        
        Args:
            ref_id: ID of the reference model
            reference: The reference model to index
        """
        # Index by features
        for feature in reference.key_features:
            if feature not in self.feature_index:
                self.feature_index[feature] = set()
            self.feature_index[feature].add(ref_id)
        
        # Index by inspiration type
        if reference.inspiration_type in self.inspiration_index:
            self.inspiration_index[reference.inspiration_type].add(ref_id)
    
    def _remove_from_indexes(self, ref_id: str, reference: BiologicalReference) -> None:
        """
        Remove a reference model from indexes.
        
        Args:
            ref_id: ID of the reference model
            reference: The reference model to remove
        """
        # Remove from feature index
        for feature in reference.key_features:
            if feature in self.feature_index and ref_id in self.feature_index[feature]:
                self.feature_index[feature].remove(ref_id)
                # Clean up empty sets
                if not self.feature_index[feature]:
                    del self.feature_index[feature]
        
        # Remove from inspiration index
        if reference.inspiration_type in self.inspiration_index and ref_id in self.inspiration_index[reference.inspiration_type]:
            self.inspiration_index[reference.inspiration_type].remove(ref_id)