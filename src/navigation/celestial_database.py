"""
Celestial Database Integration for Star Tracker.

Provides functionality to load and manage star catalog data from external sources.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.navigation.star_tracker import StarCatalogEntry

# Configure logger
logger = logging.getLogger(__name__)


class CelestialDatabase:
    """Interface for loading and managing celestial object databases."""
    
    def __init__(self, database_path: str = "/Users/yessine/Oblivion/data/celestial"):
        """
        Initialize celestial database.
        
        Args:
            database_path: Path to celestial database files
        """
        self.database_path = database_path
        self.catalogs: Dict[str, List[StarCatalogEntry]] = {}
        
        # Create database directory if it doesn't exist
        os.makedirs(self.database_path, exist_ok=True)
        
        logger.info(f"Initialized celestial database at {self.database_path}")
    
    def load_catalog(self, catalog_name: str) -> List[StarCatalogEntry]:
        """
        Load a star catalog by name.
        
        Args:
            catalog_name: Name of the catalog to load
            
        Returns:
            List of star catalog entries
        """
        # Check if already loaded
        if catalog_name in self.catalogs:
            return self.catalogs[catalog_name]
        
        # Determine file path
        catalog_path = os.path.join(self.database_path, f"{catalog_name}.json")
        
        # Check if file exists
        if not os.path.exists(catalog_path):
            logger.warning(f"Catalog file not found: {catalog_path}")
            return []
        
        try:
            # Load catalog from file
            with open(catalog_path, 'r') as f:
                star_data = json.load(f)
            
            # Convert to StarCatalogEntry objects
            entries = []
            for i, star in enumerate(star_data):
                entry = StarCatalogEntry(
                    id=star.get("id", i),
                    name=star.get("name", f"Star-{i}"),
                    right_ascension=star.get("ra", 0.0),
                    declination=star.get("dec", 0.0),
                    magnitude=star.get("magnitude", 0.0),
                    spectral_class=star.get("spectral_class", "")
                )
                entries.append(entry)
            
            # Store in cache
            self.catalogs[catalog_name] = entries
            
            logger.info(f"Loaded {len(entries)} stars from catalog '{catalog_name}'")
            return entries
            
        except Exception as e:
            logger.error(f"Error loading catalog '{catalog_name}': {str(e)}")
            return []
    
    def save_catalog(self, catalog_name: str, entries: List[StarCatalogEntry]) -> bool:
        """
        Save a star catalog to file.
        
        Args:
            catalog_name: Name of the catalog to save
            entries: List of star catalog entries
            
        Returns:
            Success status
        """
        # Determine file path
        catalog_path = os.path.join(self.database_path, f"{catalog_name}.json")
        
        try:
            # Convert entries to dictionaries
            star_data = []
            for entry in entries:
                star_data.append({
                    "id": entry.id,
                    "name": entry.name,
                    "ra": entry.right_ascension,
                    "dec": entry.declination,
                    "magnitude": entry.magnitude,
                    "spectral_class": entry.spectral_class
                })
            
            # Save to file
            with open(catalog_path, 'w') as f:
                json.dump(star_data, f, indent=2)
            
            # Update cache
            self.catalogs[catalog_name] = entries
            
            logger.info(f"Saved {len(entries)} stars to catalog '{catalog_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error saving catalog '{catalog_name}': {str(e)}")
            return False
    
    def create_default_catalog(self, catalog_name: str = "default", num_stars: int = 1000) -> List[StarCatalogEntry]:
        """
        Create a default star catalog with the brightest stars and random additions.
        
        Args:
            catalog_name: Name of the catalog to create
            num_stars: Total number of stars to include
            
        Returns:
            List of star catalog entries
        """
        # Create a basic catalog with the brightest stars
        bright_stars = [
            StarCatalogEntry(0, "Sirius", 101.287, -16.716, -1.46, "A1V"),
            StarCatalogEntry(1, "Canopus", 95.987, -52.696, -0.72, "F0II"),
            StarCatalogEntry(2, "Alpha Centauri", 219.902, -60.834, -0.27, "G2V"),
            StarCatalogEntry(3, "Arcturus", 213.915, 19.182, -0.05, "K1.5III"),
            StarCatalogEntry(4, "Vega", 279.234, 38.783, 0.03, "A0V"),
            StarCatalogEntry(5, "Capella", 79.172, 45.998, 0.08, "G5III"),
            StarCatalogEntry(6, "Rigel", 78.634, -8.202, 0.13, "B8Ia"),
            StarCatalogEntry(7, "Procyon", 114.825, 5.225, 0.34, "F5IV-V"),
            StarCatalogEntry(8, "Betelgeuse", 88.793, 7.407, 0.45, "M1-2Ia-Iab"),
            StarCatalogEntry(9, "Achernar", 24.429, -57.237, 0.46, "B6V"),
        ]
        
        # Generate additional stars to reach desired size
        entries = list(bright_stars)
        num_additional = min(num_stars - len(bright_stars), 10000)
        
        for i in range(num_additional):
            # Generate random star data
            star_id = i + len(bright_stars)
            ra = np.random.uniform(0, 360)
            dec = np.random.uniform(-90, 90)
            mag = np.random.uniform(1.0, 6.0)
            
            # Create star entry
            star = StarCatalogEntry(
                id=star_id,
                name=f"Star-{star_id}",
                right_ascension=ra,
                declination=dec,
                magnitude=mag
            )
            
            entries.append(star)
        
        # Save the catalog
        self.save_catalog(catalog_name, entries)
        
        return entries