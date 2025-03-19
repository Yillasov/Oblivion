"""
Visual signature matching algorithms for active camouflage systems.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from enum import Enum
from dataclasses import dataclass
import cv2


class MatchingAlgorithm(Enum):
    """Visual signature matching algorithms."""
    COLOR_HISTOGRAM = "color_histogram"
    TEXTURE_ANALYSIS = "texture_analysis"
    EDGE_DETECTION = "edge_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    ADAPTIVE_BLENDING = "adaptive_blending"


@dataclass
class VisualSignature:
    """Visual signature data structure."""
    dominant_colors: List[Tuple[int, int, int]]  # RGB colors
    color_distribution: np.ndarray  # Histogram
    texture_features: np.ndarray
    edge_density: float
    pattern_complexity: float
    light_levels: float  # 0.0-1.0
    contrast_ratio: float


class VisualSignatureMatcher:
    """
    Visual signature matching for active camouflage systems.
    Analyzes environment and generates matching visual patterns.
    """
    
    def __init__(self, 
                resolution: Tuple[int, int] = (640, 480),
                matching_algorithm: MatchingAlgorithm = MatchingAlgorithm.ADAPTIVE_BLENDING):
        """
        Initialize visual signature matcher.
        
        Args:
            resolution: Output resolution (width, height)
            matching_algorithm: Algorithm to use for matching
        """
        self.resolution = resolution
        self.matching_algorithm = matching_algorithm
        self.current_signature = None
        self.reference_signatures = {}
        
        # Initialize algorithm-specific parameters
        self.params = {
            "color_bins": 32,  # Number of bins for color histograms
            "texture_scale": 1.0,  # Scale for texture analysis
            "edge_threshold": 50,  # Threshold for edge detection
            "adaptation_rate": 0.3,  # Rate of adaptation to new environments
            "blend_factor": 0.7,  # Blending factor for transitions
        }
    
    def analyze_environment(self, image: np.ndarray) -> VisualSignature:
        """
        Analyze environment image to extract visual signature.
        
        Args:
            image: RGB image of environment
            
        Returns:
            Visual signature of the environment
        """
        # Resize image if needed
        if image.shape[:2] != (self.resolution[1], self.resolution[0]):
            image = cv2.resize(image, self.resolution)
            
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image)
        
        # Calculate color distribution (histogram)
        color_dist = self._calculate_color_histogram(image)
        
        # Extract texture features
        texture_features = self._extract_texture_features(image)
        
        # Calculate edge density
        edge_density = self._calculate_edge_density(image)
        
        # Calculate pattern complexity
        pattern_complexity = self._calculate_pattern_complexity(texture_features, edge_density)
        
        # Calculate light levels
        light_levels = float(np.mean(image) / 255.0)
        
        # Calculate contrast ratio
        contrast_ratio = self._calculate_contrast_ratio(image)
        
        # Create and store visual signature
        signature = VisualSignature(
            dominant_colors=dominant_colors,
            color_distribution=color_dist,
            texture_features=texture_features,
            edge_density=edge_density,
            pattern_complexity=pattern_complexity,
            light_levels=light_levels,
            contrast_ratio=contrast_ratio
        )
        
        self.current_signature = signature
        return signature
    
    def generate_camouflage_pattern(self, 
                                   signature: Optional[VisualSignature] = None, 
                                   environment_type: Optional[str] = None) -> np.ndarray:
        """
        Generate camouflage pattern based on visual signature.
        
        Args:
            signature: Visual signature to match (uses current if None)
            environment_type: Environment type for predefined patterns
            
        Returns:
            RGB image of generated camouflage pattern
        """
        # Use current signature if none provided
        if signature is None:
            if self.current_signature is None:
                # If no signature available, use predefined pattern
                return self._generate_predefined_pattern(environment_type)
            signature = self.current_signature
            
        # Select algorithm based on configuration
        if self.matching_algorithm == MatchingAlgorithm.COLOR_HISTOGRAM:
            return self._generate_color_based_pattern(signature)
        elif self.matching_algorithm == MatchingAlgorithm.TEXTURE_ANALYSIS:
            return self._generate_texture_based_pattern(signature)
        elif self.matching_algorithm == MatchingAlgorithm.EDGE_DETECTION:
            return self._generate_edge_based_pattern(signature)
        elif self.matching_algorithm == MatchingAlgorithm.PATTERN_RECOGNITION:
            return self._generate_pattern_based_camouflage(signature)
        else:  # Default to adaptive blending
            return self._generate_adaptive_blend_pattern(signature, environment_type)
    
    def blend_patterns(self, 
                      current_pattern: np.ndarray, 
                      new_pattern: np.ndarray, 
                      blend_factor: Optional[float] = None) -> np.ndarray:
        """
        Blend between two camouflage patterns for smooth transitions.
        
        Args:
            current_pattern: Current camouflage pattern
            new_pattern: New camouflage pattern
            blend_factor: Blending factor (0.0-1.0, None for default)
            
        Returns:
            Blended camouflage pattern
        """
        if blend_factor is None:
            blend_factor = self.params["blend_factor"]
            
        # Ensure patterns are the same size
        if current_pattern.shape != new_pattern.shape:
            new_pattern = cv2.resize(new_pattern, 
                                    (current_pattern.shape[1], current_pattern.shape[0]))
            
        # Linear blending - ensure inputs are the right type
        alpha = float(1.0 - (blend_factor if blend_factor is not None else 0.0))
        beta = float(blend_factor if blend_factor is not None else 0.0)
        gamma = 0.0
        blended = cv2.addWeighted(current_pattern.astype(np.uint8), alpha, 
                                 new_pattern.astype(np.uint8), beta, gamma)
        return blended
    
    def store_reference_signature(self, environment_type: str, signature: VisualSignature) -> None:
        """
        Store a reference signature for an environment type.
        
        Args:
            environment_type: Environment type identifier
            signature: Visual signature to store
        """
        self.reference_signatures[environment_type] = signature
    
    def get_reference_signature(self, environment_type: str) -> Optional[VisualSignature]:
        """
        Get a stored reference signature.
        
        Args:
            environment_type: Environment type identifier
            
        Returns:
            Stored visual signature or None if not found
        """
        return self.reference_signatures.get(environment_type)
    
    def calculate_signature_similarity(self, 
                                      signature1: VisualSignature, 
                                      signature2: VisualSignature) -> float:
        """
        Calculate similarity between two visual signatures.
        
        Args:
            signature1: First visual signature
            signature2: Second visual signature
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Compare color distributions using histogram intersection
        color_similarity = cv2.compareHist(
            signature1.color_distribution.astype(np.float32),
            signature2.color_distribution.astype(np.float32),
            cv2.HISTCMP_INTERSECT
        )
        # Normalize color similarity
        color_similarity = color_similarity / np.sum(signature1.color_distribution)
        
        # Compare texture features using cosine similarity
        texture_similarity = np.dot(signature1.texture_features, signature2.texture_features) / (
            np.linalg.norm(signature1.texture_features) * np.linalg.norm(signature2.texture_features)
        )
        
        # Compare edge density
        edge_similarity = 1.0 - abs(signature1.edge_density - signature2.edge_density)
        
        # Compare pattern complexity
        pattern_similarity = 1.0 - abs(signature1.pattern_complexity - signature2.pattern_complexity)
        
        # Calculate weighted similarity
        similarity = (
            0.4 * color_similarity +
            0.3 * texture_similarity +
            0.15 * edge_similarity +
            0.15 * pattern_similarity
        )
        
        return float(similarity)
    
    def _extract_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        # Reshape image for k-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Use k-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        # Fix the kmeans call by providing proper parameters
        flags = cv2.KMEANS_RANDOM_CENTERS
        attempts = 10
        # Create a proper output array for labels
        labels = np.zeros(pixels.shape[0], dtype=np.int32)
        _, labels, centers = cv2.kmeans(pixels, num_colors, labels, criteria, attempts, flags)
        
        # Convert centers to RGB tuples
        dominant_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in centers]
        
        return dominant_colors
    
    def _calculate_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Calculate color histogram."""
        bins = self.params["color_bins"]
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], 
                           [0, 256, 0, 256, 0, 256])
        return hist
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using GLCM."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM features (simplified)
        # In a real implementation, use proper GLCM calculation
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        mag = cv2.magnitude(dx, dy)
        
        # Calculate basic statistics as texture features
        features = np.array([
            np.mean(mag),
            np.std(mag),
            np.percentile(mag, 25),
            np.percentile(mag, 75),
            np.max(mag) - np.min(mag)
        ])
        
        return features
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, self.params["edge_threshold"], self.params["edge_threshold"] * 2)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return float(edge_density)
    
    def _calculate_pattern_complexity(self, texture_features: np.ndarray, edge_density: float) -> float:
        """Calculate pattern complexity."""
        # Combine texture variance and edge density
        texture_variance = texture_features[1]  # Standard deviation
        complexity = (0.7 * (texture_variance / 50.0) + 0.3 * edge_density)
        
        # Normalize to 0.0-1.0
        complexity = min(1.0, max(0.0, complexity))
        
        return float(complexity)
    
    def _calculate_contrast_ratio(self, image: np.ndarray) -> float:
        """Calculate contrast ratio."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast as (max - min) / (max + min)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        if max_val + min_val == 0:
            return 0.0
            
        contrast = (max_val - min_val) / (max_val + min_val)
        return float(contrast)
    
    def _generate_predefined_pattern(self, environment_type: Optional[str]) -> np.ndarray:
        """Generate predefined pattern for environment type."""
        # Create blank canvas
        pattern = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Default to urban if environment_type is None
        if environment_type is None:
            environment_type = "urban"
        
        if environment_type == "desert":
            # Desert pattern (sandy colors with some texture)
            base_color = np.array([210, 180, 140])
            pattern[:] = base_color
            
            # Add some texture
            noise = np.random.randint(0, 30, (self.resolution[1], self.resolution[0], 3))
            pattern = np.clip(pattern + noise - 15, 0, 255).astype(np.uint8)
            
        elif environment_type == "forest":
            # Forest pattern (green with darker patches)
            base_color = np.array([34, 139, 34])
            pattern[:] = base_color
            
            # Create random dark patches
            for _ in range(50):
                x = np.random.randint(0, self.resolution[0])
                y = np.random.randint(0, self.resolution[1])
                radius = np.random.randint(10, 50)
                color = np.array([25, 100, 25])
                cv2.circle(pattern, (x, y), radius, color.tolist(), -1)
                
        elif environment_type == "urban":
            # Urban pattern (gray with geometric shapes)
            base_color = np.array([128, 128, 128])
            pattern[:] = base_color
            
            # Add geometric shapes
            for _ in range(30):
                x = np.random.randint(0, self.resolution[0])
                y = np.random.randint(0, self.resolution[1])
                w = np.random.randint(20, 80)
                h = np.random.randint(20, 80)
                color = np.random.randint(100, 160, 3)
                cv2.rectangle(pattern, (x, y), (x+w, y+h), color.tolist(), -1)
                
        elif environment_type == "ocean":
            # Ocean pattern (blue with wave-like texture)
            base_color = np.array([0, 105, 148])
            pattern[:] = base_color
            
            # Create wave-like texture
            for y in range(self.resolution[1]):
                wave = int(10 * np.sin(y / 20.0))
                for x in range(self.resolution[0]):
                    if (x + wave) % 40 < 20:
                        pattern[y, x] = np.array([0, 120, 160])
                        
        elif environment_type == "night":
            # Night pattern (dark with subtle variation)
            base_color = np.array([25, 25, 25])
            pattern[:] = base_color
            
            # Add subtle noise
            noise = np.random.randint(0, 10, (self.resolution[1], self.resolution[0], 3))
            pattern = np.clip(pattern + noise - 5, 0, 255).astype(np.uint8)
        
        else:
            # Default pattern (gray)
            pattern[:] = np.array([128, 128, 128])
        
        return pattern


# Move these methods inside the VisualSignatureMatcher class
# Delete these standalone functions at the bottom of the file

# Inside the VisualSignatureMatcher class, add these methods:
    def _generate_color_based_pattern(self, signature: VisualSignature) -> np.ndarray:
        """Generate pattern based on color histogram."""
        # Create blank canvas
        pattern = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Use dominant colors to create a pattern
        dominant_colors = signature.dominant_colors
        
        # Create regions with dominant colors
        num_regions = min(len(dominant_colors), 5)
        region_height = self.resolution[1] // num_regions
        
        for i in range(num_regions):
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < num_regions - 1 else self.resolution[1]
            
            # Fill region with dominant color
            pattern[y_start:y_end, :] = dominant_colors[i]
            
            # Add some noise for texture
            noise = np.random.randint(0, 20, (y_end - y_start, self.resolution[0], 3))
            pattern[y_start:y_end, :] = np.clip(pattern[y_start:y_end, :] + noise - 10, 0, 255)
        
        return pattern
    
    def _generate_texture_based_pattern(self, signature: VisualSignature) -> np.ndarray:
        """Generate pattern based on texture features."""
        # Create base pattern with primary color
        pattern = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Use first dominant color as base
        if signature.dominant_colors:
            pattern[:] = signature.dominant_colors[0]
        else:
            pattern[:] = (128, 128, 128)  # Default gray
        
        # Apply texture based on edge density and pattern complexity
        texture_scale = int(signature.pattern_complexity * 50) + 10
        
        # Create perlin-like noise (simplified)
        for y in range(self.resolution[1]):
            for x in range(self.resolution[0]):
                noise_val = (np.sin(x / texture_scale) + np.cos(y / texture_scale)) * 20
                pattern[y, x] = np.clip(pattern[y, x] + noise_val, 0, 255)
        
        return pattern
    
    def _generate_edge_based_pattern(self, signature: VisualSignature) -> np.ndarray:
        """Generate pattern based on edge features."""
        # Create base pattern
        pattern = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Use dominant colors
        colors = signature.dominant_colors
        if not colors:
            colors = [(128, 128, 128)]
        
        # Base color
        pattern[:] = colors[0]
        
        # Add edge-like patterns based on edge density
        num_edges = int(signature.edge_density * 100) + 10
        
        for _ in range(num_edges):
            # Random line parameters
            pt1 = (np.random.randint(0, self.resolution[0]), 
                   np.random.randint(0, self.resolution[1]))
            pt2 = (np.random.randint(0, self.resolution[0]), 
                   np.random.randint(0, self.resolution[1]))
            
            # Random color from dominant colors
            color_idx = np.random.randint(0, len(colors))
            color = colors[color_idx]
            
            # Draw line
            cv2.line(pattern, pt1, pt2, color, thickness=np.random.randint(1, 5))
        
        return pattern
    
    def _generate_pattern_based_camouflage(self, signature: VisualSignature) -> np.ndarray:
        """Generate pattern based on pattern recognition."""
        # Create base pattern
        pattern = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Use dominant colors
        colors = signature.dominant_colors
        if not colors:
            colors = [(128, 128, 128)]
        
        # Base color
        pattern[:] = colors[0]
        
        # Generate pattern based on complexity
        complexity = signature.pattern_complexity
        
        if complexity < 0.3:
            # Simple pattern - large blocks
            block_size = 80
        elif complexity < 0.7:
            # Medium pattern - medium blocks
            block_size = 40
        else:
            # Complex pattern - small blocks
            block_size = 20
        
        # Create block pattern
        for y in range(0, self.resolution[1], block_size):
            for x in range(0, self.resolution[0], block_size):
                # Random color from dominant colors
                color_idx = np.random.randint(0, len(colors))
                color = colors[color_idx]
                
                # Draw block
                end_y = min(y + block_size, self.resolution[1])
                end_x = min(x + block_size, self.resolution[0])
                pattern[y:end_y, x:end_x] = color
        
        return pattern
    
    def _generate_adaptive_blend_pattern(self, signature: VisualSignature, 
                                       environment_type: Optional[str]) -> np.ndarray:
        """Generate pattern using adaptive blending."""
        # Generate base pattern using color-based approach
        color_pattern = self._generate_color_based_pattern(signature)
        
        # Generate texture pattern
        texture_pattern = self._generate_texture_based_pattern(signature)
        
        # Generate predefined pattern if environment type is provided
        if environment_type:
            predefined_pattern = self._generate_predefined_pattern(environment_type)
            
            # Blend all three patterns - fix the addWeighted calls
            blend1 = cv2.addWeighted(color_pattern, 0.5, texture_pattern, 0.5, 0.0)
            final_pattern = cv2.addWeighted(blend1, 0.7, predefined_pattern, 0.3, 0.0)
            return final_pattern
        else:
            # Blend color and texture patterns - fix the addWeighted call
            return cv2.addWeighted(color_pattern, 0.6, texture_pattern, 0.4, 0.0)