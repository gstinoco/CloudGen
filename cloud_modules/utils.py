"""
Utility Module - Geometric Calculations and Helper Functions

This module provides essential geometric utilities and helper functions used across
the Cloud Generation system. It encapsulates core mathematical logic to ensure
consistency and reusability.

Core Functionality:
1. Cloud Size Calculation: Adaptive spacing determination based on geometry.
2. Contour Processing: Utilities for closing and validating geometric contours.
3. Boundary Refinement: Dynamic tolerance calculation for node classification.

Key Features:
- Robust Euclidean distance calculations using NumPy
- Intelligent cloud size adaptation for varying region scales
- Consistent handling of open/closed contours

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- NumPy
- Logging
"""

import numpy as np
import logging
from .config import CLOUD_FACTORS

def calculate_cloud_size(region_points: list[tuple[float, float]]) -> float:
    """
    Calculate adaptive cloud size based on region geometry characteristics.
    
    This function analyzes the spacing between consecutive boundary points to determine
    an optimal cloud size for point generation. The cloud size is calculated as a
    fraction of the average distance between boundary points, ensuring appropriate
    point density for the given geometry.
    
    Args:
        region_points (list): List of (x, y) coordinate tuples defining the region boundary
    
    Returns:
        float: Calculated cloud size (spacing between points) based on geometry analysis.
               Returns default cloud size if calculation fails or insufficient points.
    
    Algorithm:
        1. Calculate distances between consecutive boundary points
        2. Compute average distance between points
        3. Apply adaptive factor to determine optimal cloud size
        4. Return default size if insufficient data or calculation error
    
    Note:
        Uses CLOUD_FACTORS["adaptive_factor"] (0.15) to scale the average distance
        and CLOUD_FACTORS["default_cloud_size"] (0.05) as fallback value.
    """
    try:
        if len(region_points) < 2:
            return CLOUD_FACTORS["default_cloud_size"]
        
        distances = []
        for i in range(len(region_points) - 1):
            x1, y1 = region_points[i]
            x2, y2 = region_points[i + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            cloud_size = avg_distance * CLOUD_FACTORS["adaptive_factor"]
            return cloud_size
        else:
            return CLOUD_FACTORS["default_cloud_size"]
            
    except Exception as e:
        logging.error(f"Error calculating cloud size: {e}")
        return CLOUD_FACTORS["default_cloud_size"]

def create_closed_contour(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Ensure a contour is properly closed for geometric operations.
    
    This function validates that a contour forms a closed polygon by checking if the
    first and last points are identical. If not, it adds the first point at the end
    to create a closed contour, which is required for proper polygon operations.
    
    Args:
        points (list): List of (x, y) coordinate tuples defining the contour
    
    Returns:
        list: Closed contour with first point repeated at the end if necessary
    
    Raises:
        ValueError: If fewer than 3 points are provided (insufficient for a polygon)
    
    Note:
        A minimum of 3 points is required to form a valid polygon. The function
        ensures geometric consistency for subsequent Shapely polygon operations.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are needed to create a contour")
    
    if points[0] != points[-1]:
        return points + [points[0]]
    else:
        return points

def calculate_dynamic_boundary_refinement(points: np.ndarray, cloud_size: float = None) -> float:
    """
    Calculate dynamic boundary refinement based on cloud density.
    
    This function calculates an appropriate boundary tolerance based on:
    1. Average distance between nearest neighbors (point density)
    2. Cloud size parameter if provided
    3. Domain size for scaling
    
    Args:
        points (numpy.ndarray): Array of point coordinates
        cloud_size (float, optional): Cloud size parameter used in generation
    
    Returns:
        float: Dynamic boundary refinement value
    """
    if len(points) < 2:
        return CLOUD_FACTORS["boundary_refinement"]  # Fallback to default
    
    try:
        # Convert to numpy array for calculations
        points_array = np.array(points, dtype=float)
        
        # Calculate domain dimensions
        domain_width = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
        domain_height = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
        domain_area = domain_width * domain_height
        
        # Method 1: Based on average nearest neighbor distance
        
        # Sample a subset of points for efficiency (max 500 points)
        sample_size = min(500, len(points_array))
        sample_indices = np.random.choice(len(points_array), sample_size, replace=False)
        sample_points = points_array[sample_indices]
        
        # Calculate distances between all sample points using NumPy broadcasting
        # Shape: (sample_size, 1, 2) - (1, sample_size, 2) -> (sample_size, sample_size, 2)
        diff = sample_points[:, np.newaxis, :] - sample_points[np.newaxis, :, :]
        # Calculate Euclidean distance
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        
        # For each point, find the distance to its nearest neighbor (excluding itself)
        np.fill_diagonal(distances, np.inf)  # Exclude self-distances
        nearest_distances = np.min(distances, axis=1)
        
        # Calculate average nearest neighbor distance
        avg_nearest_distance = np.mean(nearest_distances)
        
        # Method 2: Based on point density
        point_density = len(points_array) / domain_area if domain_area > 0 else 1
        density_based_refinement = 1.0 / np.sqrt(point_density) if point_density > 0 else 0.01
        
        # Method 3: Based on cloud_size if provided
        cloud_size_based_refinement = cloud_size * 0.5 if cloud_size else None
        
        # Combine methods with weights
        refinement_candidates = []
        
        # Primary: Average nearest neighbor distance (scaled down)
        # We want to be careful not to overestimate this
        refinement_candidates.append(avg_nearest_distance * 0.3)
        
        # Secondary: Density-based calculation
        refinement_candidates.append(density_based_refinement * 0.1)
        
        # Tertiary: Cloud size based (if available) - this is the most reliable metric for Regular clouds
        if cloud_size_based_refinement:
            # We want to be extremely strict with boundary distance. Since boundary nodes are placed
            # exactly on the contour, distance should be close to 0. Use 5% of cloud_size to prevent
            # interior nodes (which are typically ~0.5 * cloud_size away) from being marked as boundary.
            refinement_candidates.append(cloud_size * 0.05)
        
        # Use a weighted approach or median, but favor the cloud_size if available
        if cloud_size:
            dynamic_refinement = cloud_size * 0.05
        else:
             # Fallback to median of candidates
             dynamic_refinement = np.median(refinement_candidates)
        
        # Apply reasonable bounds
        min_refinement = 0.0001  # Minimum threshold
        if cloud_size:
            # Ensure we don't go too low, but allow our 0.05*cloud_size target
            min_refinement = max(min_refinement, cloud_size * 0.01)
            
        max_refinement = min(domain_width, domain_height) * 0.02
        
        # If cloud_size is provided, allow a larger refinement relative to it
        if cloud_size:
            max_refinement = max(max_refinement, cloud_size * 1.0)
        
        dynamic_refinement = np.clip(dynamic_refinement, min_refinement, max_refinement)
        
        logging.info(f"Dynamic boundary refinement calculation:")
        logging.info(f"  - Average nearest neighbor distance: {avg_nearest_distance:.6f}")
        logging.info(f"  - Point density: {point_density:.2f} points/unit²")
        logging.info(f"  - Domain size: {domain_width:.3f} x {domain_height:.3f}")
        logging.info(f"  - Cloud size parameter: {cloud_size}")
        logging.info(f"  - Calculated refinement: {dynamic_refinement:.6f}")
        logging.info(f"  - Default refinement: {CLOUD_FACTORS['boundary_refinement']:.6f}")
        
        return dynamic_refinement
        
    except Exception as e:
        logging.warning(f"Failed to calculate dynamic boundary refinement: {e}")
        logging.info(f"Falling back to default boundary refinement: {CLOUD_FACTORS['boundary_refinement']}")
        return CLOUD_FACTORS["boundary_refinement"]
