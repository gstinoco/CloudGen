"""
Classification Module - Node Type Identification

This module is responsible for classifying generated nodes as either 'boundary'
or 'interior'. It uses advanced geometric operations and spatial indexing to
ensure accurate and efficient classification, even for complex geometries.

Core Functionality:
1. Geometric Analysis: Uses Shapely to determine node proximity to boundaries.
2. Spatial Indexing: Implements STRtree (R-tree) for O(log N) query performance.
3. Dynamic Tolerance: Adapts classification thresholds based on local point density.
4. Region Awareness: Handles special boundary conditions for interior regions/holes.

Key Features:
- High-precision classification using vector geometry
- Optimized performance for large point clouds using spatial trees
- Robust fallback mechanisms for edge cases
- Support for 'inside_regions' logic where holes define interior boundaries

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- NumPy
- Shapely
- Logging
"""

import numpy as np
import logging
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
from .utils import calculate_dynamic_boundary_refinement

def classify_nodes(points: np.ndarray, regions_list: list[int], original_regions_contours: list[list[tuple[float, float]]] = None, cloud_size: float = None, inside_regions: bool = False) -> list[str]:
    """
    Classify nodes as boundary or interior using Shapely contours for precise geometric operations.
    This approach creates LineString/Polygon objects from contours for accurate distance calculations.
    
    Optimized with STRtree (R-tree spatial index) for O(log N) nearest-neighbor queries,
    significantly reducing classification time for large point clouds.
    
    Args:
        points (numpy.ndarray): Array of point coordinates
        regions_list (list): List of region IDs for each point
        original_regions_contours (list): List of original contour points for each region
        cloud_size (float, optional): Cloud size parameter used in generation for dynamic boundary refinement
        inside_regions (bool, optional): If True, applies region-specific boundary logic (holes are not boundaries for region 1)
    
    Returns:
        list: Node classifications ("boundary" or "interior")
    """
    classifications = []
    
    # Convert to numpy arrays for better performance
    points_array = np.array(points, dtype=float)
    regions_array = np.array(regions_list)
    
    # Calculate dynamic boundary refinement based on point density
    boundary_tolerance = calculate_dynamic_boundary_refinement(points_array, cloud_size)
    
    # Pre-calculate global domain boundaries (only once)
    global_min_x, global_max_x = np.min(points_array[:, 0]), np.max(points_array[:, 0])
    global_min_y, global_max_y = np.min(points_array[:, 1]), np.max(points_array[:, 1])
    
    # Create Shapely geometries from contours
    shapely_contours = []
    valid_geometries = [] # For STRtree
    geom_id_to_index = {} # Map geometry ID to contour index
    
    if original_regions_contours:
        for idx, contour_points in enumerate(original_regions_contours):
            if contour_points and len(contour_points) >= 3:
                try:
                    # Convert contour points to valid coordinates
                    valid_coords = []
                    for cp in contour_points:
                        if len(cp) >= 2:
                            valid_coords.append((float(cp[0]), float(cp[1])))
                    
                    if len(valid_coords) >= 3:
                        # Try to create a closed polygon first
                        if valid_coords[0] != valid_coords[-1]:
                            valid_coords.append(valid_coords[0])  # Close the polygon
                        
                        # Create polygon if possible, otherwise LineString
                        geom = None
                        if len(valid_coords) >= 4:  # At least 3 unique points + closure
                            try:
                                polygon = Polygon(valid_coords)
                                if polygon.is_valid:
                                    geom = polygon.boundary  # Use boundary for distance
                                else:
                                    geom = LineString(valid_coords)
                            except:
                                geom = LineString(valid_coords)
                        else:
                            geom = LineString(valid_coords)
                        
                        shapely_contours.append(geom)
                        if geom:
                            valid_geometries.append(geom)
                            geom_id_to_index[id(geom)] = idx
                    else:
                        shapely_contours.append(None)
                except Exception as e:
                    logging.warning(f"Failed to create Shapely geometry for contour: {e}")
                    shapely_contours.append(None)
            else:
                shapely_contours.append(None)
    
    # Build STRtree for efficient spatial queries
    spatial_index = None
    if valid_geometries:
        try:
            spatial_index = STRtree(valid_geometries)
        except Exception as e:
            logging.warning(f"Failed to build STRtree: {e}")

    # Create global domain boundary as a rectangle
    domain_boundary = None
    try:
        domain_coords = [
            (global_min_x, global_min_y),
            (global_max_x, global_min_y),
            (global_max_x, global_max_y),
            (global_min_x, global_max_y),
            (global_min_x, global_min_y)
        ]
        domain_boundary = LineString(domain_coords)
    except:
        domain_boundary = None
    
    # Classify each point using Shapely geometric operations
    for i in range(len(points_array)):
        x, y = points_array[i]
        region_id = regions_array[i]
        is_boundary = False
        
        try:
            point_geom = Point(x, y)
            min_distance_to_contour = float('inf')
            
            # Primary method: Check distance to Shapely contours
            if shapely_contours:
                # Optimized: Use STRtree if available to find candidate contours
                if spatial_index:
                    # query returns candidates (geometries) that *might* be near
                    candidates = spatial_index.query(point_geom)
                    
                    # Handle different STRtree API versions
                    final_candidates = []
                    
                    if hasattr(candidates, '__iter__') and len(candidates) > 0:
                        first_item = list(candidates)[0]
                        if isinstance(first_item, (int, np.integer)):
                            # Shapely 2.0+ returns indices
                            final_candidates = [valid_geometries[i] for i in candidates]
                        elif isinstance(first_item, (Polygon, LineString, Point)):
                             # Shapely < 2.0 returns geometries
                             final_candidates = list(candidates)
                        else:
                             # Fallback
                             final_candidates = valid_geometries
                    else:
                        final_candidates = valid_geometries
                    
                    if not final_candidates:
                        final_candidates = valid_geometries
                        
                    for contour in final_candidates:
                        # Check if this contour is a valid boundary for the current point's region
                        if inside_regions:
                            contour_idx = geom_id_to_index.get(id(contour))
                            if contour_idx is not None:
                                # Region 1: Only contour 0 is boundary (holes are interfaces)
                                if region_id == 1 and contour_idx != 0:
                                    continue
                                # Region K (>1): Only contour K-1 is boundary
                                if region_id > 1 and contour_idx != (region_id - 1):
                                    continue
                        
                        # Calculate distance
                        dist = point_geom.distance(contour)
                        # Handle potential array return (unlikely for single point, but safe)
                        if isinstance(dist, (np.ndarray, list)):
                            dist = np.min(dist)
                        min_distance_to_contour = min(min_distance_to_contour, float(dist))
                else:
                    # Fallback to checking all contours
                    for idx, contour in enumerate(shapely_contours):
                        if contour is not None:
                            # Check if this contour is a valid boundary for the current point's region
                            if inside_regions:
                                # Region 1: Only contour 0 is boundary
                                if region_id == 1 and idx != 0:
                                    continue
                                # Region K (>1): Only contour K-1 is boundary
                                if region_id > 1 and idx != (region_id - 1):
                                    continue
                                    
                            min_distance_to_contour = min(min_distance_to_contour, point_geom.distance(contour))
                
                # Classify as boundary if close to any contour
                if min_distance_to_contour <= boundary_tolerance:
                    is_boundary = True

            
            # Fallback method: Check distance to global domain boundary
            if not is_boundary and domain_boundary is not None:
                distance_to_domain = point_geom.distance(domain_boundary)
                if distance_to_domain <= boundary_tolerance:
                    is_boundary = True
            
        except Exception as e:
            # Fallback to simple coordinate-based classification if Shapely fails
            logging.warning(f"Shapely operation failed for point ({x}, {y}): {e}")
            if (abs(x - global_min_x) < boundary_tolerance or 
                abs(x - global_max_x) < boundary_tolerance or
                abs(y - global_min_y) < boundary_tolerance or 
                abs(y - global_max_y) < boundary_tolerance):
                is_boundary = True
        
        classifications.append("boundary" if is_boundary else "interior")
    
    # Log classification statistics
    boundary_count = sum(1 for c in classifications if c == "boundary")
    interior_count = len(classifications) - boundary_count
    logging.info(f"Node classification using Shapely: {boundary_count} boundary, {interior_count} interior")
    logging.info(f"Boundary tolerance used: {boundary_tolerance}")
    
    return classifications
