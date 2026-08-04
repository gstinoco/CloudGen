"""
Point Generation Module - Core Cloud Generation Algorithms

This module implements the core algorithms for generating point clouds using
both Regular (Grid-based) and Natural (Poisson Disk Sampling) distributions.
It handles the complex logic of boundary and interior point generation for
multi-region geometries.

Core Functionality:
1. Boundary Generation: Creates uniformly spaced points along contours.
2. Interior Generation: Fills regions using Grid or Poisson algorithms.
3. Hole Handling: Sophisticated logic for excluding interior holes from main regions.
4. Parallel Processing: Optimized generation of multiple regions using process pools.

Key Features:
- Two distinct generation strategies: Regular (Uniform) and Natural (Poisson)
- Vectorized grid generation using OpenCV for high performance
- Advanced Poisson Disk Sampling for organic point distribution
- Robust handling of complex geometries with nested holes
- Parallel execution for improved performance on multi-core systems

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- NumPy
- Shapely
- OpenCV (cv2)
- Concurrent Futures
"""

import numpy as np
from shapely.geometry import Point, Polygon
import cv2
import logging
import random
from scipy.spatial import Voronoi
import os
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path
from .utils import calculate_cloud_size, create_closed_contour

def create_fast_polygon_checker(polygon: Polygon):
    """Returns a fast point-in-polygon checker function using matplotlib Path."""
    ext_path = Path(np.array(polygon.exterior.coords))
    hole_paths = [Path(np.array(interior.coords)) for interior in polygon.interiors]
    
    def contains(x, y):
        pt = (x, y)
        if not ext_path.contains_point(pt): return False
        for hp in hole_paths:
            if hp.contains_point(pt): return False
        return True
    return contains

def generate_boundary_points(contour: list[tuple[float, float]], cloud_size: float) -> np.ndarray:
    """
    Generate points along the boundary of a contour.
    
    Args:
        contour (list): List of (x, y) coordinate tuples defining the boundary
        cloud_size (float): Desired spacing between points
    
    Returns:
        numpy.ndarray: Array of (x, y) coordinates of boundary points
    """
    if len(contour) < 2:
        return np.empty((0, 2))
    
    boundary_points = []
    
    for i in range(len(contour) - 1):
        x1, y1 = contour[i]
        x2, y2 = contour[i + 1]
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if distance > 0:
            # Use ceil to ensure we don't under-sample the boundary
            # This prevents large gaps when distance is slightly less than a multiple of cloud_size
            num_points = max(1, int(np.ceil(distance / cloud_size)))
            
            for j in range(num_points):
                t = j / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                boundary_points.append([x, y])
    
    if boundary_points:
        boundary_points = np.array(boundary_points)
        rounded_points = np.round(boundary_points / (cloud_size * 0.1)) * (cloud_size * 0.1)
        _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
        boundary_points = boundary_points[unique_indices]
        return boundary_points
    
    return np.empty((0, 2))

def generate_interior_points(polygon: Polygon, cloud_size: float) -> np.ndarray:
    """
    Generate points inside a polygon using a vectorized grid-based approach.
    Optimized using OpenCV mask for fast point-in-polygon checks (replacing Matplotlib Path).
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon to fill with points
        cloud_size (float): Desired spacing between points
    
    Returns:
        numpy.ndarray: Array of (x, y) coordinates of generated points
    """
    try:
        bounds = polygon.bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Create grid of points
        x_range = np.arange(x_min, x_max + cloud_size, cloud_size)
        y_range = np.arange(y_min, y_max + cloud_size, cloud_size)
        
        if len(x_range) == 0 or len(y_range) == 0:
            return np.empty((0, 2))
            
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.vstack((xx.ravel(), yy.ravel())).T
        
        # Create a mask using OpenCV
        width = len(x_range)
        height = len(y_range)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Function to convert world coords to pixel coords
        def to_pixel_coords(coords):
            pixel_coords = []
            for x, y in coords:
                px = int(round((x - x_min) / cloud_size))
                py = int(round((y - y_min) / cloud_size))
                pixel_coords.append([px, py])
            return np.array(pixel_coords, dtype=np.int32)

        # Draw exterior
        ext_coords = list(polygon.exterior.coords)
        ext_pixels = to_pixel_coords(ext_coords)
        cv2.fillPoly(mask, [ext_pixels], 1)
        
        # Exclude exterior boundary (set to 0) to avoid overlapping with explicit boundary nodes
        cv2.polylines(mask, [ext_pixels], isClosed=True, color=0, thickness=1)
        
        # Draw interiors (holes)
        for interior in polygon.interiors:
            int_coords = list(interior.coords)
            int_pixels = to_pixel_coords(int_coords)
            cv2.fillPoly(mask, [int_pixels], 0)
            
        # Select points
        selected_mask = mask.astype(bool).ravel()
        points = grid_points[selected_mask]
        
        return points if len(points) > 0 else np.empty((0, 2))
        
    except Exception as e:
        logging.error(f"Error in vectorized point generation: {e}")
        # Fallback to Path based iterative method
        logging.info("Falling back to Path iterative method")
        points = []
        # Re-generate grid points if needed (though they are local in try block)
        x_range = np.arange(x_min, x_max + cloud_size, cloud_size)
        y_range = np.arange(y_min, y_max + cloud_size, cloud_size)
        fast_contains = create_fast_polygon_checker(polygon)
        
        for x in x_range:
            for y in y_range:
                if fast_contains(x, y):
                    points.append([x, y])
        return np.array(points) if points else np.empty((0, 2))

from scipy.stats.qmc import PoissonDisk
from scipy.spatial import cKDTree

def poisson_disk_sampling(polygon: Polygon, radius: float, k: int = 30, boundary_points: list[tuple[float, float]] = None) -> list[tuple[float, float]]:
    """
    Generate points using Poisson Disk Sampling algorithm via SciPy QMC engine for extreme speed.
    
    Args:
        polygon (shapely.geometry.Polygon): Target polygon to fill with points
        radius (float): Minimum distance between any two generated points
        k (int, optional): Ignored in SciPy version, kept for API compatibility
        boundary_points (list, optional): Pre-existing boundary points to check distances against.
    
    Returns:
        list: List of (x, y) coordinate tuples representing generated points
    """
    minx, miny, maxx, maxy = polygon.bounds
    
    # SciPy PoissonDisk operates in [0, 1] space.
    # We must scale our dimensions and the radius down to [0, 1].
    width = maxx - minx
    height = maxy - miny
    
    # If the polygon is just a point or a line, we can't fill it
    if width <= 0 or height <= 0:
        return []
        
    scale_factor = max(width, height)
    scaled_radius = radius / scale_factor
    
    try:
        # Initialize SciPy engine
        engine = PoissonDisk(d=2, radius=scaled_radius)
        # Generate samples in [0, 1]^2
        samples = engine.fill_space()
    except Exception as e:
        logging.error(f"SciPy PoissonDisk failed: {e}. Falling back to random uniform points.")
        # Very rough fallback if radius is too large for the bounding box
        return []

    # Scale back to world coordinates
    samples[:, 0] = samples[:, 0] * scale_factor + minx
    samples[:, 1] = samples[:, 1] * scale_factor + miny
    
    # Filter points inside polygon using vectorized OpenCV mask for extreme speed
    try:
        # Create a mask with a resolution high enough to capture the polygon accurately
        resolution = max(radius * 0.1, (maxx - minx) / 2000.0) # at most 2000px wide
        if resolution <= 0: resolution = 1.0
        
        width_px = int(np.ceil((maxx - minx) / resolution)) + 1
        height_px = int(np.ceil((maxy - miny) / resolution)) + 1
        
        mask = np.zeros((height_px, width_px), dtype=np.uint8)
        
        def to_pixel_coords(coords):
            pixel_coords = []
            for x, y in coords:
                px = int(round((x - minx) / resolution))
                py = int(round((y - miny) / resolution))
                pixel_coords.append([px, py])
            return np.array(pixel_coords, dtype=np.int32)
            
        # Draw exterior
        ext_pixels = to_pixel_coords(list(polygon.exterior.coords))
        cv2.fillPoly(mask, [ext_pixels], 1)
        # Exclude exterior boundary to avoid overlapping
        cv2.polylines(mask, [ext_pixels], isClosed=True, color=0, thickness=1)
        
        # Draw holes
        for interior in polygon.interiors:
            int_pixels = to_pixel_coords(list(interior.coords))
            cv2.fillPoly(mask, [int_pixels], 0)
            
        # Map samples to pixel coordinates and check mask
        sample_px = np.round((samples[:, 0] - minx) / resolution).astype(int)
        sample_py = np.round((samples[:, 1] - miny) / resolution).astype(int)
        
        # Filter bounds to avoid IndexError
        valid_idx = (sample_px >= 0) & (sample_px < width_px) & (sample_py >= 0) & (sample_py < height_px)
        
        # Of the valid indices, check the mask
        in_polygon = np.zeros(len(samples), dtype=bool)
        in_polygon[valid_idx] = mask[sample_py[valid_idx], sample_px[valid_idx]] > 0
        
        interior_points = samples[in_polygon].tolist()
        interior_points = [(p[0], p[1]) for p in interior_points]
        
    except Exception as e:
        logging.error(f"Vectorized masking failed in Poisson: {e}. Falling back to Path checker.")
        fast_contains = create_fast_polygon_checker(polygon)
        interior_points = [
            (x, y) for x, y in samples 
            if fast_contains(x, y)
        ]
    
    if not interior_points:
        return []
        
    if boundary_points and len(boundary_points) > 0:
        # Filter points that are too close to the boundary points using cKDTree
        boundary_array = np.array(boundary_points)
        interior_array = np.array(interior_points)
        
        # Build KDTree for boundary points
        tree = cKDTree(boundary_array)
        
        # Query distances. k=1 gets the closest boundary point to each interior point
        distances, _ = tree.query(interior_array, k=1)
        
        # Keep only interior points that are at least `radius` away from any boundary
        valid_indices = distances >= (radius * 0.8) # Relaxing the radius slightly to avoid extreme culling near edges
        
        final_points = interior_array[valid_indices].tolist()
        return [(p[0], p[1]) for p in final_points]
        
    return interior_points

def generate_interior_points_poisson(polygon: Polygon, cloud_size: float) -> list[tuple[float, float]]:
    """
    Generate interior points using Poisson Disk Sampling for more natural distribution.
    
    Args:
        polygon (Polygon): Shapely polygon representing the region
        cloud_size (float): Target spacing between points (used as radius)
    
    Returns:
        list: List of (x, y) tuples representing interior points
    """
    # Use cloud_size as the minimum distance between points
    radius = cloud_size * 0.8  # Slightly smaller to get more points
    
    try:
        interior_points = poisson_disk_sampling(polygon, radius)
        logging.info(f"Generated {len(interior_points)} interior points using Poisson Disk Sampling")
        return interior_points
    except Exception as e:
        logging.error(f"Error in Poisson Disk Sampling, falling back to grid method: {e}")
        # Fallback to original method
        return generate_interior_points(polygon, cloud_size)

def generate_region_cloud(region_points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a complete cloud for a single region.
    This function generates the SAME result every time for the same input.
    """
    try:
        # Calculate cloud size for this region
        cloud_size = calculate_cloud_size(region_points)
        
        # Create closed contour
        contour = create_closed_contour(region_points)
        
        # Generate boundary points
        boundary_points = generate_boundary_points(contour, cloud_size)
        
        # Create polygon for interior point generation
        polygon = Polygon(contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        
        # Generate interior points
        interior_points = generate_interior_points(polygon, cloud_size)
        
        logging.info(f"Generated {len(boundary_points)} boundary points and {len(interior_points)} interior points")
        
        return boundary_points, interior_points, cloud_size
        
    except Exception as e:
        logging.error(f"Error generating region cloud: {e}")
        return None, None, None

def generate_region_cloud_with_uniform_density(region_points: list[tuple[float, float]], cloud_size: float) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a complete cloud for a single region using a specified cloud size.
    This ensures uniform density across different regions.
    """
    try:
        # Create closed contour
        contour = create_closed_contour(region_points)
        
        # Generate boundary points using the specified cloud size
        boundary_points = generate_boundary_points(contour, cloud_size)
        
        # Create polygon for interior point generation
        polygon = Polygon(contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        
        # Generate interior points using the specified cloud size
        interior_points = generate_interior_points(polygon, cloud_size)
        
        logging.info(f"Generated {len(boundary_points)} boundary points and {len(interior_points)} interior points with uniform density")
        
        return boundary_points, interior_points, cloud_size
        
    except Exception as e:
        logging.error(f"Error generating region cloud with uniform density: {e}")
        return None, None, None

def lloyd_relaxation(interior_points: np.ndarray, boundary_points: np.ndarray, polygon: Polygon, iterations: int = 5, tolerance: float = 1e-4) -> np.ndarray:
    """
    Applies Lloyd's relaxation using Voronoi diagrams to regularize the point cloud into a honeycomb-like pattern.
    Only interior points are moved; boundary points are fixed.
    
    Args:
        interior_points: Array of interior (x, y) points.
        boundary_points: Array of boundary (x, y) points.
        polygon: Shapely polygon to restrict the points.
        iterations: Number of relaxation steps.
        tolerance: Minimum movement threshold for early stopping.
    
    Returns:
        Relaxed interior points.
    """
    if len(interior_points) == 0:
        return interior_points

    # To avoid boundary issues in Voronoi, add dummy points far away (bounding box corners)
    minx, miny, maxx, maxy = polygon.bounds
    dx, dy = maxx - minx, maxy - miny
    dummy_points = np.array([
        [minx - dx, miny - dy], [maxx + dx, miny - dy],
        [maxx + dx, maxy + dy], [minx - dx, maxy + dy]
    ])

    int_pts = np.array(interior_points)
    bnd_pts = np.array(boundary_points) if len(boundary_points) > 0 else np.empty((0, 2))
    
    fast_contains = create_fast_polygon_checker(polygon)

    for step in range(iterations):
        # Combine all points: [interior, boundary, dummy]
        pts = np.vstack([int_pts, bnd_pts, dummy_points]) if len(bnd_pts) > 0 else np.vstack([int_pts, dummy_points])
        
        try:
            vor = Voronoi(pts)
        except Exception as e:
            logging.warning(f"Voronoi computation failed during Lloyd relaxation: {e}")
            break
            
        new_int_pts = []
        max_movement = 0.0
        
        # Update only interior points (indices 0 to len(int_pts)-1)
        for i in range(len(int_pts)):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if -1 in region or len(region) == 0:
                new_int_pts.append(int_pts[i])
                continue
                
            # Get vertices of this Voronoi region
            cell_vertices = vor.vertices[region]
            
            # Simple centroid of the polygon formed by vertices
            centroid = np.mean(cell_vertices, axis=0)
            
            # Ensure centroid is strictly inside polygon using fast checker
            if fast_contains(centroid[0], centroid[1]):
                new_int_pts.append(centroid)
                dist = np.sqrt((centroid[0] - int_pts[i][0])**2 + (centroid[1] - int_pts[i][1])**2)
                if dist > max_movement:
                    max_movement = dist
            else:
                new_int_pts.append(int_pts[i]) # keep original
                
        int_pts = np.array(new_int_pts)
        
        if max_movement < tolerance:
            logging.info(f"Lloyd relaxation converged early after {step+1} iterations (max movement: {max_movement:.6f})")
            break

    return int_pts

def generate_region_cloud_poisson(region_points: list[tuple[float, float]], cloud_size: float) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a complete cloud for a single region using Poisson Disk Sampling for more natural distribution.
    """
    try:
        # Create closed contour
        contour = create_closed_contour(region_points)
        
        # Generate boundary points using the specified cloud size
        boundary_points = generate_boundary_points(contour, cloud_size)
        
        # Create polygon for interior point generation
        polygon = Polygon(contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        
        # Generate interior points using Poisson Disk Sampling
        interior_points = generate_interior_points_poisson(polygon, cloud_size)
        
        # Apply Lloyd's relaxation to regularize the mesh (Honeycomb effect)
        if len(interior_points) > 0:
            interior_points = lloyd_relaxation(np.array(interior_points), np.array(boundary_points), polygon, iterations=5).tolist()
        
        logging.info(f"Generated {len(boundary_points)} boundary points and {len(interior_points)} interior points using Poisson Disk Sampling")
        
        return boundary_points, interior_points, cloud_size
        
    except Exception as e:
        logging.error(f"Error generating region cloud with Poisson Disk Sampling: {e}")
        # Fallback to uniform density method
        return generate_region_cloud_with_uniform_density(region_points, cloud_size)

def generate_region_cloud_with_holes(main_region_points: list[tuple[float, float]], hole_regions_list: list[list[tuple[float, float]]], cloud_size: float = None, inside_regions: bool = False) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a cloud for the main region (region 1) considering interior regions as holes.
    """
    try:
        # Calculate cloud size for the main region if not provided
        if cloud_size is None:
            cloud_size = calculate_cloud_size(main_region_points)
        
        # Create closed contour for main region
        main_contour = create_closed_contour(main_region_points)
        
        # Generate boundary points for main region
        boundary_points = generate_boundary_points(main_contour, cloud_size)
        
        # Create main polygon
        main_polygon = Polygon(main_contour)
        if not main_polygon.is_valid:
            main_polygon = main_polygon.buffer(0)
        
        # Create hole polygons and generate boundary points for holes
        hole_polygons = []
        hole_boundary_count = 0
        
        for hole_region in hole_regions_list:
            hole_contour = create_closed_contour(hole_region)
            
            # Generate boundary points for this hole (it is an interior boundary)
            # Only if inside_regions is False (meaning we are NOT filling the holes with other regions)
            # If inside_regions is True, these boundaries belong to the interior regions (regions 2, 3, etc.)
            if not inside_regions:
                hole_points = generate_boundary_points(hole_contour, cloud_size)
                if len(hole_points) > 0:
                    boundary_points = np.vstack((boundary_points, hole_points))
                    hole_boundary_count += len(hole_points)
                
            hole_polygon = Polygon(hole_contour)
            if not hole_polygon.is_valid:
                hole_polygon = hole_polygon.buffer(0)
            hole_polygons.append(hole_polygon)
        
        if hole_boundary_count > 0:
            logging.info(f"Generated {hole_boundary_count} boundary points from {len(hole_regions_list)} hole regions (inside_regions={inside_regions})")
        
        # Create polygon with holes
        if hole_polygons:
            # Create a polygon with holes
            polygon_with_holes = main_polygon
            for hole in hole_polygons:
                # Subtract each hole from the main polygon
                polygon_with_holes = polygon_with_holes.difference(hole)
        else:
            polygon_with_holes = main_polygon
        
        # Generate interior points avoiding holes
        interior_points = generate_interior_points(polygon_with_holes, cloud_size)
        
        logging.info(f"Generated main region with holes: {len(boundary_points)} boundary points and {len(interior_points)} interior points")
        logging.info(f"Excluded {len(hole_polygons)} hole regions from main region")
        
        return boundary_points, interior_points, cloud_size
        
    except Exception as e:
        logging.error(f"Error generating region cloud with holes: {e}")
        return None, None, None

def generate_region_cloud_with_holes_poisson(main_region_points: list[tuple[float, float]], hole_regions_list: list[list[tuple[float, float]]], cloud_size: float = None, inside_regions: bool = False) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a cloud for the main region (region 1) considering interior regions as holes,
    using Poisson Disk Sampling for more natural distribution.
    """
    try:
        # Calculate cloud size for the main region if not provided
        if cloud_size is None:
            cloud_size = calculate_cloud_size(main_region_points)
        
        # Create closed contour for main region
        main_contour = create_closed_contour(main_region_points)
        
        # Generate boundary points for main region
        boundary_points = generate_boundary_points(main_contour, cloud_size)
        
        # Create main polygon
        main_polygon = Polygon(main_contour)
        if not main_polygon.is_valid:
            main_polygon = main_polygon.buffer(0)
        
        # Create hole polygons and generate boundary points for holes
        hole_polygons = []
        hole_boundary_count = 0
        
        for hole_region in hole_regions_list:
            hole_contour = create_closed_contour(hole_region)
            
            # Generate boundary points for this hole (it is an interior boundary)
            # Only if inside_regions is False (meaning we are NOT filling the holes with other regions)
            # If inside_regions is True, these boundaries belong to the interior regions (regions 2, 3, etc.)
            if not inside_regions:
                hole_points = generate_boundary_points(hole_contour, cloud_size)
                if len(hole_points) > 0:
                    boundary_points = np.vstack((boundary_points, hole_points))
                    hole_boundary_count += len(hole_points)
                
            hole_polygon = Polygon(hole_contour)
            if not hole_polygon.is_valid:
                hole_polygon = hole_polygon.buffer(0)
            hole_polygons.append(hole_polygon)
        
        if hole_boundary_count > 0:
            logging.info(f"Generated {hole_boundary_count} boundary points from {len(hole_regions_list)} hole regions (inside_regions={inside_regions})")
        
        # Create polygon with holes
        if hole_polygons:
            # Create a polygon with holes
            polygon_with_holes = main_polygon
            for hole in hole_polygons:
                # Subtract each hole from the main polygon
                polygon_with_holes = polygon_with_holes.difference(hole)
        else:
            polygon_with_holes = main_polygon
        
        # Generate interior points avoiding holes using Poisson Disk Sampling
        interior_points = generate_interior_points_poisson(polygon_with_holes, cloud_size)
        
        # Apply Lloyd's relaxation
        if len(interior_points) > 0:
            interior_points = lloyd_relaxation(np.array(interior_points), np.array(boundary_points), polygon_with_holes, iterations=5).tolist()
        
        logging.info(f"Generated main region with holes using Poisson: {len(boundary_points)} boundary points and {len(interior_points)} interior points")
        logging.info(f"Excluded {len(hole_polygons)} hole regions from main region")
        
        return boundary_points, interior_points, cloud_size
        
    except Exception as e:
        logging.error(f"Error generating region cloud with holes using Poisson: {e}")
        # Fallback to grid-based method with holes
        return generate_region_cloud_with_holes(main_region_points, hole_regions_list)

def generate_region_task(region_points: list[tuple[float, float]], region_id: int, main_cloud_size: float) -> tuple[np.ndarray, np.ndarray, int] | None:
    """
    Helper function for parallel region generation.
    Must be at top level for ProcessPoolExecutor (pickling).
    """
    try:
        poly = Polygon(region_points)
        minx, miny, maxx, maxy = poly.bounds
        min_dim = min(maxx - minx, maxy - miny)
        
        # Shrink cloud size for very small regions
        actual_cloud_size = main_cloud_size
        if min_dim < main_cloud_size * 2:
            actual_cloud_size = min_dim / 3.0
            logging.info(f"Region {region_id} is small (min_dim={min_dim:.6f}). Adapted cloud size to {actual_cloud_size:.6f}")

        boundary_points, interior_points, _ = generate_region_cloud_with_uniform_density(region_points, actual_cloud_size)
        
        # Fallback to representative point if interior is empty
        if interior_points is not None and len(interior_points) == 0:
            logging.warning(f"Region {region_id} has 0 interior points. Using representative point fallback.")
            rep = poly.representative_point()
            interior_points = np.array([[rep.x, rep.y]])

        if boundary_points is not None and interior_points is not None:
            logging.info(f"Generated cloud for interior region {region_id} with uniform density (cloud_size: {actual_cloud_size:.6f})")
            return (boundary_points, interior_points, region_id)
        else:
            logging.warning(f"Failed to generate cloud for interior region {region_id}")
            return None
    except Exception as e:
        logging.error(f"Error generating cloud for interior region {region_id}: {e}")
        return None

def generate_interior_regions_clouds(regions: list[list[tuple[float, float]]], main_cloud_size: float) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate clouds for interior regions (regions 2 onwards) using the same cloud size as the main region.
    This ensures uniform density across all regions.
    """
    interior_clouds = []
    
    # Process regions 2 onwards (skip region 1 which is index 0)
    # Using ProcessPoolExecutor for parallelization
    
    # Determine max workers (leave one core free or use all if few)
    max_workers = max(1, os.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(1, len(regions)):
            region_points = regions[i]
            region_id = i + 1
            futures.append(executor.submit(generate_region_task, region_points, region_id, main_cloud_size))
        
        for future in futures:
            try:
                result = future.result()
                if result:
                    interior_clouds.append(result)
            except Exception as e:
                logging.error(f"Error retrieving parallel task result: {e}")
    
    return interior_clouds

def generate_region_task_poisson(region_points: list[tuple[float, float]], region_id: int, main_cloud_size: float) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Helper for parallel Poisson interior regions."""
    try:
        poly = Polygon(region_points)
        minx, miny, maxx, maxy = poly.bounds
        min_dim = min(maxx - minx, maxy - miny)
        
        actual_cloud_size = main_cloud_size
        if min_dim < main_cloud_size * 2:
            actual_cloud_size = min_dim / 3.0
            logging.info(f"Region {region_id} is small (min_dim={min_dim:.6f}). Adapted Poisson cloud size to {actual_cloud_size:.6f}")
            
        boundary_points, interior_points, _ = generate_region_cloud_poisson(region_points, actual_cloud_size)
        
        if interior_points is not None and len(interior_points) == 0:
            logging.warning(f"Region {region_id} has 0 Poisson interior points. Using representative point fallback.")
            rep = poly.representative_point()
            interior_points = np.array([[rep.x, rep.y]])
            
        if boundary_points is not None and interior_points is not None:
            return (boundary_points, interior_points, region_id)
        return None
    except Exception as e:
        logging.error(f"Error in poisson interior task {region_id}: {e}")
        return None

def generate_interior_regions_clouds_poisson(regions: list[list[tuple[float, float]]], main_cloud_size: float) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Parallel generation for interior regions using Poisson."""
    interior_clouds = []
    max_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(1, len(regions)):
            futures.append(executor.submit(generate_region_task_poisson, regions[i], i + 1, main_cloud_size))
        
        for future in futures:
            try:
                result = future.result()
                if result:
                    interior_clouds.append(result)
            except Exception as e:
                logging.error(f"Error in parallel poisson task: {e}")
    return interior_clouds
