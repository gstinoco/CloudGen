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
import os
from concurrent.futures import ProcessPoolExecutor
from .utils import calculate_cloud_size, create_closed_contour

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
        # Fallback to Shapely iterative method
        logging.info("Falling back to Shapely iterative method")
        points = []
        # Re-generate grid points if needed (though they are local in try block)
        x_range = np.arange(x_min, x_max + cloud_size, cloud_size)
        y_range = np.arange(y_min, y_max + cloud_size, cloud_size)
        for x in x_range:
            for y in y_range:
                if polygon.contains(Point(x, y)):
                    points.append([x, y])
        return np.array(points) if points else np.empty((0, 2))

def is_valid_poisson_point(x: float, y: float, minx: float, miny: float, cell_size: float, grid: np.ndarray, points: list[tuple[float, float]], radius: float) -> bool:
    """
    Check if a point is valid for Poisson disk sampling.
    
    Args:
        x, y (float): Coordinates of the point to check
        minx, miny (float): Minimum bounds of the region
        cell_size (float): Size of grid cells
        grid (np.array): Spatial grid for fast lookup
        points (list): List of existing points
        radius (float): Minimum distance between points
    
    Returns:
        bool: True if point is valid, False otherwise
    """
    # Convert to grid coordinates
    grid_x = int((x - minx) / cell_size)
    grid_y = int((y - miny) / cell_size)
    
    # Check neighboring cells
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            neighbor_x = grid_x + dx
            neighbor_y = grid_y + dy
            
            if (neighbor_x < 0 or neighbor_x >= grid.shape[1] or 
                neighbor_y < 0 or neighbor_y >= grid.shape[0]):
                continue
            
            point_idx = grid[neighbor_y, neighbor_x]
            if point_idx != -1:
                px, py = points[point_idx]
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                if distance < radius:
                    return False
    
    return True

def poisson_disk_sampling(polygon: Polygon, radius: float, k: int = 30, boundary_points: list[tuple[float, float]] = None) -> list[tuple[float, float]]:
    """
    Generate points using Poisson Disk Sampling algorithm for natural point distribution.
    
    Args:
        polygon (shapely.geometry.Polygon): Target polygon to fill with points
        radius (float): Minimum distance between any two generated points
        k (int, optional): Number of attempts to generate points around each active point.
                          Higher values increase density but also computation time. Default: 30
        boundary_points (list, optional): Pre-existing boundary points to incorporate.
                                        If provided, these points are added to the grid first.
    
    Returns:
        list: List of (x, y) coordinate tuples representing generated points
    """
    minx, miny, maxx, maxy = polygon.bounds
    
    cell_size = radius / np.sqrt(2)
    
    grid_width = int(np.ceil((maxx - minx) / cell_size))
    grid_height = int(np.ceil((maxy - miny) / cell_size))
    
    grid = np.full((grid_height, grid_width), -1, dtype=int)
    
    points = []
    active_list = []
    
    if boundary_points:
        for i, (x, y) in enumerate(boundary_points):
            if polygon.contains(Point(x, y)) or polygon.boundary.contains(Point(x, y)):
                points.append((x, y))
                grid_x = int((x - minx) / cell_size)
                grid_y = int((y - miny) / cell_size)
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    grid[grid_y, grid_x] = len(points) - 1
    
    if not points:
        for _ in range(100):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if polygon.contains(Point(x, y)):
                points.append((x, y))
                active_list.append(len(points) - 1)
                grid_x = int((x - minx) / cell_size)
                grid_y = int((y - miny) / cell_size)
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    grid[grid_y, grid_x] = len(points) - 1
                break
    else:
        for i in range(min(5, len(points))):
            active_list.append(i)

    while active_list:
        active_idx = random.randint(0, len(active_list) - 1)
        point_idx = active_list[active_idx]
        center_x, center_y = points[point_idx]
        
        found_valid = False
        for _ in range(k):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(radius, 2 * radius)
            
            new_x = center_x + distance * np.cos(angle)
            new_y = center_y + distance * np.sin(angle)
            
            if not polygon.contains(Point(new_x, new_y)):
                continue
            
            if is_valid_poisson_point(new_x, new_y, minx, miny, cell_size, grid, points, radius):
                points.append((new_x, new_y))
                active_list.append(len(points) - 1)
                
                grid_x = int((new_x - minx) / cell_size)
                grid_y = int((new_y - miny) / cell_size)
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    grid[grid_y, grid_x] = len(points) - 1
                
                found_valid = True
                break
        
        if not found_valid:
            # Optimized removal: Swap with last element and pop (O(1)) instead of pop(idx) (O(N))
            active_list[active_idx] = active_list[-1]
            active_list.pop()
    
    if boundary_points:
        boundary_set = set(boundary_points)
        interior_points = [p for p in points if p not in boundary_set]
        return interior_points
    
    return points

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
        boundary_points, interior_points, _ = generate_region_cloud_with_uniform_density(region_points, main_cloud_size)
        if boundary_points is not None and interior_points is not None:
            logging.info(f"Generated cloud for interior region {region_id} with uniform density (cloud_size: {main_cloud_size:.6f})")
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
        boundary_points, interior_points, _ = generate_region_cloud_poisson(region_points, main_cloud_size)
        if boundary_points is not None and interior_points is not None:
            return (boundary_points, interior_points, region_id)
        return None
    except Exception:
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
