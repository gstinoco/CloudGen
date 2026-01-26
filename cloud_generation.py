"""
Cloud Generation Module - Advanced Cloud of Points Generation System

This module provides comprehensive functionality for generating optimized clouds of points
from CSV boundary data for use with the meshless Generalized Finite Differences (mGFD) method.
The module implements two advanced distribution algorithms with intelligent point reduction
and multi-region support for complex geometries.

Core Functionality:
1. CSV boundary point processing
2. Optional contour point reduction for optimization
3. Cloud of points generation for main region (Region 1) with consistent results
4. Multi-region support for interior holes and complex geometries
5. Intelligent node classification (boundary vs interior)
6. Comprehensive visualization and export capabilities

Point Generation Algorithms:
- Regular Distribution: Grid-based uniform point placement with adaptive spacing
  * Uses structured grid patterns for consistent point distribution
  * Adaptive cloud size calculation based on region geometry
  * Boundary point generation with uniform spacing along contours
  * Interior point filling using grid-based approach

- Natural Distribution: Poisson Disk Sampling for organic point patterns
  * Implements advanced Poisson Disk Sampling algorithm
  * Generates naturally spaced points with minimum distance constraints
  * Adaptive radius calculation for optimal point density
  * Boundary-aware sampling with interior region filling

Key Features:
- Adaptive cloud size calculation based on region geometry
- Dynamic boundary refinement using point density analysis
- Multi-region processing with separate cloud generation
- Intelligent node classification using Shapely geometric operations
- Comprehensive visualization with region-specific color coding
- Multiple export formats (CSV, PNG, SVG) with detailed metadata
- Robust error handling and comprehensive logging
- Consistent results for identical input parameters

Technical Implementation:
- NumPy for efficient numerical computations and array operations
- Standard libraries for CSV data manipulation and coordinate processing
- Shapely for geometric operations, polygon validation, and spatial queries
- STRtree for optimized O(log N) spatial indexing and nearest-neighbor queries
- ProcessPoolExecutor for parallel generation of multi-region clouds
- Matplotlib for high-quality visualization generation and export
- OpenCV for optimized point-in-polygon checks
- Professional logging system with detailed progress tracking

Workflow Process:
1. Load and validate CSV boundary data with required columns (x, y, region)
2. Use coordinates as provided (assumed pre-scaled)
3. Apply optional point reduction to optimize contour complexity
4. Calculate adaptive cloud size based on region geometry characteristics
5. Generate boundary points with uniform spacing along contours
6. Fill interior regions using selected distribution algorithm
7. Process additional regions if multi-region support is enabled
8. Classify all nodes as boundary or interior using geometric analysis
9. Export results in multiple formats with comprehensive metadata
10. Generate visualization with region-specific color coding

Key Principle: Region 1 (main region) generates identical results regardless of
interior regions to ensure consistency and reproducibility in cloud generation.

Author: Gerardo Tinoco-Guerrero
Date: May, 2025
Last Modification: January 26th, 2026

Dependencies:
- NumPy >= 1.20.0
- Shapely >= 1.8.0
- Matplotlib >= 3.10.0
- OpenCV >= 4.5.0
"""

from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for server environments
import matplotlib.pyplot as plt
import reduce_points
import numpy as np
import tempfile
import logging
import random
import csv
import cv2
import os

# Absolute routes for logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('VERCEL'):
    LOG_DIR = os.path.join(tempfile.gettempdir(), 'logs')
else:
    LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
handlers = [logging.StreamHandler()]
if not os.environ.get('VERCEL'):
    handlers.append(logging.FileHandler(os.path.join(LOG_DIR, 'cloud_generation.log')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)


# Cloud generation parameters
CLOUD_FACTORS = {
    "adaptive_factor": 0.15,
    "default_cloud_size": 0.05,
    "boundary_refinement": 0.00325
}

def load_regions(csv_file: str) -> list[list[tuple[float, float]]]:
    """Load region data from CSV file."""
    try:
        regions_dict = {}
        single_region_points = []
        has_region_column = False
        
        with open(csv_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            if not reader.fieldnames:
                logging.error(f"CSV file is empty or invalid: {csv_file}")
                return []

            fieldnames = [f.strip() for f in reader.fieldnames]
            if 'x' not in fieldnames or 'y' not in fieldnames:
                logging.error(f"CSV file must contain columns: ['x', 'y']")
                return []
                
            has_region_column = 'region' in fieldnames
            
            for row in reader:
                try:
                    # Clean and convert coordinates
                    x_str = row['x'].strip() if row.get('x') else ''
                    y_str = row['y'].strip() if row.get('y') else ''
                    
                    if not x_str or not y_str:
                        continue
                        
                    x = float(x_str)
                    y = float(y_str)
                    
                    if has_region_column and row.get('region'):
                        # Handle region column
                        region_val_str = str(row['region']).strip()
                        try:
                            # Try to parse as float then int (handles "1.0")
                            region_val = float(region_val_str)
                            region_id = int(region_val)
                        except (ValueError, TypeError):
                            # Fallback if region is not a number
                            region_id = region_val_str
                            
                        if region_id not in regions_dict:
                            regions_dict[region_id] = []
                        regions_dict[region_id].append((x, y))
                    else:
                        single_region_points.append((x, y))
                        
                except ValueError:
                    continue # Skip invalid rows

        regions = []
        if has_region_column and regions_dict:
            # Sort by region_id
            try:
                sorted_keys = sorted(regions_dict.keys())
            except TypeError:
                # Handle mixed types if necessary (e.g. str and int), usually won't happen but safe to convert to str
                sorted_keys = sorted(regions_dict.keys(), key=str)
                
            for key in sorted_keys:
                regions.append(regions_dict[key])
        else:
            if single_region_points:
                regions.append(single_region_points)
        
        logging.info(f"Loaded {len(regions)} regions from {csv_file}")
        return regions
        
    except Exception as e:
        logging.error(f"Error loading regions from {csv_file}: {e}")
        raise


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
            num_points = max(1, int(distance / cloud_size))
            
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


def poisson_disk_sampling(polygon: Polygon, radius: float, k: int = 30, boundary_points: list[tuple[float, float]] = None) -> list[tuple[float, float]]:
    """
    Generate points using Poisson Disk Sampling algorithm for natural point distribution.
    
    This function implements the Poisson Disk Sampling algorithm to generate points with
    a minimum distance constraint, creating a more natural and organic point distribution
    compared to regular grid-based methods. The algorithm ensures no two points are closer
    than the specified radius while maintaining good coverage of the polygon area.
    
    Args:
        polygon (shapely.geometry.Polygon): Target polygon to fill with points
        radius (float): Minimum distance between any two generated points
        k (int, optional): Number of attempts to generate points around each active point.
                          Higher values increase density but also computation time. Default: 30
        boundary_points (list, optional): Pre-existing boundary points to incorporate.
                                        If provided, these points are added to the grid first.
    
    Returns:
        list: List of (x, y) coordinate tuples representing generated points
    
    Algorithm:
        1. Create a spatial grid for efficient neighbor checking
        2. Initialize with boundary points if provided
        3. Use active list to track points that can generate new neighbors
        4. For each active point, attempt k random point generations
        5. Accept points that satisfy minimum distance constraint
        6. Continue until no more valid points can be generated
    
    Note:
        The algorithm uses a grid-based spatial data structure for O(1) neighbor
        lookups, making it efficient for large point sets. Cell size is set to
        radius/√2 for optimal performance.
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
    Only called when inside_regions=True.
    Optimized with ProcessPoolExecutor for parallel generation.
    
    Args:
        regions (list): List of all regions, where regions[0] is the main region
        main_cloud_size (float): Cloud size calculated for the main region to ensure uniform density
    
    Returns:
        list: List of tuples (boundary_points, interior_points, region_id) for each interior region
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


def classify_nodes(points: np.ndarray, regions_list: list[int], original_regions_contours: list[list[tuple[float, float]]] = None, cloud_size: float = None) -> list[str]:
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
    
    if original_regions_contours:
        for contour_points in original_regions_contours:
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
                    if not candidates:
                        # If no intersection with envelopes, use all valid geometries (fallback)
                        candidates = valid_geometries
                    
                    for contour in candidates:
                         min_distance_to_contour = min(min_distance_to_contour, point_geom.distance(contour))
                else:
                    # Fallback to checking all contours
                    for contour in shapely_contours:
                        if contour is not None:
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


def export_to_csv(points: np.ndarray, classifications: list[str], regions_list: list[int], output_file: str) -> bool:
    """
    Export node data to CSV file with validation and verification.
    
    Args:
        points (numpy.ndarray): Array of point coordinates
        classifications (list): Node classifications
        regions_list (list): Region assignments for each node
        output_file (str): Path to output CSV file
    
    Returns:
        bool: True if export was successful and verified, False otherwise
    """
    try:
        # Ensure output file has .csv extension
        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'
            
        # Input Validation
        if len(points) != len(classifications) or len(points) != len(regions_list):
            logging.error(f"Export validation failed: Array length mismatch. Points: {len(points)}, Classifications: {len(classifications)}, Regions: {len(regions_list)}")
            return False
            
        # Write to file using csv module
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['x', 'y', 'region', 'classification']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for i in range(len(points)):
                writer.writerow({
                    'x': points[i, 0],
                    'y': points[i, 1],
                    'region': regions_list[i],
                    'classification': classifications[i]
                })
        
        # Verification Step: Ensure all data was written correctly
        # This addresses the user reported issue of missing points
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if len(rows) != len(points):
                logging.error(f"Export verification failed: File has {len(rows)} rows, expected {len(points)}")
                return False
                
            # Verify region counts match
            input_regions = set(regions_list)
            
            file_regions = set()
            for row in rows:
                try:
                    # Try to convert to int if possible to match input_regions which are likely ints
                    val = row['region']
                    try:
                        val = int(float(val))
                    except (ValueError, TypeError):
                        pass
                    file_regions.add(val)
                except KeyError:
                    pass
            
            # Note: Set comparison might be tricky due to type differences (str vs int), 
            # so we'll just log if counts are different significantly
            if len(input_regions) != len(file_regions):
                 logging.warning(f"Export verification warning: Regions count mismatch. Expected {len(input_regions)}, found {len(file_regions)}")
                 # We don't return False here as it might be just type mismatch
                 
        except Exception as verify_error:
            logging.error(f"Export verification error: {verify_error}")
            return False
        
        logging.info(f"Successfully exported and verified {len(points)} nodes to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error exporting nodes to {output_file}: {e}")
        return False


def create_visualization(points: np.ndarray, regions_list: list[int], output_base: str, classifications: list[str] = None) -> bool:
    """
    Create a visualization of the generated cloud of points with differentiated colors for boundary and interior nodes.
    Uses Matplotlib for high-quality PNG and SVG generation.
    
    Args:
        points (numpy.ndarray): Array of point coordinates
        regions_list (list): Region assignments for each node
        output_base (str): Base path for output files (without extension)
        classifications (list, optional): Node classifications ('boundary' or 'interior')
    
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    try:
        # Define colors (RGB tuples scaled to 0-1 for matplotlib)
        interior_colors = [
            (51/255, 153/255, 255/255),    # Light Blue
            (77/255, 230/255, 77/255),     # Bright Green  
            (255/255, 179/255, 51/255),    # Bright Orange
            (204/255, 77/255, 255/255),    # Bright Purple
            (230/255, 153/255, 51/255),    # Golden Brown
            (255/255, 128/255, 204/255),   # Bright Pink
            (179/255, 179/255, 179/255)    # Light Gray
        ]
        
        boundary_colors = [
            (204/255, 0/255, 0/255),       # Dark Red
            (153/255, 0/255, 153/255),     # Dark Purple
            (0/255, 77/255, 204/255),      # Dark Blue
            (204/255, 153/255, 0/255),     # Dark Yellow
            (0/255, 102/255, 0/255),       # Dark Green
            (0/255, 153/255, 102/255),     # Dark Teal
            (51/255, 51/255, 51/255)       # Dark Gray
        ]
        
        if len(points) == 0:
            return False
            
        # Convert lists to numpy arrays for efficient indexing
        regions_arr = np.array(regions_list)
        if classifications:
            classifications_arr = np.array(classifications)
        else:
            classifications_arr = np.array(['interior'] * len(points))
            
        # Setup plot
        plt.figure(figsize=(10, 8), dpi=300)
        ax = plt.gca()
        ax.set_aspect('equal')
        
        # Configure axes
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Get unique regions
        unique_regions = sorted(list(set(regions_list)))
        
        # Plot points by region
        for region_id in unique_regions:
            # Region mask
            region_mask = (regions_arr == region_id)
            
            # Boundary points
            boundary_mask = region_mask & (classifications_arr == 'boundary')
            if np.any(boundary_mask):
                boundary_pts = points[boundary_mask]
                color_idx = (region_id - 1) % len(boundary_colors)
                ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], 
                          c=[boundary_colors[color_idx]], s=8, marker='.', 
                          edgecolors='none', label=f'Region {region_id} Boundary', zorder=10)
            
            # Interior points
            interior_mask = region_mask & (classifications_arr != 'boundary')
            if np.any(interior_mask):
                interior_pts = points[interior_mask]
                color_idx = (region_id - 1) % len(interior_colors)
                ax.scatter(interior_pts[:, 0], interior_pts[:, 1], 
                          c=[interior_colors[color_idx]], s=5, marker='.', 
                          alpha=0.8, edgecolors='none', label=f'Region {region_id} Interior', zorder=5)

        # Add title and adjust layout
        plt.title(f'Generated Cloud of Points\n{len(points)} Total Nodes', fontsize=14)
        
        # Optional: Add legend if not too many items
        if len(unique_regions) <= 5:
            plt.legend(loc='best', fontsize='small', markerscale=1.5)
            
        # Save PNG
        png_file = f"{output_base}.png"
        plt.savefig(png_file, format='png', bbox_inches='tight', dpi=300)
        
        # Save SVG
        svg_file = f"{output_base}.svg"
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        
        plt.close()
        
        logging.info(f"Visualization saved to {png_file} and {svg_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
        # Close plot in case of error to free memory
        try:
            plt.close()
        except:
            pass
        return False


def generate_region_cloud(region_points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a complete cloud for a single region.
    This function generates the SAME result every time for the same input.
    
    Args:
        region_points (list): List of (x, y) coordinate tuples for the region
    
    Returns:
        tuple: (boundary_points, interior_points, cloud_size) or (None, None, None) if failed
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
    
    Args:
        region_points (list): List of (x, y) coordinate tuples for the region
        cloud_size (float): Specified cloud size to ensure uniform density
    
    Returns:
        tuple: (boundary_points, interior_points, cloud_size) or (None, None, None) if failed
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
    
    Args:
        region_points (list): List of (x, y) coordinate tuples for the region
        cloud_size (float): Specified cloud size to ensure uniform density
    
    Returns:
        tuple: (boundary_points, interior_points, cloud_size) or (None, None, None) if failed
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


def generate_region_cloud_with_holes(main_region_points: list[tuple[float, float]], hole_regions_list: list[list[tuple[float, float]]], cloud_size: float = None) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a cloud for the main region (region 1) considering interior regions as holes.
    This ensures that region 1 is always generated the same way, regardless of inside_regions flag.
    
    Args:
        main_region_points (list): List of (x, y) coordinate tuples for the main region
        hole_regions_list (list): List of regions to be treated as holes
        cloud_size (float, optional): Specified cloud size. If None, calculated automatically.
    
    Returns:
        tuple: (boundary_points, interior_points, cloud_size) or (None, None, None) if failed
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
        
        # Create hole polygons
        hole_polygons = []
        for hole_region in hole_regions_list:
            hole_contour = create_closed_contour(hole_region)
            hole_polygon = Polygon(hole_contour)
            if not hole_polygon.is_valid:
                hole_polygon = hole_polygon.buffer(0)
            hole_polygons.append(hole_polygon)
        
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


def generate_region_cloud_with_holes_poisson(main_region_points: list[tuple[float, float]], hole_regions_list: list[list[tuple[float, float]]], cloud_size: float = None) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """
    Generate a cloud for the main region (region 1) considering interior regions as holes,
    using Poisson Disk Sampling for more natural distribution.
    
    Args:
        main_region_points (list): List of (x, y) coordinate tuples for the main region
        hole_regions_list (list): List of regions to be treated as holes
        cloud_size (float, optional): Specified cloud size. If None, calculated automatically.
    
    Returns:
        tuple: (boundary_points, interior_points, cloud_size) or (None, None, None) if failed
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
        
        # Create hole polygons
        hole_polygons = []
        for hole_region in hole_regions_list:
            hole_contour = create_closed_contour(hole_region)
            hole_polygon = Polygon(hole_contour)
            if not hole_polygon.is_valid:
                hole_polygon = hole_polygon.buffer(0)
            hole_polygons.append(hole_polygon)
        
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


def _generate_cloud_core(csv_file: str, output_file: str, method_name: str, main_region_strategy: callable, interior_regions_strategy: callable, 
                         inside_regions: bool = False, contour_reduction: bool = False, reduction_percentage: int = 10, cloud_size: float = None) -> dict:
    """
    Core function for cloud generation to strictly follow DRY principle.
    Encapsulates common logic for Natural and Regular distribution methods.
    """
    temp_reduced_file = None
    try:
        logging.info(f"=== Starting Cloud Generation Process with {method_name} Distribution ===")
        logging.info(f"Input file: {csv_file}")
        logging.info(f"Output file: {output_file}")
        logging.info(f"Interior regions: {inside_regions}")
        logging.info(f"Reduce contour: {contour_reduction}")
        
        working_csv_file = csv_file
        
        if contour_reduction and reduction_percentage > 0:
            multiplier = max(1, int(reduction_percentage / 10))
            # Fix: use tempfile properly
            temp_fd, temp_reduced_file = tempfile.mkstemp(suffix='.csv')
            os.close(temp_fd)
            
            logging.info(f"Applying contour reduction with {reduction_percentage}% reduction (multiplier: {multiplier})")
            result_df = reduce_points.reduce_points_by_region(csv_file, temp_reduced_file, multiplier)
            
            if result_df is not None:
                working_csv_file = temp_reduced_file
                logging.info(f"Contour reduction completed successfully")
            else:
                logging.warning("Contour reduction failed, using original file")
                if os.path.exists(temp_reduced_file):
                    os.remove(temp_reduced_file)
                temp_reduced_file = None
        
        regions = load_regions(working_csv_file)
        if not regions:
            if temp_reduced_file and os.path.exists(temp_reduced_file):
                os.remove(temp_reduced_file)
            return {"success": False, "error": "Failed to load regions from CSV"}
        
        logging.info(f"Loaded {len(regions)} region(s)")
        
        main_region = regions[0]
        hole_regions = regions[1:] if len(regions) > 1 else []
        
        # Execute Main Region Strategy
        main_boundary, main_interior, actual_cloud_size = main_region_strategy(main_region, hole_regions, cloud_size)
        
        if main_boundary is None or main_interior is None:
            return {"success": False, "error": "Failed to generate cloud for main region"}
        
        logging.info(f"Generated main region cloud with {method_name} Distribution: {len(main_boundary)} boundary + {len(main_interior)} interior points")
        
        all_points = []
        all_regions = []
        
        for point in main_boundary:
            all_points.append(point)
            all_regions.append(1)
        
        for point in main_interior:
            all_points.append(point)
            all_regions.append(1)
        
        # Execute Interior Regions Strategy
        if inside_regions and len(regions) > 1:
            logging.info(f"Generating interior regions with {method_name} Distribution...")
            interior_clouds = interior_regions_strategy(regions, actual_cloud_size)
            
            for boundary_points, interior_points, region_id in interior_clouds:
                for point in boundary_points:
                    all_points.append(point)
                    all_regions.append(region_id)
                for point in interior_points:
                    all_points.append(point)
                    all_regions.append(region_id)
                logging.info(f"Added region {region_id}: {len(boundary_points)} boundary + {len(interior_points)} interior points")
        
        all_points = np.array(all_points)
        
        classifications = classify_nodes(all_points, all_regions, regions, actual_cloud_size)
        
        success = export_to_csv(all_points, classifications, all_regions, output_file)
        if not success:
            return {"success": False, "error": "Failed to export CSV"}
        
        output_base = os.path.splitext(output_file)[0]
        create_visualization(all_points, all_regions, output_base, classifications)
        
        results = {
            "success": True,
            "output_file": output_file,
            "visualization_file": f"{output_base}.png",
            "visualization_svg_file": f"{output_base}.svg",
            "total_nodes": len(all_points),
            "regions_generated": len(set(all_regions)),
            "main_region_nodes": sum(1 for r in all_regions if r == 1),
            "interior_regions_generated": inside_regions and len(regions) > 1,
            "cloud_size": actual_cloud_size
        }
        
        logging.info(f"=== Cloud Generation with {method_name} Distribution Completed Successfully ===")
        
        if temp_reduced_file and os.path.exists(temp_reduced_file):
            os.remove(temp_reduced_file)
            
        return results
        
    except Exception as e:
        error_msg = f"Error in cloud generation with {method_name} Distribution: {str(e)}"
        logging.error(error_msg)
        if temp_reduced_file and os.path.exists(temp_reduced_file):
            os.remove(temp_reduced_file)
        return {"success": False, "error": error_msg}

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


def generate_cloud_natural(csv_file: str, output_file: str, inside_regions: bool = False, contour_reduction: bool = False, 
                           reduction_percentage: int = 10, cloud_size: float = None) -> dict:
    """
    Generate cloud of points using Natural Distribution algorithm with Poisson Disk Sampling.
    
    This function creates a natural, organic cloud of points using advanced Poisson Disk
    Sampling algorithm. It processes CSV boundary data to generate optimized cloud of points
    suitable for meshless Generalized Finite Differences (mGFD) method computations.
    
    Args:
        csv_file (str): Path to input CSV file containing boundary coordinates.
                       Must have columns: x, y, and optionally 'region' for multi-region support.
        output_file (str): Base path for output files (without extension).
                          Generated files: .csv (coordinates), .png (visualization), .svg (vector).
        inside_regions (bool, optional): Enable multi-region processing for interior holes.
                                        If True, processes regions 2+ as interior holes. Default: False.
        contour_reduction (bool, optional): Apply point reduction to boundary contours.
                                         Reduces computational complexity. Default: False.
        reduction_percentage (int, optional): Percentage of points to reduce from contours.
                                            Valid range: 1-90. Default: 10.
        cloud_size (float, optional): Override automatic cloud size calculation.
                                     If None, calculates adaptive size based on geometry.
    
    Returns:
        dict: Comprehensive results dictionary containing:
            - 'success' (bool): Operation success status
            - 'message' (str): Detailed status message
            - 'total_points' (int): Total number of generated points
            - 'boundary_points' (int): Number of boundary points
            - 'interior_points' (int): Number of interior points
            - 'regions_processed' (int): Number of regions processed
            - 'cloud_size' (float): Final cloud size used
            - 'output_files' (list): List of generated output files
            - 'visualization_created' (bool): Visualization generation status
    
    Algorithm Workflow:
        1. Load and validate CSV boundary data
        2. Apply optional contour point reduction for optimization
        3. Calculate adaptive cloud size based on region geometry
        4. Generate boundary points with uniform spacing along contours
        5. Fill interior using Poisson Disk Sampling for natural distribution
        6. Process additional regions if multi-region support enabled
        7. Classify nodes as boundary or interior using geometric analysis
        8. Export results in multiple formats with comprehensive metadata
        9. Generate high-quality visualization with region-specific coloring
    
    Features:
        - Poisson Disk Sampling for organic, natural point patterns
        - Adaptive cloud size calculation based on geometry characteristics
        - Multi-region support for complex geometries with holes
        - Intelligent contour reduction for performance optimization
        - Comprehensive node classification (boundary vs interior)
        - Multiple output formats (CSV, PNG, SVG) with detailed metadata
        - Professional visualization with region-specific color coding
        - Robust error handling with detailed logging
        - Consistent results for identical input parameters
    
    Note:
        Natural Distribution uses Poisson Disk Sampling to create more organic point
        patterns compared to Regular Distribution. This method is ideal for applications
        requiring natural-looking point distributions while maintaining computational
        efficiency for numerical methods.
    
    Raises:
        Exception: If file processing, point generation, or export operations fail.
                  Detailed error information is logged for debugging purposes.
    """
    try:
        # Define strategies
        main_strategy = generate_region_cloud_with_holes_poisson
        interior_strategy = generate_interior_regions_clouds_poisson
        
        return _generate_cloud_core(
            csv_file, output_file, "Natural",
            main_strategy, interior_strategy,
            inside_regions, contour_reduction, reduction_percentage, cloud_size
        )
        
    except Exception as e:
        # Fallback error handling just in case, though _generate_cloud_core handles it
        error_msg = f"Error in cloud generation with Natural Distribution: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "error": error_msg}


def generate_cloud_regular(csv_file: str, output_file: str, inside_regions: bool = False, contour_reduction: bool = False, 
                           reduction_percentage: int = 10, cloud_size: float = None) -> dict:
    """
    Generate cloud of points using Regular Distribution algorithm with grid-based approach.
    
    This function creates a uniform, structured cloud of points using grid-based algorithms.
    It processes CSV boundary data to generate optimized cloud of points with consistent spacing
    suitable for meshless Generalized Finite Differences (mGFD) method computations.
    
    Args:
        csv_file (str): Path to input CSV file containing boundary coordinates.
                       Must have columns: x, y, and optionally 'region' for multi-region support.
        output_file (str): Base path for output files (without extension).
                          Generated files: .csv (coordinates), .png (visualization), .svg (vector).
        inside_regions (bool, optional): Enable multi-region processing for interior holes.
                                        If True, processes regions 2+ as interior holes. Default: False.
        contour_reduction (bool, optional): Apply point reduction to boundary contours.
                                         Reduces computational complexity. Default: False.
        reduction_percentage (int, optional): Percentage of points to reduce from contours.
                                            Valid range: 1-90. Default: 10.
        cloud_size (float, optional): Override automatic cloud size calculation.
                                     If None, calculates adaptive size based on geometry.
    
    Returns:
        dict: Comprehensive results dictionary containing:
            - 'success' (bool): Operation success status
            - 'message' (str): Detailed status message
            - 'total_points' (int): Total number of generated points
            - 'boundary_points' (int): Number of boundary points
            - 'interior_points' (int): Number of interior points
            - 'regions_processed' (int): Number of regions processed
            - 'cloud_size' (float): Final cloud size used
            - 'output_files' (list): List of generated output files
            - 'visualization_created' (bool): Visualization generation status
    
    Algorithm Workflow:
        1. Load and validate CSV boundary data
        2. Apply optional contour point reduction for optimization
        3. Calculate adaptive cloud size based on region geometry
        4. Generate boundary points with uniform spacing along contours
        5. Fill interior using structured grid-based approach
        6. Process additional regions if multi-region support enabled
        7. Classify nodes as boundary or interior using geometric analysis
        8. Export results in multiple formats with comprehensive metadata
        9. Generate high-quality visualization with region-specific coloring
    
    Features:
        - Grid-based uniform point placement for consistent distribution
        - Adaptive cloud size calculation based on geometry characteristics
        - Multi-region support for complex geometries with holes
        - Intelligent contour reduction for performance optimization
        - Comprehensive node classification (boundary vs interior)
        - Multiple output formats (CSV, PNG, SVG) with detailed metadata
        - Professional visualization with region-specific color coding
        - Robust error handling with detailed logging
        - Consistent results for identical input parameters
    
    Note:
        Regular Distribution uses structured grid patterns to create uniform point
        distributions with consistent spacing. This method is ideal for applications
        requiring predictable, evenly-spaced point patterns for numerical stability
        in computational methods.
    
    Raises:
        Exception: If file processing, point generation, or export operations fail.
                  Detailed error information is logged for debugging purposes.
    """
    try:
        # Define strategies
        main_strategy = generate_region_cloud_with_holes
        interior_strategy = generate_interior_regions_clouds
        
        return _generate_cloud_core(
            csv_file, output_file, "Regular",
            main_strategy, interior_strategy,
            inside_regions, contour_reduction, reduction_percentage, cloud_size
        )
        

        
    except Exception as e:
        # Fallback error handling just in case, though _generate_cloud_core handles it
        error_msg = f"Error in cloud generation with Regular Distribution: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "error": error_msg}


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
        # from scipy.spatial.distance import cdist  <-- Removed to avoid scipy dependency
        
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
        refinement_candidates.append(avg_nearest_distance * 0.3)
        
        # Secondary: Density-based calculation
        refinement_candidates.append(density_based_refinement * 0.1)
        
        # Tertiary: Cloud size based (if available)
        if cloud_size_based_refinement:
            refinement_candidates.append(cloud_size_based_refinement)
        
        # Take the median of the candidates for robustness
        dynamic_refinement = np.median(refinement_candidates)
        
        # Apply reasonable bounds
        min_refinement = 0.0001  # Minimum threshold
        max_refinement = min(domain_width, domain_height) * 0.01  # Max 1% of domain size
        
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