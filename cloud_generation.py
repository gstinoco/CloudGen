"""Cloud of Points Generation Module for mGFD CloudGenerator

This module provides advanced functionality for generating optimized clouds of points
using the generator for the meshless Generalized Finite Differences (mGFD) method.
It processes CSV files containing contour coordinates and generates high-quality
clouds of points with adaptive sizing and comprehensive visualization capabilities.

Key Features:
- Adaptive cloud sizing based on contour geometry
- Multi-region support with independent processing
- Advanced node classification (interior, boundary, interface)
- Comprehensive visualization with PNG and SVG output
- Memory-efficient processing with garbage collection
- Robust error handling and logging
- CSV export with detailed node information

Technical Implementation:
- Generator integration for geometric modeling and cloud generation
- NumPy for efficient numerical computations
- Pandas for data manipulation and CSV handling
- Matplotlib for high-quality visualization
- Advanced memory management for large datasets
- Professional logging system for debugging and monitoring

The module supports both single and multi-region geometries, automatically
detecting interfaces between regions and classifying nodes accordingly.
Adaptive sizing algorithms ensure optimal point distribution based on
local geometric features.

Author: Gerardo Tinoco-Guerrero
Date: May 2025
Last Modification: 23 August 2025

Dependencies:
- GMSH >= 4.8.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.3.0
"""

import pandas as pd
import numpy as np
import gmsh
import matplotlib.pyplot as plt
import os
import gc
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cloud_generation.log'),
        logging.StreamHandler()
    ]
)

def safe_gmsh_initialize():
    """
    Safely initialize the generator with optimized settings for cloud generation.
    
    This function initializes the generator with specific configuration
    settings optimized for automated cloud of points generation. It disables
    terminal output, reduces verbosity, and enables expert mode for better
    performance in web application environments.
    
    The function includes comprehensive error handling to ensure graceful
    failure in case of generator initialization issues.
    
    Returns:
        bool: True if the generator was successfully initialized, False otherwise
    
    Raises:
        Logs errors for any generator initialization failures
    
    Note:
        Sets environment variable to prevent signal handling
        conflicts in multi-threaded web applications.
    """
    try:
        os.environ['GMSH_NO_SIGNAL'] = '1'
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.AbortOnError", 0)
        gmsh.option.setNumber("General.ExpertMode", 1)
        
        return True
    except Exception as e:
        logging.error(f"Error initializing the generator: {e}")
        return False

# Global configuration constants for cloud generation algorithms
CLOUD_FACTORS = (
    ("contour_preservation", 0.4),
    ("adaptive_factor", 0.5),
    ("default_cloud_size", 0.005),
)
"""
CLOUD_FACTORS: Tuple of configuration parameters for cloud generation.

This tuple contains key-value pairs that control various aspects of the
cloud generation algorithm:

- contour_preservation (0.4): Controls how closely the generated cloud
  follows the original contour geometry. Lower values create denser
  clouds near boundaries.
  
- adaptive_factor (0.5): Multiplier used in adaptive sizing calculations.
  This factor determines how the local geometry influences cloud density.
  
- default_cloud_size (0.005): Fallback cloud size used when adaptive
  calculations fail or insufficient data is available.

These values have been optimized for typical mGFD applications and provide
a good balance between accuracy and computational efficiency.
"""

def load_regions(csv_file):
    """
    Load and parse region data from a CSV file.
    
    This function reads a CSV file containing point coordinates and region
    identifiers, then organizes the data into separate regions for processing.
    Each region is represented as a tuple of (x, y) coordinate pairs.
    
    The function expects a CSV file with columns:
    - 'x': X-coordinate of the point
    - 'y': Y-coordinate of the point  
    - 'region': Integer identifier for the region
    
    Args:
        csv_file (str): Path to the CSV file containing region data
    
    Returns:
        tuple: Tuple of tuples, where each inner tuple contains (x, y) coordinate
               pairs for a specific region. Returns empty tuple on error.
    
    Raises:
        Logs errors for file reading issues, missing columns, or data format problems
    
    Example:
        >>> regions = load_regions('contours.csv')
        >>> print(f"Loaded {len(regions)} regions")
        Loaded 3 regions
    """
    try:
        df = pd.read_csv(csv_file)
        regions = []
        
        for region_id in sorted(df['region'].unique()):
            region_data = df[df['region'] == region_id]
            points = tuple((row['x'], row['y']) for _, row in region_data.iterrows())
            regions.append(points)
            
        return tuple(regions)
        
    except Exception as e:
        logging.error(f"Error loading regions from {csv_file}: {e}")
        return tuple()

def create_boundary(points):
    """
    Create a closed boundary contour from a sequence of points.
    
    This function ensures that a sequence of points forms a closed contour
    by automatically adding the first point at the end if the contour is
    not already closed. This is essential for geometric operations.
    
    Args:
        points (tuple): Sequence of (x, y) coordinate tuples defining the boundary
    
    Returns:
        tuple: Closed contour with the first point repeated at the end if necessary
    
    Raises:
        ValueError: If fewer than 3 points are provided (insufficient for a contour)
    
    Example:
        >>> points = ((0, 0), (1, 0), (1, 1), (0, 1))
        >>> boundary = create_boundary(points)
        >>> print(len(boundary))  # Will be 5 (original 4 + closing point)
        5
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are needed to create a contour")
    
    if points[0] != points[-1]:
        contour = points + (points[0],)
    else:
        contour = points
    
    return contour

def calc_adaptive_size(regions):
    """
    Calculate adaptive cloud size based on global geometry analysis.
    
    This function analyzes all regions to determine an optimal cloud size
    by calculating the average distance between consecutive points across
    all contours. The resulting size is scaled by the adaptive factor to
    ensure appropriate point density.
    
    The adaptive sizing approach ensures that the cloud density is appropriate
    for the overall geometry scale, providing consistent quality across
    different problem sizes.
    
    Args:
        regions (tuple): Tuple of region contours, each containing (x, y) coordinate pairs
    
    Returns:
        float: Calculated adaptive cloud size, or default size if calculation fails
    
    Note:
        Uses memory-efficient tuple comprehension and explicit garbage collection
        to handle large datasets without memory issues.
    
    Example:
        >>> regions = (((0, 0), (1, 0), (1, 1)), ((2, 0), (3, 0), (3, 1)))
        >>> size = calc_adaptive_size(regions)
        >>> print(f"Adaptive size: {size:.6f}")
        Adaptive size: 0.500000
    """
    total_distances = tuple(
        np.sqrt((region[i+1][0] - region[i][0])**2 + (region[i+1][1] - region[i][1])**2)
        for region in regions
        if len(region) >= 2
        for i in range(len(region) - 1)
    )
    
    if not total_distances:
        return dict(CLOUD_FACTORS)["default_cloud_size"]
    
    avg_distance = np.mean(total_distances)
    cloud_size = avg_distance * dict(CLOUD_FACTORS)["adaptive_factor"]
    del total_distances
    
    return cloud_size

def calc_cloud_sizes(regions):
    """
    Calculate individual cloud sizes for each region based on local geometry.
    
    This function computes region-specific cloud sizes by analyzing the
    local geometry of each contour independently. Each region gets a
    cloud size optimized for its specific geometric characteristics,
    allowing for better adaptation to varying levels of detail.
    
    The function handles edge cases such as regions with insufficient
    points by assigning a reasonable default size.
    
    Args:
        regions (tuple): Tuple of region contours, each containing (x, y) coordinate pairs
    
    Returns:
        list: List of cloud sizes corresponding to each region
    
    Note:
        Regions with fewer than 2 points receive a default size of 0.01.
        Uses memory-efficient tuple comprehension for distance calculations.
    
    Example:
        >>> regions = (((0, 0), (0.1, 0), (0.1, 0.1)), ((0, 0), (1, 0), (1, 1)))
        >>> sizes = calc_cloud_sizes(regions)
        >>> print(f"Region sizes: {[f'{s:.4f}' for s in sizes]}")
        Region sizes: ['0.0500', '0.5000']
    """
    cloud_sizes = []
    
    for region in regions:
        if len(region) < 2:
            cloud_sizes.append(0.01)
            continue
            
        distances = tuple(
            np.sqrt((region[j+1][0] - region[j][0])**2 + (region[j+1][1] - region[j][1])**2)
            for j in range(len(region) - 1)
        )
        
        if distances:
            avg_distance = np.mean(distances)
            cloud_size = avg_distance * dict(CLOUD_FACTORS)["adaptive_factor"]
            cloud_sizes.append(cloud_size)
            del distances
        else:
            cloud_sizes.append(dict(CLOUD_FACTORS)["default_cloud_size"])
    
    return tuple(cloud_sizes)

def export_nodes(node_tags, node_coords, contours, regiones_inside=False, nodes_output='cloud_nodes.csv'):
    """
    Export generated cloud nodes to CSV with comprehensive classification.
    
    This function processes the generated cloud of points and exports them to a CSV file
    with detailed classification information. Each node is classified as interior,
    boundary, or interface based on its geometric position and relationship to
    the domain boundaries and region interfaces.
    
    The function performs advanced node analysis to determine:
    - Boundary nodes (on exterior domain boundaries)
    - Interface nodes (on boundaries between regions)
    - Interior nodes (inside regions)
    - Region assignment for each node
    
    Args:
        node_tags (list): List of node tag identifiers
        node_coords (list): Flattened list of node coordinates [x1, y1, z1, x2, y2, z2, ...]
        contours (tuple): Tuple of region contours used for boundary detection
        regiones_inside (bool, optional): If True, enables interface node detection.
                                        Defaults to False.
        nodes_output (str, optional): Output CSV filename. Defaults to 'cloud_nodes.csv'.
    
    Returns:
        None: Results are saved directly to the specified CSV file
    
    CSV Output Format:
        - x: X-coordinate of the node
        - y: Y-coordinate of the node
        - region: Region identifier (integer)
        - flag: Node classification (0=interior, 1=boundary, 2=interface)
    
    Note:
        Uses memory-efficient processing with explicit garbage collection
        to handle large datasets. Includes comprehensive error handling
        for robust operation in production environments.
    
    Raises:
        Logs errors for entity access issues, file writing problems,
        or memory allocation failures
    """
    try:
        # Get node classification information
        boundary_nodes = set()
        interface_nodes = set()
        node_regions = {}
        
        line_entities = gmsh.model.getEntities(1)
        exterior_boundary_nodes = set()
        
        if len(contours) > 1:
            num_exterior_lines = len(contours[0]) - 1
            for i, (dim, tag) in enumerate(line_entities):
                line_node_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
                if i < num_exterior_lines:
                    exterior_boundary_nodes.update(line_node_tags)
                else:
                    interface_nodes.update(line_node_tags)
        else:
            for dim, tag in line_entities:
                line_node_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
                exterior_boundary_nodes.update(line_node_tags)
        
        boundary_nodes = exterior_boundary_nodes
        
        try:
            physical_groups = gmsh.model.getPhysicalGroups(2)
            for dim, tag in physical_groups:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                for entity in entities:
                    entity_node_tags, _, _ = gmsh.model.mesh.getNodes(2, entity)
                    for node_tag in entity_node_tags:
                        node_regions[node_tag] = tag
        except:
            for node_tag in node_tags:
                node_regions[node_tag] = 1
        
        x_coords = [node_coords[i * 3] for i in range(len(node_tags))]
        y_coords = [node_coords[i * 3 + 1] for i in range(len(node_tags))]
        
        data = []
        for i, node_tag in enumerate(node_tags):
            x, y = x_coords[i], y_coords[i]
            region = node_regions.get(node_tag, 1)
            
            if regiones_inside and node_tag in interface_nodes:
                flag = 2
            elif node_tag in boundary_nodes:
                flag = 1
            else:
                flag = 0
            
            data.append([x, y, region, flag])
        
        del x_coords, y_coords
        del boundary_nodes, interface_nodes, exterior_boundary_nodes
        del node_regions
        
        df = pd.DataFrame(data, columns=['x', 'y', 'region', 'flag'])
        df.to_csv(nodes_output, index=False)
        
        del data, df
        gc.collect()
        
    except Exception as e:
        logging.error(f"Export error: {e}")

def visualize_cloud(node_tags, node_coords, base_filename="cloud_visualization"):
    """
    Generate comprehensive visualization of the cloud of points.
    
    This function creates high-quality visualizations of the generated cloud,
    showing node classification and region distribution. The visualization
    uses different colors and sizes to distinguish between:
    - Interior nodes (colored by region)
    - Boundary nodes (red, larger size)
    - Interface nodes (blue, larger size)
    
    The function generates both PNG (high-resolution raster) and SVG (vector)
    formats for different use cases. The visualization is optimized for
    scientific publication and technical documentation.
    
    Args:
        node_tags (list): List of node tag identifiers
        node_coords (list): Flattened list of node coordinates [x1, y1, z1, x2, y2, z2, ...]
        base_filename (str, optional): Base filename for output files (without extension).
                                     Defaults to "cloud_visualization".
    
    Returns:
        None: Visualization files are saved directly to disk
    
    Output Files:
        - {base_filename}.png: High-resolution raster image (300 DPI)
        - {base_filename}.svg: Scalable vector graphics format
    
    Visualization Features:
        - Equal aspect ratio for accurate geometric representation
        - Color-coded regions for multi-region problems
        - Enhanced visibility for boundary and interface nodes
        - Professional layout suitable for publications
        - Memory-efficient processing with garbage collection
    
    Note:
        Uses matplotlib for rendering with optimized settings for
        scientific visualization. Includes comprehensive error handling
        for robust operation with various node configurations.
    
    Raises:
        Logs errors for entity access issues, plotting failures,
        or file saving problems
    """
    try:
        
        x_nodes = [node_coords[i * 3] for i in range(len(node_tags))]
        y_nodes = [node_coords[i * 3 + 1] for i in range(len(node_tags))]
        
        node_regions = {}
        node_flags = {}
        
        try:
            # Get regions using physical groups
            physical_groups = gmsh.model.getPhysicalGroups(2)
            for dim, tag in physical_groups:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                for entity in entities:
                    entity_node_tags, _, _ = gmsh.model.mesh.getNodes(2, entity)
                    for node_tag in entity_node_tags:
                        node_regions[node_tag] = tag
            
            # Get boundary nodes
            boundary_nodes = set()
            line_entities = gmsh.model.getEntities(1)
            for dim, tag in line_entities:
                line_node_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
                boundary_nodes.update(line_node_tags)
            
            # Get interface nodes (interior boundaries)
            interface_nodes = set()
            if len(physical_groups) > 1:
                for dim, tag in physical_groups[1:]:
                    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                    for entity in entities:
                        boundary_lines = gmsh.model.getBoundary([(2, entity)], combined=False, oriented=False)
                        for line_dim, line_tag in boundary_lines:
                            line_node_tags, _, _ = gmsh.model.mesh.getNodes(abs(line_dim), abs(line_tag))
                            interface_nodes.update(line_node_tags)
            
            # Assign flags
            for node_tag in node_tags:
                if node_tag in interface_nodes:
                    node_flags[node_tag] = 2  # Interface
                elif node_tag in boundary_nodes:
                    node_flags[node_tag] = 1  # Boundary
                else:
                    node_flags[node_tag] = 0  # Interior
                    
        except Exception as e:
            logging.error(f"Error getting node classification: {e}")
            # Default values in case of error
            for node_tag in node_tags:
                node_regions[node_tag] = 1
                node_flags[node_tag] = 0
        
        # Free memory from original coordinates after extraction
        del node_coords
        
        # Configure visualization
        plt.figure(figsize=(12, 8))
        
        # Define colors
        flag_colors = {
            0: 'lightgray',    # Interior nodes
            1: 'red',          # Exterior boundary nodes
            2: 'blue'          # Interior interface nodes
        }
        
        region_colors = {
            1: 'lightblue', 2: 'lightcoral', 3: 'lightgreen', 4: 'lightyellow',
            5: 'lightpink', 6: 'lightgray', 7: 'lavender', 8: 'peachpuff'
        }
        
        # Group nodes by type
        flag_groups = {0: [], 1: [], 2: []}
        
        for i, node_tag in enumerate(node_tags):
            flag = node_flags.get(node_tag, 0)
            region = node_regions.get(node_tag, 1)
            
            if flag == 0:  # Interior nodes - color by region
                color = region_colors.get(region, 'lightgray')
                flag_groups[0].append((x_nodes[i], y_nodes[i], color))
            else:  # Boundary or interface nodes - color by flag
                color = flag_colors[flag]
                flag_groups[flag].append((x_nodes[i], y_nodes[i], color))
        
        # Draw interior nodes by region
        if flag_groups[0]:
            region_data = {}
            for x, y, color in flag_groups[0]:
                if color not in region_data:
                    region_data[color] = {'x': [], 'y': []}
                region_data[color]['x'].append(x)
                region_data[color]['y'].append(y)
            
            for color, data in region_data.items():
                plt.scatter(data['x'], data['y'], c=color, s=1, alpha=0.7)
        
        # Draw boundary nodes (more visible)
        if flag_groups[1]:
            x_boundary = [point[0] for point in flag_groups[1]]
            y_boundary = [point[1] for point in flag_groups[1]]
            plt.scatter(x_boundary, y_boundary, c='red', s=3, alpha=0.8, label='Boundary nodes')
        
        # Draw interface nodes (more visible)
        if flag_groups[2]:
            x_interface = [point[0] for point in flag_groups[2]]
            y_interface = [point[1] for point in flag_groups[2]]
            plt.scatter(x_interface, y_interface, c='blue', s=3, alpha=0.8, label='Interface nodes')
        
        # Configure plot
        plt.axis('equal')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save files (base_filename already includes output/ path)
        png_file = f"{base_filename}.png"
        svg_file = f"{base_filename}.svg"
        
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')

        plt.close()
        
    except Exception as e:
        logging.error(f"Visualization error: {e}")

def generate_cloud_of_points(csv_file, regiones_inside=False, timestamp=None, original_base_name=None):
    """
    Generate optimized cloud of points from contour data using the generator.
    
    This is the main function of the module that orchestrates the complete
    cloud generation process. It reads contour data from a CSV file, creates
    geometric models using the generator, generates adaptive clouds of points, and
    produces comprehensive output including CSV data and visualizations.
    
    The function supports both single-region and multi-region geometries,
    with automatic detection of interfaces and adaptive sizing algorithms
    that optimize point distribution based on local geometric features.
    
    Process Overview:
    1. Load and validate region data from CSV
    2. Calculate adaptive cloud sizes for each region
    3. Create geometric model using the generator
    4. Generate cloud of points with optimized distribution
    5. Classify nodes (interior, boundary, interface)
    6. Export results to CSV with detailed classification
    7. Generate high-quality visualizations (PNG + SVG)
    
    Args:
        csv_file (str): Path to CSV file containing contour coordinates.
                       Expected columns: 'x', 'y', 'region'
        regiones_inside (bool, optional): Enable multi-region processing with
                                        interface detection. Defaults to False.
        timestamp (str, optional): Custom timestamp for output files.
                                 If None, current datetime is used.
        original_base_name (str, optional): Custom base name for output files.
                                          If None, derived from input filename.
    
    Returns:
        bool: True if cloud generation completed successfully, False otherwise
    
    Output Files:
        - {base_name}_nodes_{timestamp}.csv: Node data with classification
        - {base_name}_visualization_{timestamp}.png: High-resolution visualization
        - {base_name}_visualization_{timestamp}.svg: Vector visualization
    
    Technical Features:
        - Adaptive sizing based on local geometry analysis
        - Memory-efficient processing with garbage collection
        - Robust error handling and recovery
        - Professional logging for debugging and monitoring
        - Generator integration with optimized settings
        - Multi-format output for different use cases
    
    Example:
        >>> success = generate_cloud_of_points('contours.csv', regiones_inside=True)
        >>> if success:
        ...     print("Cloud generation completed successfully")
        Cloud generation completed successfully
    
    Note:
        Requires the generator for geometric modeling and cloud generation.
        All output files are saved to the 'output' directory, which is
        created automatically if it doesn't exist.
    
    Raises:
        Logs errors for file access issues, generator failures, memory problems,
        or any other exceptions during the generation process
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if original_base_name is not None:
        base_name = original_base_name
    else:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    regions = load_regions(csv_file)
    if not regions:
        logging.error("Could not load regions")
        return False
    
    if regiones_inside:
        cloud_sizes = calc_cloud_sizes(regions)
    else:
        cloud_size = calc_adaptive_size(regions)
        cloud_sizes = tuple(cloud_size for _ in range(len(regions)))
    
    contours = []
    
    for i, region in enumerate(regions):
        try:
            contour = create_boundary(region)
            contours.append(contour)
        except Exception as e:
            logging.error(f"Error in region {i+1}: {e}")
            contours.append(tuple())
    
    contours = tuple(contours)
    
    try:
        if not safe_gmsh_initialize():
            logging.error("Error: Could not initialize")
            return False
        
        try:
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)
        except:
            pass
        
        gmsh.model.add("region_cloud")
        
        point_tags = []
        line_tags = []
        curve_loop_tags = []
        
        for i, contour in enumerate(contours):
            if not contour:
                continue
            
            region_cloud_size = cloud_sizes[i] if i < len(cloud_sizes) else cloud_sizes[0]
            
            region_point_tags = []
            preservation_factor = dict(CLOUD_FACTORS)["contour_preservation"]
            
            for x, y in contour[:-1]:
                tag = gmsh.model.geo.addPoint(x, y, 0, region_cloud_size * preservation_factor)
                region_point_tags.append(tag)
            
            region_line_tags = [
                gmsh.model.geo.addLine(
                    region_point_tags[j], 
                    region_point_tags[(j + 1) % len(region_point_tags)]
                )
                for j in range(len(region_point_tags))
            ]
            
            curve_loop_tag = gmsh.model.geo.addCurveLoop(region_line_tags)
            curve_loop_tags.append(curve_loop_tag)
            
            point_tags.extend(region_point_tags)
            line_tags.extend(region_line_tags)
        
        if not curve_loop_tags:
            logging.error("Could not create valid contours")
            gmsh.finalize()
            return False
        
        main_loop = curve_loop_tags[0]
        hole_loops = curve_loop_tags[1:] if len(curve_loop_tags) > 1 else []
        
        surface_tag = gmsh.model.geo.addPlaneSurface([main_loop] + hole_loops)
        
        if regiones_inside and hole_loops:
            for i, hole_loop in enumerate(hole_loops):
                hole_surface_tag = gmsh.model.geo.addPlaneSurface([hole_loop])
                gmsh.model.addPhysicalGroup(2, [hole_surface_tag], i + 2)
                gmsh.model.setPhysicalName(2, i + 2, f"Interior_{i+1}")
        
        gmsh.model.addPhysicalGroup(2, [surface_tag], 1)
        gmsh.model.setPhysicalName(2, 1, "Principal")
        
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_tags, element_node_tags = gmsh.model.mesh.getElementsByType(2)
        
        del element_tags, element_node_tags
        
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        csv_nodes_file = os.path.join(output_dir, f"{base_name}_nodes_{timestamp}.csv")
        export_nodes(node_tags, node_coords, contours, regiones_inside, csv_nodes_file)
        
        visualization_base = os.path.join(output_dir, f"{base_name}_visualization_{timestamp}")
        visualize_cloud(node_tags, node_coords, visualization_base)
        
        del node_tags, node_coords
        del contours, regions, cloud_sizes
        del point_tags, line_tags, curve_loop_tags
        
        gc.collect()
        
        gmsh.finalize()
        return True
        
    except Exception as e:
        logging.error(f"Cloud of points generation failed for {csv_file}: {str(e)}")
        
        gc.collect()
        try:
            gmsh.finalize()
        except:
            pass
        return False