"""
Cloud Generation Module - Advanced Point Cloud Generation System

This module provides comprehensive functionality for generating optimized point clouds
from CSV boundary data for use with the meshless Generalized Finite Differences (mGFD) method.
The module implements two advanced distribution algorithms with intelligent point reduction
and multi-region support for complex geometries.

Core Functionality:
1. CSV boundary point processing
2. Optional contour point reduction for optimization
3. Point cloud generation for main region (Region 1) with consistent results
4. Multi-region support for interior holes and complex geometries
5. Intelligent node classification (boundary vs interior)
6. Comprehensive visualization and export capabilities

Author: Gerardo Tinoco-Guerrero
Date: May, 2025
Last Modification: March, 2026

Dependencies:
- NumPy
- Cloud Modules (Internal Package)
- Reduce Points (Internal Module)
"""

import os
import logging
import tempfile
import numpy as np
from cloud_modules import reduction as reduce_points

# Import from new modules
from cloud_modules.config import setup_logging
from cloud_modules.data_processing import load_regions
from cloud_modules.point_generation import (
    generate_region_cloud_with_holes,
    generate_region_cloud_with_holes_poisson,
    generate_interior_regions_clouds,
    generate_interior_regions_clouds_poisson
)
from cloud_modules.classification import classify_nodes
from cloud_modules.visualization import create_visualization
from cloud_modules.export import export_to_csv

# Setup logging
setup_logging()

def _generate_cloud_core(csv_file: str, output_file: str, method_name: str, main_region_strategy: callable, interior_regions_strategy: callable, 
                         inside_regions: bool = False, cloud_size: float = None, density_multiplier: float = 1.0) -> dict:
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
        logging.info(f"Interior regions: {inside_regions}")
        
        working_csv_file = csv_file
        
        regions = load_regions(working_csv_file)
        if not regions:
            return {"success": False, "error": "Failed to load regions from CSV"}
        
        logging.info(f"Loaded {len(regions)} region(s)")
        
        main_region = regions[0]
        hole_regions = regions[1:] if len(regions) > 1 else []
        
        # Calculate explicit cloud size with density multiplier
        from cloud_modules.utils import calculate_cloud_size
        if cloud_size is None:
            base_cloud_size = calculate_cloud_size(main_region)
            # Higher density multiplier = smaller cloud size = more points
            cloud_size = base_cloud_size / density_multiplier
            logging.info(f"Calculated base cloud size: {base_cloud_size}, final with multiplier: {cloud_size}")
            
        # Execute Main Region Strategy
        # Pass inside_regions to control hole boundary generation
        main_boundary, main_interior, actual_cloud_size = main_region_strategy(main_region, hole_regions, cloud_size, inside_regions)
        
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
        
        # Note: classify_nodes now imports from cloud_modules.classification
        # We need to pass original_regions_contours (regions) to it for accurate classification
        classifications = classify_nodes(all_points, all_regions, regions, actual_cloud_size, inside_regions=inside_regions)
        
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
        logging.error(error_msg)
        return {"success": False, "error": error_msg}

def generate_cloud_natural(csv_file: str, output_file: str, inside_regions: bool = False,
                           cloud_size: float = None, density_multiplier: float = 1.0) -> dict:
    """
    Generate point cloud using Natural Distribution algorithm with Poisson Disk Sampling.
    
    This function creates a natural, organic point cloud using advanced Poisson Disk
    Sampling algorithm. It processes CSV boundary data to generate optimized point cloud
    suitable for meshless Generalized Finite Differences (mGFD) method computations.
    
    Args:
        csv_file (str): Path to input CSV file containing boundary coordinates.
                       Must have columns: x, y, and optionally 'region' for multi-region support.
        output_file (str): Base path for output files (without extension).
                          Generated files: .csv (coordinates), .png (visualization), .svg (vector).
        inside_regions (bool, optional): Enable multi-region processing for interior holes.
                                        If True, processes regions 2+ as interior holes. Default: False.

        cloud_size (float, optional): Override automatic cloud size calculation.
                                     If None, calculates adaptive size based on geometry.
    
    Returns:
        dict: Comprehensive results dictionary
    """
    try:
        # Define strategies
        main_strategy = generate_region_cloud_with_holes_poisson
        interior_strategy = generate_interior_regions_clouds_poisson
        
        return _generate_cloud_core(
            csv_file, output_file, "Natural",
            main_strategy, interior_strategy,
            inside_regions, cloud_size, density_multiplier
        )
        
    except Exception as e:
        error_msg = f"Error in cloud generation with Natural Distribution: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "error": error_msg}

def generate_cloud_regular(csv_file: str, output_file: str, inside_regions: bool = False,
                           cloud_size: float = None, density_multiplier: float = 1.0) -> dict:
    """
    Generate point cloud using Regular Distribution algorithm with grid-based approach.
    
    This function creates a uniform, structured point cloud using grid-based algorithms.
    It processes CSV boundary data to generate optimized point cloud with consistent spacing
    suitable for meshless Generalized Finite Differences (mGFD) method computations.
    
    Args:
        csv_file (str): Path to input CSV file containing boundary coordinates.
                       Must have columns: x, y, and optionally 'region' for multi-region support.
        output_file (str): Base path for output files (without extension).
                          Generated files: .csv (coordinates), .png (visualization), .svg (vector).
        inside_regions (bool, optional): Enable multi-region processing for interior holes.
                                        If True, processes regions 2+ as interior holes. Default: False.

        cloud_size (float, optional): Override automatic cloud size calculation.
                                     If None, calculates adaptive size based on geometry.
    
    Returns:
        dict: Comprehensive results dictionary
    """
    try:
        # Define strategies
        main_strategy = generate_region_cloud_with_holes
        interior_strategy = generate_interior_regions_clouds
        
        return _generate_cloud_core(
            csv_file, output_file, "Regular",
            main_strategy, interior_strategy,
            inside_regions, cloud_size, density_multiplier
        )
        
    except Exception as e:
        error_msg = f"Error in cloud generation with Regular Distribution: {str(e)}"
        logging.error(error_msg)
        return {"success": False, "error": error_msg}
