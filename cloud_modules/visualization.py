"""
Visualization Module - Point Cloud Rendering

This module handles the generation of high-quality visual representations of the
generated point clouds. It uses Matplotlib to create static images (PNG) and
vector graphics (SVG) that clearly distinguish between regions and node types.

Core Functionality:
1. Plotting: Renders points with region-specific colors.
2. Styling: Distinguishes boundary vs. interior nodes via markers/colors.
3. Layout: Optimizes plot layout, legends, and aspect ratios.
4. Export: Saves visualizations in multiple formats (PNG, SVG).

Key Features:
- Professional-grade visualization suitable for publication
- distinct color palettes for up to 7 unique regions (cycling)
- External legend placement to prevent data occlusion
- Non-interactive backend usage for server-side stability

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- Matplotlib
- NumPy
- Logging
"""

import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import logging

def create_visualization(points: np.ndarray, regions_list: list[int], output_base: str, classifications: list[str] = None) -> bool:
    """
    Create a visualization of the generated point cloud with differentiated colors for boundary and interior nodes.
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
                          edgecolors='none', label=f'{region_id} Boundary', zorder=10)
            
            # Interior points
            interior_mask = region_mask & (classifications_arr != 'boundary')
            if np.any(interior_mask):
                interior_pts = points[interior_mask]
                color_idx = (region_id - 1) % len(interior_colors)
                ax.scatter(interior_pts[:, 0], interior_pts[:, 1], 
                          c=[interior_colors[color_idx]], s=5, marker='.', 
                          alpha=0.8, edgecolors='none', label=f'{region_id} Interior', zorder=5)

        # Add title and adjust layout
        plt.title(f'Generated Point Cloud\n{len(points)} Total Nodes', fontsize=14)
        
        # Add legend outside the plot area to prevent overlap
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', markerscale=1.5)
        
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
