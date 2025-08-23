# --- Imports ---
# Standard library imports for basic functionality
from datetime import datetime                                                                           # For timestamp generation and date handling
from threading import Timer                                                                             # For scheduling delayed operations
import logging                                                                                          # For application logging
import os                                                                                               # For file and path operations

# Third-party library imports for geometric operations and data handling
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point, Polygon                                                             # For creating and manipulating geometric objects
from shapely.ops import unary_union                                                                     # For combining multiple geometric objects
from shapely.prepared import prep                                                                       # For geometry preparation
from functools import partial
from rtree import index                                                                                 # For spatial indexing
import pandas as pd                                                                                     # For data manipulation and analysis
import numpy as np                                                                                      # For numerical operations
import dmsh                                                                                             # For mesh generation
import csv                                                                                              # For CSV file operations

# Matplotlib imports for plotting and visualization
import matplotlib
matplotlib.use('Agg')                                                                                   # Use Agg backend for non-GUI environments (server-side)
import matplotlib.pyplot as plt                                                                         # For creating plots
from matplotlib.lines import lineStyles                                                                 # For line style customization
from matplotlib.colors import to_rgb                                                                    # For color conversion
from matplotlib.offsetbox import OffsetImage, AnnotationBbox                                            # For adding images to plots
import matplotlib.image as mpimg                                                                        # For image handling in plots

# Custom logger import
from utils import app_logger, log_error                                                                 # Import custom logging utilities

'''
======================================
1. DATA PROCESSING
======================================
'''

def process_csv(file_path):
    """
    Reads and processes a CSV file to extract point data grouped by region.

    Parameters:
        file_path (str):                                Path to the CSV file.

    Returns:
        dict:                                           A dictionary where keys are region flags and values 
                                                        are lists of (x, y) coordinate pairs.

    Raises:
        ValueError:                                     If the CSV file does not contain the required columns.
        FileNotFoundError:                             If the specified file does not exist.
        pd.errors.EmptyDataError:                      If the CSV file is empty.
        RuntimeError:                                   If any other error occurs during file processing.
    """

    try:
        # Verify file existence before processing
        if not os.path.exists(file_path):                                                               # Check if the file exists at the specified path
            error_msg = f"CSV file does not exist at path: {file_path}"                                 # Prepare error message
            log_error(FileNotFoundError(error_msg), "Error processing CSV")                             # Log the error with custom error handler
            raise FileNotFoundError(error_msg)                                                          # Raise exception to caller

        # Load and validate CSV file
        app_logger.info(f"Starting CSV file processing: {file_path}")                                   # Log start of file processing
        df = pd.read_csv(file_path)                                                                     # Read CSV file into pandas DataFrame

        # Check if the DataFrame is empty
        if df.empty:                                                                                    # Verify if the CSV file has any data
            error_msg = "The CSV file is empty"                                                         # Prepare empty file error message
            log_error(pd.errors.EmptyDataError(error_msg), "Error processing CSV")                      # Log empty file error
            raise pd.errors.EmptyDataError(error_msg)                                                   # Raise empty file exception

        # Define and validate required columns
        required_columns = {'x', 'y', 'flag'}                                                           # Set of required column names

        # Ensure all required columns are present
        if not required_columns.issubset(df.columns):                                                   # Check if all required columns exist
            error_msg = "CSV file must contain columns: 'x', 'y', 'flag'"                               # Prepare missing columns error
            log_error(ValueError(error_msg), "Error processing CSV")                                    # Log missing columns error
            raise ValueError(error_msg)                                                                 # Raise missing columns exception
        
        # Extract only necessary columns
        df = df[['x', 'y', 'flag']]                                                                     # Select only the columns we need

        # Validate numeric data in coordinates
        if not pd.to_numeric(df['x'], errors='coerce').notnull().all():                                 # Check if all x-coordinates are numeric
            error_msg = "Column 'x' contains non-numeric values"                                        # Prepare invalid x-coordinate error
            log_error(ValueError(error_msg), "Error processing CSV")                                    # Log invalid x-coordinate error
            raise ValueError(error_msg)                                                                 # Raise invalid x-coordinate exception

        if not pd.to_numeric(df['y'], errors='coerce').notnull().all():                                 # Check if all y-coordinates are numeric
            error_msg = "Column 'y' contains non-numeric values"                                        # Prepare invalid y-coordinate error
            log_error(ValueError(error_msg), "Error processing CSV")                                    # Log invalid y-coordinate error
            raise ValueError(error_msg)                                                                 # Raise invalid y-coordinate exception

        # Group points by region and convert to coordinate pairs
        points_by_region = (                                                                            # Create dictionary of points by region
            df.groupby('flag', group_keys=False)                                                        # Group data by region flag
            .apply(lambda group: group[['x', 'y']].values.tolist(), include_groups=False)               # Convert each group to list of coordinates
            .to_dict()                                                                                  # Convert to dictionary format
        )

        # Log successful processing and return results
        app_logger.info(f"CSV file processed successfully: {len(points_by_region)} regions found")      # Log success
        return points_by_region                                                                         # Return processed data

    except (pd.errors.EmptyDataError, ValueError, FileNotFoundError) as e:                              # Handle expected exceptions
        raise e                                                                                         # Re-raise expected exceptions
    except Exception as e:                                                                              # Handle unexpected exceptions
        error_msg = f"Error inesperado al procesar el archivo: {str(e)}"                                # Prepare unexpected error message
        log_error(e, "Error al procesar CSV")                                                           # Log unexpected error
        raise RuntimeError(error_msg)                                                                   # Raise as RuntimeError

def generate_polygons(points_by_region):
    """
    Generates polygons for each region based on provided point data.

    Parameters:
        points_by_region (dict):                        Dictionary where keys are region identifiers and 
                                                        values are lists of (x, y) coordinates.

    Returns:
        dict:                                           Dictionary where keys are region identifiers and 
                                                        values are Shapely Polygon objects.
    """

    polygons_by_region = {}                                                                             # Initialize dictionary to store polygons by region.
    
    for region, points in points_by_region.items():                                                     # Iterate over each region and its points.
        # Remove duplicate points and validate input.
        points = pd.DataFrame(points, columns = ['x', 'y']).drop_duplicates()                           # Convert to DataFrame and remove duplicate coordinates.
        
        try:
            # Ensure the region has enough points to form a polygon.
            if len(points) < 3:                                                                         # A polygon requires at least 3 unique points.
                raise ValueError(f"Region {region} has less than 3 unique points, cannot form a polygon.")
            
            # Create a polygon using Shapely.
            polygon = Polygon(points.values)                                                            # Convert the points into a Polygon object.

            # Validate the polygon and apply Convex Hull if necessary.
            if not polygon.is_valid:                                                                    # Check if the polygon is valid.
                print(f"Polygon for region {region} is invalid. Attempting to create a Convex Hull.")
                polygon = Polygon(points.values).convex_hull                                            # Use Convex Hull as a fallback.
            
            # Store the resulting polygon in the dictionary.
            polygons_by_region[region] = polygon                                                        # Assign the polygon to its corresponding region.
        except Exception as e:
            print(f"Error processing region {region}: {e}")                                             # Print an error message if processing fails.

    return polygons_by_region                                                                           # Return the dictionary containing the polygons.

'''
======================================
2. REGION ANALYSIS AND VALIDATION
======================================
'''

def test_region_containment(polygon_1, polygon_2):
    """
    Tests if one polygon is contained within another.

    Parameters:
        polygon_1 (Polygon):                            The main polygon (larger region).
        polygon_2 (Polygon):                            The secondary polygon (potentially inside polygon_1).

    Returns:
        bool:                                           True if polygon_2 is within polygon_1 or if they intersect
                                                        and polygon_1 is larger. False otherwise.

    Raises:
        ValueError:                                     If any of the provided polygons are None.
    """

    # Validate that both polygons are provided.
    if polygon_1 is None or polygon_2 is None:                                                          # Ensure both polygons exist.
        raise ValueError("The provided polygons are not valid.")                                        # Raise an error if any is missing.
    
    # Check if polygon_2 is completely within polygon_1
    check = polygon_2.within(polygon_1) or (polygon_2.intersects(polygon_1) and polygon_1.area > polygon_2.area)

    return check                                                                                        # Return True if polygon_2 is inside polygon_1 or if they intersect and polygon_1 is larger.

def test_all_region_containments(polygons_by_region):
    """
    Tests containment relationships between multiple regions using spatial indexing for optimization.

    Parameters:
        polygons_by_region (dict):                      Dictionary where keys are region identifiers and 
                                                        values are Shapely Polygon objects.

    Returns:
        dict:                                           Dictionary where keys are region identifiers and 
                                                        values are lists of other regions that they contain.
    """

    # Initialize spatial index for efficient querying
    idx = index.Index()                                                                                 # Create R-tree spatial index
    for region_id, polygon in polygons_by_region.items():                                               # Add each region to index
        idx.insert(region_id, polygon.bounds)                                                           # Insert region bounds

    # Initialize results dictionary
    containment_results = {region: [] for region in polygons_by_region}                                 # Create empty lists for each region

    # Prepare polygons for optimized operations
    prepared_polygons = {region: prep(polygon) for region, polygon in polygons_by_region.items()}       # Preprocess geometries

    # Check containment relationships
    for region_1 in polygons_by_region:                                                                 # Iterate through each region
        polygon_1 = polygons_by_region[region_1]                                                        # Get first region's polygon
        prepared_polygon_1 = prepared_polygons[region_1]                                                # Get prepared version

        # Find potential containing regions
        for region_2 in [r for r in idx.intersection(polygon_1.bounds) if r != region_1]:               # Use spatial index
            polygon_2 = polygons_by_region[region_2]                                                    # Get second region's polygon
            
            # Test containment relationship
            if prepared_polygon_1.contains(polygon_2) or (prepared_polygon_1.intersects(polygon_2) and polygon_1.area > polygon_2.area):
                                                                                                        # Check if region_1 contains region_2 Or if they intersect and region_1 is larger
                containment_results[region_1].append(region_2)                                          # Add to containment list

    return containment_results                                                                          # Return containment relationships

def remove_duplicate_containments(containment_results):
    """
    Removes redundant containment relationships, ensuring only direct containments are kept.

    Parameters:
        containment_results (dict):                     Dictionary where keys are region identifiers and 
                                                        values are lists of regions they contain.

    Returns:
        dict:                                           Dictionary with duplicate containment relationships removed.
    """

    # Iterate over each region to clean up redundant containment entries.
    for region in containment_results:                                                                  # Loop through each region.
        direct_contained = set(containment_results[region])                                             # Get the directly contained regions.

        for subregion in list(direct_contained):                                                        # Iterate through each directly contained region.
            nested = set(containment_results.get(subregion, []))                                        # Get regions contained by the subregion.
            direct_contained -= nested                                                                  # Remove nested containments, keeping only direct ones.
        
        containment_results[region] = list(direct_contained)                                            # Update the dictionary with cleaned results.

    return containment_results                                                                          # Return the refined containment dictionary.

'''
======================================
3. CLOUD OF POINTS GENERATION
======================================
'''

def distance(x, y, hole_coordinates = None):
    """
    Computes the average distance between consecutive points in the main boundary and optional holes.

    Parameters:
        - x (list or array):                            List of x-coordinates of the boundary points.
        - y (list or array):                            List of y-coordinates of the boundary points.
        - hole_coordinates (list of tuples, optional):  A list containing hole coordinates as (x, y) tuples.

    Returns:
        - float:                                        The average distance between consecutive points in the boundary, including holes if provided.
    """

    # Combine main boundary coordinates into a single array.
    coords = np.column_stack((x, y))                                                                    # Create an array with main boundary coordinates (x, y).
    
    # Add hole coordinates, if provided.
    if hole_coordinates:                                                                                # If hole coordinates are provided.
        for hx, hy in hole_coordinates:                                                                 # For each of the coordinate sets.
            hole_coords = np.column_stack((hx, hy))                                                     # Create an array for each hole's coordinates.
            coords      = np.vstack((coords, hole_coords))                                              # Append hole coordinates to the main array.
    
    # Compute the distances.
    dists = np.sqrt(np.sum(np.diff(coords, axis = 0)**2, axis = 1))                                     # Compute Euclidean distances between consecutive points.
    
    return np.mean(dists)                                                                               # Return the average distance.

def generate_grid(min_x, max_x, min_y, max_y, spacing):
    """
    Generates a structured grid of points within a given domain.

    Parameters:
        min_x (float):                                  Minimum x-coordinate of the bounding box.
        max_x (float):                                  Maximum x-coordinate of the bounding box.
        min_y (float):                                  Minimum y-coordinate of the bounding box.
        max_y (float):                                  Maximum y-coordinate of the bounding box.
        spacing (float):                                Distance between points in the grid.

    Returns:
        np.array:                                       Array of grid points.
    """

    # Generate x and y coordinate arrays.
    x_coords = np.arange(min_x, max_x, spacing)                                                         # Generate x-coordinates.
    y_coords = np.arange(min_y, max_y, spacing)                                                         # Generate y-coordinates.
    
    # Create a 2D grid of points.
    grid = np.meshgrid(x_coords, y_coords)                                                              # The structured grid is created within the region.

    # Flatten the grid into an array of (x, y) coordinate pairs.
    grid_points = np.array(grid).T.reshape(-1, 2)                                                       # Flatten the grid to obtain the grind points in the proper order.

    return grid_points                                                                                  # Create a grid of points.

def CreateCloud(xb, yb, h_coor_sets, dist, rand, region_flag, method = 1):
    """
    Generates a cloud of points for a given region, considering boundaries and holes.

    Parameters:
        xb (list or array):                             X-coordinates of the main boundary.
        yb (list or array):                             Y-coordinates of the main boundary.
        h_coor_sets (list of tuples):                   List of hole coordinates, each as a tuple of (hx, hy).
        dist (float):                                   Distance between points in the generated cloud.
        rand (float):                                   Random perturbation factor for generated points.
        region_flag (int):                              Identifier flag for the region.
        method (int, optional):                         Point generation method (0: dmsh, 1: grid-based). Default is 1.

    Returns:
        np.array:                                       Array of points in the cloud, including boundary, holes, and generated points.
    """

    # Create a polygon representing the main boundary.
    boundary_polygon = Polygon(np.column_stack((xb, yb)))                                               # Convert boundary coordinates into a Polygon object.

    # Create polygons for each hole and unify them.
    holes       = [Polygon(np.column_stack((hx, hy))) for hx, hy in h_coor_sets]                        # Convert each hole coordinate set into a Polygon.
    holes_union = unary_union(holes) if holes else Polygon()                                            # Unify all holes into a single geometric object.
    
    # Check the method for cloud generation.
    if method == 0:
        # Method 0: Use dmsh for point generation.

        # Create the geometric domain using dmsh.
        geo = dmsh.Polygon(np.column_stack((xb, yb)))                                                   # Define the main boundary in dmsh.
        for hx, hy in h_coor_sets:                                                                      # For each of the given holes.
            geo -= dmsh.Polygon(np.column_stack((hx, hy)))                                              # Subtract hole polygons from the main domain.
        
        # Generate points using dmsh.
        X, _ = dmsh.generate(geo, dist)                                                                 # Generate the mesh with specified point spacing.
        
        # Classify the generated points.
        points = [Point(p[0], p[1]) for p in X]                                                         # Convert to Point objects.
        A      = np.zeros([len(points), 1])                                                             # Initialize an array for point classification.
        cloud  = np.column_stack((X, np.full((len(X), 1), region_flag), A))                             # Combine coordinates, region flag, and classification.
    elif method == 1:
        # Method 1: Use a structured grid and filter points.

        # Define the bounding box for grid generation.
        min_x, min_y, max_x, max_y = boundary_polygon.bounds                                            # Get the boundary limits.
        grid_points = generate_grid(min_x, max_x, min_y, max_y, dist)                                   # Generate a structured grid of points.
        
        # Filter points that are inside the boundary but outside the holes.
        generated_points = [
            [x, y, region_flag, 0] for x, y in grid_points                                              # Assign region_flag and label as "interior" (0).
            if boundary_polygon.contains(Point(x, y)) and not holes_union.contains(Point(x, y))         # Keep points inside the boundary but outside holes.
        ]
        cloud = np.array(generated_points)                                                              # Convert list to a NumPy array.
    else:
        raise ValueError("Unsupported method. Use '0' for dmsh or '1' for grid-based approach.")        # Raise an error if an invalid method is provided.
    
    # Add boundary and hole points to the cloud.
    points = [[x, y, region_flag, 1] for x, y in zip(xb, yb)]                                           # Mark boundary points with label 1.
    for hx, hy in h_coor_sets:                                                                          # For each of the given holes.
        points.extend([[x, y, region_flag, 2] for x, y in zip(hx, hy)])                                 # Mark hole points with label 2.
    
    # Merge boundary, hole, and generated points.
    cloud = np.vstack((points, cloud)) if cloud.size else np.array(points)                              # Ensure cloud includes all required points.
    
    # Apply random perturbation if required.
    if rand != 0 and cloud.size:
        mask = cloud[:, 3] == 0                                                                         # Select only generated points (label 0).
        perturbation = 0.5*dist*(np.random.rand(np.sum(mask), 2) - 0.5)                                 # Compute a random displacement within a range.
        cloud[mask, 0:2] += perturbation                                                                # Apply the displacement to the x and y coordinates.
    
    return cloud                                                                                        # Return the final cloud of points.

def generate_clouds_for_all_regions(polygons_by_region, containment_results, num, rand, mod, gen):
    """
    Generates clouds of points for all specified regions, considering boundaries and holes.

    Parameters:
        polygons_by_region (dict):                      Dictionary where keys are region identifiers and 
                                                        values are Shapely Polygon objects.
        containment_results (dict):                     Dictionary mapping each region to a list of contained regions (holes).
        num (int):                                      Factor to control point density.
        rand (float):                                   Random perturbation factor for generated points.
        mod (int):                                      Method to generate the cloud of points (0: dmsh, 1: grid-based).
        gen (int):                                      Determines whether to generate all clouds or stop after one.

    Returns:
        np.array:                                       Array containing all generated cloud of points.
                                                        Returns an empty array if no clouds are generated.
    """

    all_points = []                                                                                     # List to store generated cloud of points.
    
    def process_region(region_data):
        region, polygon = region_data
        try:
            # Validate that the polygon is valid and has an exterior contour.
            if not polygon.is_valid or not polygon.exterior:                                            # Check if the polygon is well-defined.
                raise ValueError(f"The polygon for region {region} is not valid.")                      # Raise an error if invalid.

            # Retrieve hole coordinates for the current region.
            holes = []                                                                                  # List to store hole coordinates.
            for contained_region in containment_results.get(region, []):                                # Check for contained regions (holes).
                # Validate that the hole exists and is valid.
                if contained_region not in polygons_by_region:                                          # Ensure the hole region exists.
                    print(f"Warning: Region {contained_region} is not in polygons_by_region.")
                    continue

                contained_polygon = polygons_by_region[contained_region]                                # Get the polygon for the contained region.

                if not contained_polygon.is_valid or not contained_polygon.exterior:                    # Ensure the hole polygon is valid.
                    print(f"Warning: Inner polygon {contained_region} is not valid.")
                    continue

                holes.append(np.array(contained_polygon.exterior.xy)[:, :-1])                           # Extract hole boundary coordinates.
            
            # Format the holes as a list of (x, y) pairs.
            hole_coordinates = [(hx, hy) for hx, hy in holes]                                           # Convert to a list of coordinate pairs.

            # Extract boundary coordinates, removing the last duplicate point.
            xb, yb = np.array(polygon.exterior.xy)[:, :-1]                                              # Extract main boundary coordinates.

            # Compute point spacing based on region dimensions.
            dist = distance(xb, yb, hole_coordinates)/num                                               # Compute initial spacing based on distance.
            while max(max(xb),max(yb))/dist < 21:                                                       # Ensure an adequate number of points.
                dist = dist/2                                                                           # Reduce spacing if necessary.

            # Generate the cloud of points for the region.
            return CreateCloud(xb, yb, hole_coordinates, dist, rand, region, mod)                       # Call the function to create the cloud.

        except Exception as e:
            print(f"Error generating cloud for region {region}: {e}")                                   # Print error message if cloud generation fails.
            return np.array([])

    # Process regions in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Map the processing function to all regions
        results = list(executor.map(process_region, polygons_by_region.items()))
        
        # If only one cloud is needed, take just the first result
        if gen == 0 and results:
            results = [results[0]]
    
    # Combine all results efficiently using numpy
    all_points = [r for r in results if r.size > 0]

    # Combine all generated clouds of points into a single array.
    if all_points:
        return np.vstack(all_points)                                                                    # Merge all generated clouds into one array.
    else:
        print("No cloud of points were generated for any region.")                                      # Print a message if no clouds were created.
        return np.array([])                                                                             # Return an empty array if no clouds were generated.

'''
======================================
4. VISUALIZATION AND EXPORTING
======================================
'''

def GraphCloud(all_clouds, folder, image_name, eps_name):
    """
    Plots and saves a graphical representation of the generated cloud of points.

    Parameters:
        all_clouds (np.array):                          Array containing the generated cloud of points data.
                                                        Each row is a point with the structure [x, y, region_flag, type_flag].
        folder (str):                                   Directory where the output images will be saved.
        image_name (str):                               Name of the PNG image file.
        eps_name (str):                                 Name of the EPS image file.

    Returns:
        None                                            Saves the generated plots as PNG and EPS files in the specified folder.
    """
    
    # Create a figure for visualization.
    plt.figure(figsize = (12, 8))                                                                       # Set figure size for better visualization.
    ax = plt.gca()                                                                                      # Get the current axis.

    # Identify unique region flags in the cloud of points.
    unique_flags = np.unique(all_clouds[:, 2].astype(int))                                              # Extract unique region identifiers.
    cmap = plt.colormaps.get_cmap('viridis')                                                            # Create a cmap with 20 different colors.
    norm = plt.Normalize(vmin = min(unique_flags), vmax = max(unique_flags))                            # Generate a colormap for different regions.

    color_map = {}
    for flag in unique_flags:
        if flag == 1:
            color_map[flag] = 'blue'
        else:
            rgba = cmap(norm(flag))
            color_map[flag] = to_rgb(rgba)

    # Scatter points by region.
    for flag in unique_flags:                                                                           # Loop through each unique region.
        cloud = all_clouds[all_clouds[:, 2] == flag]                                                    # Filter points belonging to the current region.
        color = color_map[flag]                                                                         # Get the color for the current region.
        plt.scatter(cloud[:, 0], cloud[:, 1], s=5, c=[color], alpha=1.0, label=f'Region {int(flag)}')   # Plot points for this region.

        # Plot boundary of the region.
        boundary = cloud[cloud[:, 3] == 1]                                                              # Extract boundary points (type_flag == 1).
        boundary = np.vstack([boundary, boundary[0]])                                                   # Close the boundary loop by repeating the first point.
        plt.plot(boundary[:, 0], boundary[:, 1], color=color, linestyle='solid')                        # Plot the boundary outline.

    # Configure plot properties.
    plt.grid(True)                                                                                      # Enable grid for better readability.
    plt.axis('equal')                                                                                   # Fix the axes for visualization.
    plt.title("Generated Cloud of Points")                                                              # Set plot title.

    # Add watermark
    try:
        # Load the logo image (assuming it's in the project's assets folder)
        logo = mpimg.imread('static/images/logo_b.png')                                                 # Load the logo image.
        
        # Create an OffsetImage with reduced opacity
        imagebox = OffsetImage(logo, zoom = 0.08, alpha = 0.3)                                          # Create an OffsetImage with reduced opacity.
        
        # Position the watermark in the bottom right corner
        ab = AnnotationBbox(imagebox, (0.95, 0.05),
                           xycoords = 'axes fraction',
                           box_alignment = (1, 0),
                           frameon = False)                                                             # Create an AnnotationBbox for the watermark.
        ax.add_artist(ab)
    except Exception as e:
        print(f"Warning: Could not add watermark: {e}")

    # Save the figure in PNG format.
    plt.savefig(os.path.join(folder, image_name), format = 'png')                                       # Save the plot as a PNG file.

    # Save the figure in EPS format.
    plt.savefig(os.path.join(folder, eps_name), format = 'eps')                                         # Save the plot as an EPS file.

    # Close the plot to free memory.
    plt.close()                                                                                         # Close the figure to avoid display issues.