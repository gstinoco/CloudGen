# --- Imports ---
# Standard library imports
from datetime import datetime
from threading import Timer
import logging
import os

# Third-party library imports
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import pandas as pd
import numpy as np
import dmsh
import csv

# Plot imports
import matplotlib
matplotlib.use('Agg')                                                                                   # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles


'''
======================================
2. DATA PROCESSING
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
        RuntimeError:                                   If any error occurs during file processing.
    """
    try:
        # Load the CSV file into a DataFrame.
        df = pd.read_csv(file_path)                                                                     # Read the CSV file.

        # Define the required columns.
        required_columns = {'x', 'y', 'flag'}                                                           # Columns needed in the dataset.

        # Check if all required columns are present.
        if not required_columns.issubset(df.columns):                                                   # Ensure that the file has the correct structure.
            raise ValueError("The CSV file must contain the columns: 'x', 'y', 'flag'")                 # Raise an error if columns are missing.
        
        # Select only the necessary columns.
        df = df[['x', 'y', 'flag']]                                                                     # Keep only the required columns.

        # Group points by region flag and store them as lists of coordinate pairs.
        points_by_region = (
            df.groupby('flag', group_keys = False)                                                      # Group data by the 'flag' column.
            .apply(lambda group: group[['x', 'y']].values.tolist(), include_groups = False)             # Convert each group into a list of (x, y) coordinates.
            .to_dict()                                                                                  # Convert the grouped data into a dictionary.
        )
        return points_by_region                                                                         # Return the dictionary of points grouped by region.

    except Exception as e:
        raise RuntimeError(f"Error processing the file: {e}")                                           # Raise an error if processing fails.

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
3. REGION ANALYSIS AND VALIDATION
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
        raise ValueError("Los polígonos proporcionados no son válidos.")                                # Raise an error if any is missing.
    
    # Check if polygon_2 is completely within polygon_1
    check = polygon_2.within(polygon_1) or (polygon_2.intersects(polygon_1) and polygon_1.area > polygon_2.area)

    return check                                                                                        # Return True if polygon_2 is inside polygon_1 or if they intersect and polygon_1 is larger.

def test_all_region_containments(polygons_by_region):
    """
    Tests containment relationships between multiple regions.

    Parameters:
        polygons_by_region (dict):                      Dictionary where keys are region identifiers and 
                                                        values are Shapely Polygon objects.

    Returns:
        dict:                                           Dictionary where keys are region identifiers and 
                                                        values are lists of other regions that they contain.
    """

    # Initialize a dictionary to store containment results.
    containment_results = {region: [] for region in polygons_by_region}                                 # Create an empty list for each region.

    # Iterate through all region pairs to check containment.
    for region_1, polygon_1 in polygons_by_region.items():                                              # Loop through each region's polygon.
        for region_2, polygon_2 in polygons_by_region.items():                                          # Compare with every other region.
            if region_1 != region_2 and test_region_containment(polygon_1, polygon_2):                  # Ensure regions are different and check containment.
                if region_2 not in containment_results[region_1]:                                       # Avoid duplicate entries.
                    containment_results[region_1].append(region_2)                                      # Add region_2 as contained within region_1.

    return containment_results                                                                          # Return the dictionary of containment relationships.

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
4. CLOUD OF POINTS GENERATION
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
    
    for region, polygon in polygons_by_region.items():                                                  # Iterate through all regions.
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
            cloud = CreateCloud(xb, yb, hole_coordinates, dist, rand, region, mod)                      # Call the function to create the cloud.
            
            # Store the generated cloud of points.
            all_points.append(cloud)                                                                    # Append to the list of generated clouds.

            # If only one cloud is needed, stop after the first generation.
            if gen == 0:
                break

        except Exception as e:
            print(f"Error generating cloud for region {region}: {e}")                                   # Print error message if cloud generation fails.
            continue

    # Combine all generated clouds of points into a single array.
    if all_points:
        return np.vstack(all_points)                                                                    # Merge all generated clouds into one array.
    else:
        print("No cloud of points were generated for any region.")                                      # Print a message if no clouds were created.
        return np.array([])                                                                             # Return an empty array if no clouds were generated.

'''
======================================
5. VISUALIZATION AND EXPORTING
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

    # Identify unique region flags in the cloud of points.
    unique_flags = np.unique(all_clouds[:, 2])                                                          # Extract unique region identifiers.
    colors = plt.cm.get_cmap('tab20', len(unique_flags))                                                # Generate a colormap for different regions.

    # Scatter points by region.
    for i, flag in enumerate(unique_flags):                                                             # Loop through each unique region.
        cloud = all_clouds[all_clouds[:, 2] == flag]                                                    # Filter points belonging to the current region.
        plt.scatter(cloud[:, 0], cloud[:, 1], s = 5, c = [colors(i)], label = f'Región {int(flag)}')    # Plot points for this region.

        # Plot boundary of the region.
        bound = cloud[cloud[:,3] == 1]                                                                  # Extract boundary points (type_flag == 1).
        bound = np.vstack([bound, bound[0]])                                                            # Close the boundary loop by repeating the first point.
        plt.plot(bound[:, 0], bound[:, 1], color = colors(i), linestyle = 'solid')                      # Plot the boundary outline.

    # Configure plot properties.
    plt.legend()                                                                                        # Add a legend to distinguish regions.
    plt.grid(True)                                                                                      # Enable grid for better readability.
    plt.title("Generated Cloud of Points")                                                              # Set plot title.

    # Save the figure in PNG format.
    plt.savefig(os.path.join(folder, image_name), format = 'png')                                       # Save the plot as a PNG file.

    # Save the figure in EPS format.
    plt.savefig(os.path.join(folder, eps_name), format = 'eps')                                         # Save the plot as an EPS file.

    # Close the plot to free memory.
    plt.close()                                                                                         # Close the figure to avoid display issues.