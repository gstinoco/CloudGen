# Standard library imports
import argparse                                                                                         # For parsing command line arguments
import os                                                                                               # For file and directory operations
from datetime import datetime                                                                           # For generating timestamps

# Third-party imports
import pandas as pd                                                                                     # For data manipulation and CSV handling
from shapely.geometry import Point, Polygon                                                             # For geometric operations
from shapely.ops import unary_union                                                                     # For combining multiple geometries

# Local application imports
from core import (                                                                                      # Import core functionality
    process_csv,                                                                                        # For processing input CSV files
    generate_polygons,                                                                                  # For creating polygon objects
    test_all_region_containments,                                                                       # For analyzing region relationships
    remove_duplicate_containments,                                                                      # For cleaning containment data
    generate_clouds_for_all_regions,                                                                    # For generating point clouds
    GraphCloud                                                                                          # For visualization
)

# Define and create output directory for generated files
OUTPUT_FOLDER = os.path.join('tmp', 'results')                                                          # Set path for output files
os.makedirs(OUTPUT_FOLDER, exist_ok=True)                                                               # Create directory if it doesn't exist

def run_batch_mode(input_file, num, rand, mod, gen):
    """Process input CSV file and generate point clouds in batch mode.

    This function handles the complete workflow for generating point clouds from a CSV file:
    1. Validates input file existence
    2. Processes CSV data to extract regions
    3. Generates polygons and analyzes their containment relationships
    4. Creates point clouds based on the specified parameters
    5. Saves results as PNG, EPS, and CSV files

    Args:
        input_file (str): Path to the input CSV file containing region points.
        num (int): Point density (higher = more points).
        rand (int): Random perturbation flag (0 = no, 1 = yes).
        mod (int): Method selection (0 = dmsh, 1 = grid).
        gen (int): Generation type (0 = exterior only, 1 = all regions).
    """
    if not os.path.exists(input_file):                                                                  # Check if input file exists
        print(f"File not found: {input_file}")                                                          # Print error message if file not found
        return                                                                                          # Exit function

    try:
        # Process input data and generate polygons
        regions = process_csv(input_file)                                                               # Read and process CSV data
        polygons = generate_polygons(regions)                                                           # Create polygon objects
        containment = test_all_region_containments(polygons)                                            # Analyze region relationships
        depurated = remove_duplicate_containments(containment)                                          # Clean containment data

        # Generate point clouds
        clouds = generate_clouds_for_all_regions(polygons, depurated, num, rand, mod, gen)              # Generate point cloud

        if clouds.size == 0:                                                                            # Check if points were generated
            print("No points were generated.")                                                          # Print message if no points
            return                                                                                      # Exit function

        # Generate output filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                                            # Create timestamp for unique filenames
        image_name = f'plot_{timestamp}.png'                                                            # PNG filename
        eps_name = f'plot_{timestamp}.eps'                                                              # EPS filename
        csv_name = f'cloud_{timestamp}.csv'                                                             # CSV filename

        # Create visualizations and save data
        GraphCloud(clouds, OUTPUT_FOLDER, image_name, eps_name)                                         # Generate and save plots

        pd.DataFrame(clouds, columns=["x", "y", "region", "boundary_flag"]).to_csv(                     # Create DataFrame
            os.path.join(OUTPUT_FOLDER, csv_name), index=False                                          # Save to CSV file
        )

        print("Cloud generated successfully:")                                                          # Print success message
        print(f"- CSV:  {csv_name}")                                                                    # Print CSV filename
        print(f"- PNG:  {image_name}")                                                                  # Print PNG filename
        print(f"- EPS:  {eps_name}")                                                                    # Print EPS filename

    except Exception as e:                                                                              # Handle any errors
        print(f"Error during execution: {e}")                                                           # Print error message


if __name__ == '__main__':
    # Initialize argument parser with description
    parser = argparse.ArgumentParser(description="Batch generator for point clouds from CSV files.")    # Create argument parser
    
    # Define command line arguments with their respective types and default values
    parser.add_argument('--input',  type = str, required = True, help = "Input CSV file containing region points.")
                                                                                                        # Input file path
    parser.add_argument('--num',    type = int, default  = 3,    help = "Point density (higher = more points).")
                                                                                                        # Point density
    parser.add_argument('--rand',   type = int, default  = 1,    help = "Random perturbation (0 = no, 1 = yes).")
                                                                                                        # Random flag
    parser.add_argument('--mod',    type = int, default  = 1,    help = "Method: 0 = dmsh, 1 = grid.")
                                                                                                        # Method selection
    parser.add_argument('--gen',    type = int, default  = 0,    help = "0 = exterior only, 1 = all regions.")
                                                                                                        # Generation type

    # Parse command line arguments and execute the batch processing
    args = parser.parse_args()                                                                          # Parse command line arguments
    run_batch_mode(args.input, args.num, args.rand, args.mod, args.gen)                                 # Run batch processing