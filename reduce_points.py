"""
Point Reduction Module for WebApp

This module provides functionality to reduce points in CSV files by region for the
mGFD CloudGenerator web application. It implements uniform point reduction specifically
for region 1 while maintaining all other regions intact.

The module is optimized for web application use, featuring:
- Uniform point reduction method (approximately 50% reduction)
- Professional logging for error handling
- Silent operation suitable for web environments
- Robust error handling for various file and data issues

Author: Gerardo Tinoco-Guerrero
Date: May 2025
Last Modification: 23 August 2025
"""

import pandas as pd
import numpy as np
import logging

def reduce_points_by_region(input_csv, output_csv):
    """
    Reduce points in region 1 using uniform method while preserving other regions.
    
    This function processes a CSV file containing point data with regions, applying
    a uniform reduction method that removes approximately 50% of points from region 1
    only. All other regions remain completely unchanged.
    
    The uniform reduction method selects every other point (indices 0, 2, 4, ...)
    from region 1, providing a consistent and predictable reduction pattern.
    
    Args:
        input_csv (str): Absolute path to the input CSV file containing point data.
                         Expected columns: 'x', 'y', 'region'
        output_csv (str): Absolute path where the reduced CSV file will be saved.
                          Will be created/overwritten if it exists
    
    Returns:
        pandas.DataFrame: DataFrame with reduced points if successful
        None: If any error occurred during processing (file not found, invalid data, etc.)
    
    Raises:
        Logs errors for:
        - FileNotFoundError: When input file doesn't exist
        - EmptyDataError: When CSV file is empty or invalid
        - ValueError: When required columns are missing
        - Exception: For any other unexpected errors
    
    Example:
        >>> result = reduce_points_by_region('/path/to/input.csv', '/path/to/output.csv')
        >>> if result is not None:
        ...     print("Points reduced successfully")
    """
    try:
        # Load data
        df = pd.read_csv(input_csv)
        
        # Verify required columns
        required_columns = ['x', 'y', 'region']
        if not all(col in df.columns for col in required_columns):
            error_msg = f"CSV file missing required columns. Expected: {required_columns}, Found: {list(df.columns)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Process each region separately
        reduced_data = []
        
        for region_id in sorted(df['region'].unique()):
            region_data = df[df['region'] == region_id].copy()
            
            # Only apply reduction to region 1 (main region)
            if region_id == 1:
                # Uniform reduction taking every second point
                region_data = region_data.reset_index(drop=True)
                reduced_region = region_data.iloc[::2]  # Take every second point starting from index 0
            else:
                # For regions that are not region 1 (holes), keep all points
                reduced_region = region_data
            
            reduced_data.append(reduced_region)
        
        # Combine all reduced regions
        final_df = pd.concat(reduced_data, ignore_index=True)
        
        # Save result
        final_df.to_csv(output_csv, index=False)
        
        return final_df
        
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {input_csv}"
        logging.error(error_msg)
        return None
    except pd.errors.EmptyDataError as e:
        error_msg = f"Input file is empty or invalid: {input_csv}"
        logging.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error processing point reduction for {input_csv}: {str(e)}"
        logging.error(error_msg)
        return None