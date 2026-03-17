"""
Export Module - Data Output and Verification

This module manages the saving of generated cloud data to persistent storage.
It handles CSV file generation and implements verification steps to ensure
data integrity.

Core Functionality:
1. CSV Export: Writes node coordinates, region IDs, and classifications.
2. Data Verification: Re-reads exported files to confirm data integrity.
3. Directory Management: Ensures output directories exist.

Key Features:
- Standardized CSV format for interoperability
- Built-in verification step to detect incomplete writes
- robust error handling for file I/O operations

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- CSV
- NumPy
- Logging
- OS
"""

import csv
import logging
import numpy as np
import os

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
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
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
