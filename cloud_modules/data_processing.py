"""
Data Processing Module - Input Handling and Validation

This module handles the loading, parsing, and validation of input data for the
Cloud Generation system. It focuses on reading CSV boundary files and structuring
the data for downstream processing.

Core Functionality:
1. CSV Loading: Robust reading of coordinate data from CSV files.
2. Data Validation: Checks for required columns and data integrity.
3. Region Parsing: Groups coordinates by region ID for multi-region support.
4. Error Handling: Graceful management of malformed or empty input files.

Key Features:
- Support for both single-region and multi-region CSV formats
- Automatic type conversion and cleaning of input data
- Detailed logging of data loading statistics

Author: Gerardo Tinoco-Guerrero
Date: March, 2026
Last Modification: March, 2026

Dependencies:
- CSV
- Logging
"""

import csv
import logging

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

def load_cloud_data(csv_file: str) -> tuple[list, list, list]:
    """
    Load cloud data from CSV file, including coordinates, regions, and classifications.
    
    Args:
        csv_file (str): Path to the CSV file.
        
    Returns:
        tuple: (points, regions, classifications)
            - points (list): List of (x, y) tuples.
            - regions (list): List of region IDs.
            - classifications (list): List of classification strings (optional).
    """
    points = []
    regions = []
    classifications = []
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            if not reader.fieldnames:
                logging.error(f"CSV file is empty or invalid: {csv_file}")
                return [], [], []
                
            fieldnames = [f.strip() for f in reader.fieldnames]
            
            has_region = 'region' in fieldnames
            has_classification = 'classification' in fieldnames
            
            for row in reader:
                try:
                    x = float(row['x'])
                    y = float(row['y'])
                    points.append((x, y))
                    
                    if has_region and row.get('region'):
                        try:
                            r_val = float(row['region'])
                            regions.append(int(r_val))
                        except (ValueError, TypeError):
                            regions.append(row['region'])
                    else:
                        regions.append(1) # Default to region 1
                        
                    if has_classification and row.get('classification'):
                        classifications.append(row['classification'])
                    else:
                        classifications.append('boundary') # Default assumption? Or handle as None
                        
                except (ValueError, KeyError):
                    continue
                    
        return points, regions, classifications
        
    except Exception as e:
        logging.error(f"Error loading cloud data from {csv_file}: {e}")
        return [], [], []
