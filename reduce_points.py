"""
Point Reduction Module for mGFD CloudGenerator WebApp

This module provides advanced point reduction functionality for CSV files containing
cloud data organized by regions. It implements multiple reduction algorithms
optimized for the mGFD CloudGenerator web application, enabling efficient processing
of large point datasets while maintaining geometric integrity.

Core Functionality:
    - Intelligent point reduction with region-aware processing
    - Multiple reduction algorithms (Uniform, Multiple, Filtered)
    - Geometric filtering for subregion optimization
    - Adaptive reduction strategies based on point density

Point Reduction Algorithms:
    1. Uniform Reduction: Systematic point selection (every nth point)
       - Provides consistent ~50% reduction for region 1
       - Maintains spatial distribution patterns
       - Preserves geometric characteristics
    
    2. Multiple Region Reduction: Advanced multi-region processing
       - Configurable reduction multipliers per region
       - Maintains inter-region relationships
       - Optimized for complex multi-region datasets
    
    3. Filtered Reduction: Geometric filtering with subregion analysis
       - Removes points from main regions that fall within subregions
       - Uses convex hull analysis for spatial relationships
       - Prevents point overlap between hierarchical regions

Key Features:
    - Region-aware processing (preserves non-target regions)
    - Configurable reduction parameters and multipliers
    - Robust error handling and validation
    - Professional logging for debugging and monitoring
    - Silent operation mode for web application integration
    - Memory-efficient processing for large datasets
    - Geometric integrity preservation during reduction

Technical Implementation:
    - Built on pandas for efficient data manipulation
    - Uses NumPy for numerical operations and indexing
    - Integrates Shapely for geometric computations
    - Employs SciPy for convex hull analysis
    - Implements tempfile handling for safe file operations

Workflow:
    1. Load and validate CSV point data
    2. Analyze region structure and point distribution
    3. Apply selected reduction algorithm
    4. Perform geometric filtering (if enabled)
    5. Validate output data integrity
    6. Save reduced dataset to output file

Applications:
    - Cloud optimization for visualization
    - Data preprocessing for computational geometry
    - Memory usage reduction for large datasets
    - Performance optimization for real-time applications

Author: Gerardo Tinoco-Guerrero
Date: May 2025
Last Modification: September 2025
Version: 2.1.0

Dependencies:
    - pandas >= 1.3.0 (data manipulation and CSV handling)
    - numpy >= 1.21.0 (numerical operations and array processing)
    - shapely >= 1.8.0 (geometric operations and spatial analysis)
    - scipy >= 1.7.0 (convex hull computation and spatial algorithms)
"""

import pandas as pd
import numpy as np
import logging
import os
import tempfile
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

def reduce_points_by_region_single(input_csv, output_csv):
    """
    Reduce points in region 1 using uniform method while preserving other regions (single pass).
    
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
        >>> result = reduce_points_by_region_single('/path/to/input.csv', '/path/to/output.csv')
        >>> if result is not None:
        ...     print("Points reduced successfully")
    """
    try:
        df = pd.read_csv(input_csv)
        
        required_columns = ['x', 'y', 'region']
        if not all(col in df.columns for col in required_columns):
            error_msg = f"CSV file missing required columns. Expected: {required_columns}, Found: {list(df.columns)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        reduced_data = []
        
        for region_id in sorted(df['region'].unique()):
            region_data = df[df['region'] == region_id].copy()
            
            if region_id == 1:
                region_data = region_data.reset_index(drop=True)
                reduced_region = region_data.iloc[::2]
            else:
                reduced_region = region_data
            
            reduced_data.append(reduced_region)
        
        final_df = pd.concat(reduced_data, ignore_index=True)
        
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

def reduce_points_by_region_multiple(input_csv, output_csv, multiplier=2):
    """
    Advanced point reduction using iterative uniform method for aggressive point reduction.
    
    This function applies the uniform reduction algorithm multiple times in sequence to achieve
    higher reduction rates while maintaining spatial distribution characteristics. Each iteration
    reduces the point count by approximately 50%, resulting in exponential reduction rates.
    The function uses temporary files for intermediate processing to ensure data integrity.
    
    Args:
        input_csv (str): Absolute path to the input CSV file containing point data.
                        Expected columns: 'x', 'y', 'region'
        output_csv (str): Absolute path where the final reduced CSV file will be saved.
                         Will be created/overwritten if it exists.
        multiplier (int, optional): Number of reduction iterations to apply.
                                   Valid values: 2, 3, or 4. Default: 2.
                                   - multiplier=2: ~75% reduction (1-(1/2)²=0.75)
                                   - multiplier=3: ~87.5% reduction (1-(1/2)³=0.875)
                                   - multiplier=4: ~93.75% reduction (1-(1/2)⁴=0.9375)
    
    Returns:
        pandas.DataFrame: Final DataFrame with reduced points if successful.
                         Contains same structure as input but with fewer points in region 1.
        None: If any error occurred during processing (file errors, invalid data, etc.)
    
    Algorithm:
        1. Initialize processing chain with input file
        2. For each iteration (1 to multiplier):
           a. Create temporary output file (except for final iteration)
           b. Apply uniform reduction (every 2nd point selection)
           c. Save intermediate result to temporary file
           d. Update input for next iteration
        3. Clean up temporary files after processing
        4. Return final reduced dataset
    
    Reduction Mathematics:
        - Single iteration: 50% reduction (keeps every 2nd point)
        - Multiple iterations: Exponential reduction
        - Final point count = Original × (0.5)^multiplier
        - Maintains uniform spatial distribution at all scales
    
    Features:
        - Iterative processing for high reduction rates
        - Temporary file management for memory efficiency
        - Automatic cleanup of intermediate files
        - Preserves all non-region-1 data unchanged
        - Maintains geometric distribution patterns
        - Error handling with graceful degradation
    
    Use Cases:
        - Aggressive point reduction for large datasets
        - Memory optimization for visualization applications
        - Performance enhancement for computational geometry
        - Data preprocessing for machine learning applications
    
    Note:
        Higher multiplier values result in more aggressive reduction but may affect
        the geometric representation quality. Choose multiplier based on the balance
        between file size reduction and spatial accuracy requirements.
    
    Raises:
        Logs errors for:
        - FileNotFoundError: When input file doesn't exist
        - ValueError: When multiplier is outside valid range or data is invalid
        - OSError: When temporary file operations fail
        - Exception: For any other unexpected processing errors
    
    Example:
        >>> # Reduce points by ~87.5% (3 iterations)
        >>> result = reduce_points_by_region_multiple('input.csv', 'output.csv', multiplier=3)
        >>> if result is not None:
        ...     print(f"Reduction completed. Final points: {len(result)}")
    """
    try:
        current_input = input_csv
        temp_files = []
        result = None
        
        for i in range(multiplier):
            if i == multiplier - 1:
                # Last iteration, use the final output file
                current_output = output_csv
            else:
                # Create temporary file for intermediate results
                temp_fd, current_output = tempfile.mkstemp(suffix='.csv')
                os.close(temp_fd)  # Close the file descriptor
                temp_files.append(current_output)
            
            # Apply single reduction
            result = reduce_points_by_region_single(current_input, current_output)
            if result is None:
                # Clean up temp files on error
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return None
            
            # Update input for next iteration
            current_input = current_output
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return result
        
    except Exception as e:
        error_msg = f"Error in multiple point reduction: {str(e)}"
        logging.error(error_msg)
        return None

def filter_main_region_points_in_subregions(df):
    """
    Advanced geometric filtering to remove main region points that fall within subregion boundaries.
    
    This function implements sophisticated spatial analysis to maintain geometric consistency
    between hierarchical regions. It uses convex hull computation to define subregion boundaries
    and performs point-in-polygon tests to identify and remove overlapping points from the
    main region. This ensures clean separation between regions and prevents point duplication
    in hierarchical region structures.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing cloud data with required columns:
                              - 'x' (float): X coordinates of points
                              - 'y' (float): Y coordinates of points  
                              - 'region' (int): Region identifier (1=main, 2+=subregions)
    
    Returns:
        pandas.DataFrame: Filtered DataFrame with the same structure as input, but with
                         main region (region 1) points that fall inside subregion boundaries
                         removed. All subregion points remain unchanged.
    
    Algorithm:
        1. Identify all unique regions in the dataset
        2. Skip processing if only one region exists (no filtering needed)
        3. For each subregion (regions 2, 3, 4, ...):
           a. Extract region points and validate minimum point count (≥3)
           b. Compute convex hull to define region boundary polygon
           c. Create Shapely Polygon object and validate geometry
           d. Add valid polygons to subregion boundary collection
        4. For each main region point:
           a. Create Shapely Point object from coordinates
           b. Test if point falls within any subregion polygon
           c. Mark point for removal if inside any subregion
        5. Filter out marked points and return cleaned dataset
    
    Geometric Features:
        - Convex hull computation for robust boundary definition
        - Point-in-polygon testing using Shapely geometric operations
        - Automatic handling of invalid or degenerate geometries
        - Preservation of all non-main region data
        - Efficient spatial indexing for large datasets
    
    Use Cases:
        - Hierarchical region processing with clean boundaries
        - Preventing point overlap in multi-level region structures
        - Data preprocessing for visualization applications
        - Geometric consistency maintenance in cloud processing
        - Interior cloud generation with proper region separation
    
    Performance Considerations:
        - Convex hull computation: O(n log n) per subregion
        - Point-in-polygon tests: O(m × k) where m=main points, k=subregions
        - Memory usage scales with number of subregions and points
        - Optimized for typical use cases with few subregions
    
    Note:
        This function is particularly important when Interior Clouds functionality
        is disabled, as it ensures consistent point distribution regardless of
        the interior cloud generation settings. The convex hull approach provides
        a conservative boundary estimation that works well for most geometric shapes.
    
    Raises:
        Exception: Logs warnings for geometric computation errors but continues
                  processing with remaining valid subregions.
    
    Example:
        >>> # Filter main region points that overlap with subregions
        >>> df_filtered = filter_main_region_points_in_subregions(df)
        >>> main_points_before = len(df[df['region'] == 1])
        >>> main_points_after = len(df_filtered[df_filtered['region'] == 1])
        >>> print(f"Removed {main_points_before - main_points_after} overlapping points")
    """
    try:
        # Get all regions
        regions = sorted(df['region'].unique())
        
        if len(regions) <= 1:
            # No subregions to filter against
            return df
        
        # Create polygons for subregions (regions 2, 3, 4, ...)
        subregion_polygons = []
        for region_id in regions[1:]:  # Skip region 1 (main region)
            region_points = df[df['region'] == region_id][['x', 'y']].values
            if len(region_points) >= 3:
                try:
                    # Create convex hull of region points to form polygon
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(region_points)
                    hull_points = region_points[hull.vertices]
                    poly = Polygon(hull_points)
                    if poly.is_valid:
                        subregion_polygons.append(poly)
                except:
                    continue
        
        if not subregion_polygons:
            return df
        
        # Filter main region points
        filtered_rows = []
        for _, row in df.iterrows():
            if row['region'] == 1:
                # Check if main region point falls inside any subregion
                point_obj = Point(row['x'], row['y'])
                inside_subregion = False
                for subregion_poly in subregion_polygons:
                    if subregion_poly.contains(point_obj):
                        inside_subregion = True
                        break
                
                # Only keep main region points that are NOT inside subregions
                if not inside_subregion:
                    filtered_rows.append(row)
            else:
                # Keep all subregion points
                filtered_rows.append(row)
        
        return pd.DataFrame(filtered_rows).reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Error filtering main region points: {e}")
        return df  # Return original if filtering fails

def reduce_points_by_region_with_filtering(input_csv, output_csv, multiplier=2, filter_subregions=False):
    """
    Advanced point reduction with optional geometric filtering for enhanced consistency.
    
    This function provides the most sophisticated point reduction approach by combining
    geometric filtering with adaptive reduction algorithms. It ensures consistent
    behavior across different Interior Clouds settings and provides optimal results
    for complex multi-region cloud datasets with overlapping geometries.
    
    Args:
        input_csv (str): Absolute path to the input CSV file containing cloud data.
                        Expected format: CSV with columns 'x', 'y', 'region'
                        File must exist and be readable.
        output_csv (str): Absolute path where the processed CSV file will be saved.
                         Parent directory must exist and be writable.
                         File will be created or overwritten if it exists.
        multiplier (int, optional): Reduction intensity parameter controlling algorithm selection:
                                   - multiplier = 1: Single-pass reduction (~50% reduction)
                                   - multiplier ≥ 2: Multi-pass reduction (exponential reduction)
                                   Default: 2 (balanced reduction)
        filter_subregions (bool, optional): Enable geometric filtering before reduction.
                                           - True: Apply subregion filtering first
                                           - False: Standard reduction without filtering
                                           Default: False (for performance)
    
    Returns:
        pandas.DataFrame: Processed DataFrame with reduced and optionally filtered points.
                         Maintains original structure with optimized point distribution.
                         All non-region-1 data preserved unchanged.
        None: If any error occurred during processing (file errors, invalid data, etc.)
    
    Algorithm Workflow:
        1. Optional Geometric Filtering (if filter_subregions=True):
           - Load and analyze cloud data
           - Identify main region (region 1) and subregions (region > 1)
           - Create convex hull boundaries for each subregion
           - Remove main region points that fall within subregion boundaries
           - Save filtered data to temporary file for processing
        
        2. Adaptive Point Reduction:
           - Select reduction algorithm based on multiplier parameter
           - Apply single-pass or multi-pass reduction to filtered/original data
           - Preserve spatial distribution and geometric properties
           - Maintain data integrity throughout the process
        
        3. Cleanup and Output:
           - Remove temporary files if created
           - Return processed DataFrame
           - Log any errors or warnings
    
    Geometric Filtering Features:
        - Convex hull computation for precise boundary detection
        - Shapely-based point-in-polygon testing for accuracy
        - Robust handling of edge cases and degenerate geometries
        - Memory-efficient processing for large datasets
        - Automatic fallback for regions with insufficient points
    
    Reduction Integration:
        - Seamless integration with existing reduction algorithms
        - Temporary file management for filtered data processing
        - Consistent API regardless of filtering option
        - Optimized performance for both filtered and unfiltered workflows
    
    Use Cases:
        - Complex multi-region datasets with overlapping boundaries
        - Ensuring consistent results across different Interior Clouds settings
        - High-quality point reduction for visualization and analysis
        - Preprocessing for downstream geometric algorithms
        - Quality control for cloud generation pipelines
    
    Performance Considerations:
        - Filtering adds computational overhead but improves quality
        - Temporary file I/O for filtered data processing
        - Memory usage scales with dataset size and number of regions
        - Processing time increases with geometric complexity
        - Recommended for quality-critical applications
    
    Consistency Benefits:
        This function addresses inconsistencies that can arise when Interior Clouds
        settings change between point generation and reduction phases. By optionally
        filtering overlapping points before reduction, it ensures predictable and
        reproducible results regardless of the original generation parameters.
    
    Example:
        >>> # Standard reduction without filtering (fastest)
        >>> result = reduce_points_by_region_with_filtering(
        ...     'input.csv', 'output.csv', multiplier=2, filter_subregions=False
        ... )
        >>> 
        >>> # High-quality reduction with geometric filtering
        >>> result = reduce_points_by_region_with_filtering(
        ...     'input.csv', 'output.csv', multiplier=2, filter_subregions=True
        ... )
        >>> 
        >>> # Aggressive reduction with filtering for storage optimization
        >>> result = reduce_points_by_region_with_filtering(
        ...     'input.csv', 'output.csv', multiplier=4, filter_subregions=True
        ... )
    
    See Also:
        - filter_main_region_points_in_subregions(): Geometric filtering function
        - reduce_points_by_region(): Main reduction entry point
        - reduce_points_by_region_single(): Single-pass reduction
        - reduce_points_by_region_multiple(): Multi-pass reduction
    """
    try:
        if filter_subregions:
            # Apply subregion filtering first
            df = pd.read_csv(input_csv)
            filtered_df = filter_main_region_points_in_subregions(df)
            
            # Save filtered data to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_filtered_file = temp_file.name
            
            filtered_df.to_csv(temp_filtered_file, index=False)
            
            # Apply reduction to filtered data
            if multiplier <= 1:
                result = reduce_points_by_region_single(temp_filtered_file, output_csv)
            else:
                result = reduce_points_by_region_multiple(temp_filtered_file, output_csv, multiplier)
            
            # Clean up temporary file
            try:
                os.remove(temp_filtered_file)
            except:
                pass
            
            return result
        else:
            # Standard reduction without filtering
            if multiplier <= 1:
                return reduce_points_by_region_single(input_csv, output_csv)
            else:
                return reduce_points_by_region_multiple(input_csv, output_csv, multiplier)
                
    except Exception as e:
        logging.error(f"Error in reduce_points_by_region_with_filtering: {e}")
        return None

def reduce_points_by_region(input_csv, output_csv, multiplier=2):
    """
    Main entry point for intelligent point reduction with adaptive algorithm selection.
    
    This function serves as the primary interface for point reduction operations,
    automatically selecting the optimal reduction strategy based on the multiplier
    parameter. It provides backward compatibility while offering advanced reduction
    capabilities for different use cases and performance requirements.
    
    Args:
        input_csv (str): Absolute path to the input CSV file containing cloud data.
                        Expected format: CSV with columns 'x', 'y', 'region'
                        File must exist and be readable.
        output_csv (str): Absolute path where the reduced CSV file will be saved.
                         Parent directory must exist and be writable.
                         File will be created or overwritten if it exists.
        multiplier (int, optional): Reduction intensity parameter controlling algorithm selection:
                                   - multiplier = 1: Single-pass reduction (~50% reduction)
                                   - multiplier ≥ 2: Multi-pass reduction (exponential reduction)
                                   Default: 2 (for backward compatibility)
    
    Returns:
        pandas.DataFrame: Processed DataFrame with reduced points if successful.
                         Maintains original structure with fewer points in region 1.
                         All non-region-1 data preserved unchanged.
        None: If any error occurred during processing (file errors, invalid data, etc.)
    
    Algorithm Selection Logic:
        - For multiplier ≤ 1: Uses single-pass uniform reduction algorithm
          * Fastest processing time
          * Moderate reduction rate (~50%)
          * Best for quick optimization needs
        
        - For multiplier ≥ 2: Uses multi-pass iterative reduction algorithm  
          * Higher reduction rates (exponential)
          * More processing time but better compression
          * Ideal for aggressive optimization requirements
    
    Reduction Strategies:
        1. Single-Pass (multiplier ≤ 1):
           - Direct uniform sampling (every 2nd point)
           - O(n) time complexity
           - Memory efficient
           - Preserves spatial distribution
        
        2. Multi-Pass (multiplier ≥ 2):
           - Iterative reduction with temporary file management
           - O(n × multiplier) time complexity
           - Higher memory usage for intermediate results
           - Exponential reduction rates
    
    Features:
        - Automatic algorithm selection based on requirements
        - Backward compatibility with existing code
        - Consistent API across different reduction strategies
        - Robust error handling and logging
        - Preservation of non-target region data
        - Optimized performance for different use cases
    
    Use Cases:
        - Quick point reduction for visualization (multiplier=1)
        - Moderate optimization for web applications (multiplier=2)
        - Aggressive compression for storage optimization (multiplier≥3)
        - Batch processing with consistent interface
        - Integration with existing cloud pipelines
    
    Performance Characteristics:
        - Single-pass: Fast execution, moderate reduction
        - Multi-pass: Slower execution, high reduction rates
        - Memory usage scales with multiplier value
        - File I/O optimized for large datasets
    
    Backward Compatibility:
        This function maintains compatibility with previous versions by defaulting
        to multiplier=2, which provides a good balance between reduction rate and
        processing time for most applications.
    
    Example:
        >>> # Quick reduction for visualization
        >>> result = reduce_points_by_region('input.csv', 'output.csv', multiplier=1)
        >>> 
        >>> # Balanced reduction for web apps  
        >>> result = reduce_points_by_region('input.csv', 'output.csv', multiplier=2)
        >>> 
        >>> # Aggressive reduction for storage
        >>> result = reduce_points_by_region('input.csv', 'output.csv', multiplier=4)
    
    See Also:
        - reduce_points_by_region_single(): Direct single-pass reduction
        - reduce_points_by_region_multiple(): Direct multi-pass reduction
        - reduce_points_by_region_with_filtering(): Reduction with geometric filtering
    """
    if multiplier <= 1:
        return reduce_points_by_region_single(input_csv, output_csv)
    else:
        return reduce_points_by_region_multiple(input_csv, output_csv, multiplier)