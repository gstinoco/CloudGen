from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app, send_file, url_for
import os
import cv2
import numpy as np
import csv
import time
import tempfile
import threading
from werkzeug.utils import secure_filename
from datetime import datetime

from cloud_modules.generator import generate_cloud_regular, generate_cloud_natural
from cloud_modules.reduction import reduce_points_by_region
from cloud_modules.data_processing import load_cloud_data

from . import cloud_bp


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS
    
@cloud_bp.route('/cloud_generator')
def cloud_generator():
    """
    Render the cloud generator page for cloud generation from CSV data.
    
    Returns:
        str: Rendered HTML template for the cloud generator interface
    """
    return render_template('cloud_generator.html')


@cloud_bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    """
    Handle CSV file uploads for coordinate data.
    
    This function processes uploaded CSV files containing coordinate data,
    validates the file format and required columns (x, y, and optionally region),
    and provides information about the uploaded data including region analysis.
    
    Expected file format:
        CSV file with required columns: 'x', 'y'
        Optional column: 'region' (for multi-region data)
        All coordinate values must be numeric
    
    Returns:
        JSON response with success status, filename, data summary including
        total points, regions count, region list, and data preview,
        or error message if validation fails.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_csv_file(file.filename):
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and validate CSV
            try:
                points = []
                with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    
                    if not reader.fieldnames:
                        raise ValueError("Empty CSV file")
                        
                    fieldnames = [f.strip() for f in reader.fieldnames]
                    
                    # Validate required columns
                    required_columns = ['x', 'y']
                    if not all(col in fieldnames for col in required_columns):
                        os.remove(filepath)
                        return jsonify({
                            'success': False,
                            'error': f'CSV file must contain columns: {", ".join(required_columns)}'
                        })
                    
                    has_region = 'region' in fieldnames
                    unique_regions = set()
                    
                    for row in reader:
                        # Validate numeric data
                        try:
                            x_val = row['x'].strip() if row.get('x') else ''
                            y_val = row['y'].strip() if row.get('y') else ''
                            
                            if not x_val or not y_val:
                                continue
                                
                            x = float(x_val)
                            y = float(y_val)
                        except ValueError:
                            os.remove(filepath)
                            return jsonify({
                                'success': False,
                                'error': 'Columns x and y must contain numeric values'
                            })
                            
                        point = {'x': x, 'y': y}
                        
                        if has_region and row.get('region'):
                            try:
                                r_val = row['region'].strip()
                                if r_val:
                                    # Try to handle "1.0" as 1
                                    try:
                                        r_num = float(r_val)
                                        r_int = int(r_num)
                                        unique_regions.add(r_int)
                                        point['region'] = r_int
                                    except ValueError:
                                        # Keep as string if not numeric
                                        unique_regions.add(r_val)
                                        point['region'] = r_val
                            except Exception:
                                pass
                        
                        points.append(point)
                
                # Analyze regions
                regions_info = []
                total_regions = 1
                
                if has_region and unique_regions:
                    total_regions = len(unique_regions)
                    # Sort regions
                    try:
                        sorted_regions = sorted(list(unique_regions))
                    except TypeError:
                         sorted_regions = sorted(list(unique_regions), key=str)
                         
                    regions_info = [f"Region {r}" for r in sorted_regions]
                else:
                    regions_info = ["Region 1"]
                
                # Generate URL for the file
                file_url = url_for('contour.uploaded_file', filename=filename)
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'url': file_url,
                    'total_points': len(points),
                    'regions': total_regions,
                    'region_list': regions_info,
                    'rows': len(points),
                    'preview': points[:5]
                })
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': f'Error processing CSV file: {str(e)}'
                })
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'})
    
    except Exception as e:
        current_app.logger.error(f"Error in upload_csv: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})


@cloud_bp.route('/generate_cloud', methods=['POST'])
def generate_cloud_api():
    """
    API endpoint to generate a point cloud from uploaded CSV data.
    
    This endpoint processes CSV files containing coordinate data and generates
    optimized point clouds using Regular Distribution algorithms.
    
    Request Format:
        POST /generate_cloud
        Content-Type: application/json
        
        {
            "csv_filename": "data.csv",
            "regiones_inside": true,
            "reduce_points": false
        }
    
    Parameters:
        csv_filename (str): Name of the uploaded CSV file to process
        regiones_inside (bool, optional): Whether to include interior regions
                                        in the cloud generation. Default: false
        reduce_points (bool, optional): Whether to apply point reduction algorithms
                                      to optimize the cloud density. Default: false
    
    Returns:
        JSON Response:
        {
            "success": true,
            "message": "Point cloud generated successfully",
            "files": ["file1.csv", "file2.csv", ...]
        }
        
        OR (on error):
        {
            "error": "Error message describing the issue"
        }
    
    HTTP Status Codes:
        200: Success - Cloud generation completed successfully
        400: Bad Request - Missing required parameters or invalid input
        404: Not Found - Specified CSV file does not exist
        500: Internal Server Error - Processing error during generation
    
    Technical Implementation:
        - Direct execution using Regular Distribution
        - Synchronous processing for immediate results
        - Optional point reduction post-processing
        - Comprehensive error handling and logging
    
    Mathematical Methods:
        - Regular Distribution for optimal point distribution
        - Adaptive cloud generation based on geometric complexity
        - Region-based point classification and optimization
        - Advanced mesh generation and node distribution algorithms
    
    Example Usage:
        curl -X POST http://localhost:8080/generate_cloud \
             -H "Content-Type: application/json" \
             -d '{
                 "csv_filename": "coordinates.csv",
                 "regiones_inside": true,
                 "reduce_points": false
             }'
    
    Example Response:
        {
            "success": true,
            "message": "Point cloud generated successfully",
            "files": ["coordinates_nodes_20250823_143022.csv", "coordinates_elements_20250823_143022.csv"]
        }
    """
    data = request.get_json()
    csv_filename = data.get('csv_filename')
    regiones_inside = data.get('regiones_inside', False)
    reduce_points_flag = data.get('reduce_points', False)
    reduce_points_multiplier = data.get('reduce_points_multiplier', 2)
    density_multiplier = float(data.get('density_multiplier', 1.0))
    
    if not csv_filename:
        return jsonify({'error': 'No CSV file specified'}), 400
    
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], csv_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(csv_filename)[0]
        
        input_file = filepath
        
        # Generate point cloud using Regular Distribution
        current_app.logger.info(f"Starting cloud generation for {csv_filename}")
        
        # Construct output file path
        output_filename = f"{base_name}_cloud_{timestamp}.csv"
        output_file = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        result = generate_cloud_regular(
            csv_file=input_file,
            output_file=output_file,
            inside_regions=regiones_inside,
            cloud_size=None,
            density_multiplier=density_multiplier
        )
        
        success = result.get('success', False) if result else False
        
        if success:
            # Get generated files from result
            generated_files = []
            
            # Add the main CSV file
            if result and result.get('output_file'):
                csv_file = os.path.basename(result['output_file'])
                if os.path.exists(result['output_file']):
                    generated_files.append(csv_file)
            
            # Add visualization files if they exist (PNG and SVG)
            if result and result.get('visualization_file'):
                png_file = os.path.basename(result['visualization_file'])
                if os.path.exists(result['visualization_file']):
                    generated_files.append(png_file)
            
            if result and result.get('visualization_svg_file'):
                svg_file = os.path.basename(result['visualization_svg_file'])
                if os.path.exists(result['visualization_svg_file']):
                    generated_files.append(svg_file)
            
            current_app.logger.info(f"Cloud generation completed successfully. Generated files: {generated_files}")
            current_app.logger.info(f"Total nodes: {result.get('total_nodes', 'unknown')}, Regions: {result.get('regions_generated', 'unknown')}")
            
            return jsonify({
                'success': True,
                'message': 'Cloud generated successfully',
                'files': generated_files
            })
        else:
            current_app.logger.error("Cloud generation failed")
            return jsonify({'error': 'Error in cloud generation'}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error in cloud generation: {str(e)}")
        return jsonify({'error': f'Error generating the cloud: {str(e)}'}), 500



@cloud_bp.route('/generate_cloud_natural', methods=['POST'])
def generate_cloud_natural_api():
    """
    API endpoint to generate a point cloud using Natural Distribution for more natural distribution.
    
    This endpoint provides an alternative to the standard grid-based approach by using
    Natural Distribution, which creates more natural and less regular point distributions.
    This method is particularly useful for applications requiring organic-looking patterns
    or when avoiding the artificial regularity of grid-based methods.
    
    Request Format:
        POST /generate_cloud_natural
        Content-Type: application/json
        
        {
            "csv_filename": "data.csv",
            "regiones_inside": true,
            "reduce_points": false
        }
    
    Parameters:
        csv_filename (str): Name of the uploaded CSV file to process
        regiones_inside (bool, optional): Whether to include interior regions
                                        in the cloud generation. Default: false
        reduce_points (bool, optional): Whether to apply point reduction algorithms
                                      to optimize the cloud density. Default: false
    
    Returns:
        JSON Response:
        {
            "success": true,
            "message": "Natural Distribution point cloud generated successfully",
            "files": ["file1.csv", "file2.csv", ...]
        }
        
        OR (on error):
        {
            "error": "Error message describing the issue"
        }
    
    HTTP Status Codes:
        200: Success - Natural Distribution cloud generation completed successfully
        400: Bad Request - Missing required parameters or invalid input
        404: Not Found - Specified CSV file does not exist
        500: Internal Server Error - Processing error during generation
    
    Technical Implementation:
        - Natural Distribution for natural point distribution
        - Uniform density across all regions
        - Fallback to standard method if Natural Distribution fails
        - Comprehensive error handling and logging
    
    Example Usage:
        curl -X POST http://localhost:8080/generate_cloud_natural \
             -H "Content-Type: application/json" \
             -d '{
                 "csv_filename": "coordinates.csv",
                 "regiones_inside": true,
                 "reduce_points": false
             }'
    """
    data = request.get_json()
    csv_filename = data.get('csv_filename')
    regiones_inside = data.get('regiones_inside', False)
    reduce_points_flag = data.get('reduce_points', False)
    reduce_points_multiplier = data.get('reduce_points_multiplier', 2)
    density_multiplier = float(data.get('density_multiplier', 1.0))
    
    if not csv_filename:
        return jsonify({'error': 'No CSV file specified'}), 400
    
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], csv_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(csv_filename)[0]
        
        input_file = filepath
        
        # Generate point cloud using Natural Distribution
        current_app.logger.info(f"Starting Natural Distribution cloud generation for {csv_filename}")
        
        # Construct output file path with "natural" identifier
        output_filename = f"{base_name}_cloud_natural_{timestamp}.csv"
        output_file = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        result = generate_cloud_natural(
            csv_file=input_file,
            output_file=output_file,
            inside_regions=regiones_inside,
            cloud_size=None,
            density_multiplier=density_multiplier
        )
        
        success = result.get('success', False) if result else False
        
        if success:
            # Get generated files from result
            generated_files = []
            
            # Add the main CSV file
            if result and result.get('output_file'):
                csv_file = os.path.basename(result['output_file'])
                if os.path.exists(result['output_file']):
                    generated_files.append(csv_file)
            
            # Add visualization files if they exist (PNG and SVG)
            if result and result.get('visualization_file'):
                png_file = os.path.basename(result['visualization_file'])
                if os.path.exists(result['visualization_file']):
                    generated_files.append(png_file)
            
            if result and result.get('visualization_svg_file'):
                svg_file = os.path.basename(result['visualization_svg_file'])
                if os.path.exists(result['visualization_svg_file']):
                    generated_files.append(svg_file)
            
            current_app.logger.info(f"Natural Distribution cloud generation completed successfully. Generated files: {generated_files}")
            current_app.logger.info(f"Total nodes: {result.get('total_nodes', 'unknown')}, Regions: {result.get('regions_generated', 'unknown')}")
            
            return jsonify({
                'success': True,
                'message': 'Cloud with Natural Distribution generated successfully',
                'files': generated_files
            })
        else:
            current_app.logger.error("Natural cloud generation failed")
            return jsonify({'error': 'Error in Natural Distribution cloud generation'}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error in Natural cloud generation: {str(e)}")
        return jsonify({'error': f'Error generating the cloud with Natural Distribution: {str(e)}'}), 500



@cloud_bp.route('/download/<filename>')
def download_file(filename):
    """
    Serve generated files for download.
    
    This function handles file downloads from the output folder, providing
    secure access to generated CSV files, images, and other output files
    created by the application's various processing functions.
    
    Args:
        filename (str): Name of the file to download from the output folder
    
    Returns:
        File download response with the requested file as attachment,
        or JSON error response with 404 status if file not found.
    """
    try:
        return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        current_app.logger.error(f"Error in download_file: {e}")
        return jsonify({'error': 'File not found'}), 404
