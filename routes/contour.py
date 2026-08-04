from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app, send_file, url_for
import os
import cv2
import numpy as np
import csv
import time
import tempfile
import threading
import re
from werkzeug.utils import secure_filename
from datetime import datetime

from contour_modules import detection as contour_detection

from . import contour_bp


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def calculate_export_scale(w, h, config=None):
    if config is None:
        config = {}
        
    method = config.get('method', 'preserve_aspect_ratio')
    
    if method == 'custom':
        return float(config.get('custom_x', 1.0)), float(config.get('custom_y', 1.0))
        
    if method == 'stretch':
        return 1.0, 1.0
        
    max_dim = max(w, h)
    if max_dim > 0:
        return w / max_dim, h / max_dim
        
    return 1.0, 1.0

def transform_coordinate(point, scale_x, scale_y):
    final_x = point['x'] * scale_x
    inverted_y = (1.0 - point['y']) * scale_y
    
    final_x = max(0.0, min(1.0, final_x))
    inverted_y = max(0.0, min(1.0, inverted_y))
    
    return final_x, inverted_y
    
@contour_bp.route('/contour_creator')
def contour_creator():
    """
    Render the contour creator page for image processing and region detection.
    
    Returns:
        str: Rendered HTML template for the contour creator interface
    """
    return render_template('contour_creator.html')


@contour_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image file uploads with validation and secure storage.
    
    Accepts image files via POST request, validates file type and extension,
    generates unique timestamped filename, and saves to upload directory.
    
    Returns:
        JSON: Response containing success status, filename, and URL on success,
              or error message on failure
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Use url_for to generate correct URL regardless of application root
            file_url = url_for('contour.uploaded_file', filename=filename)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'url': file_url
            })
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'})
    
    except Exception as e:
        current_app.logger.error(f"Error in upload_file: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})


@contour_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files from the upload directory.
    
    Args:
        filename (str): Name of the file to serve
        
    Returns:
        Response: File response for the requested uploaded file
    """
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)




@contour_bp.route('/detect_region', methods=['POST'])
def detect_region():
    """
    Detect region in image using optimized single-region segmentation.
    
    Applies adaptive flood fill segmentation to detect regions based on user click coordinates.
    Processes the resulting mask to find contours and returns the largest contour
    as normalized coordinates for frontend display.
    
    Expected JSON payload:
        filename (str): Name of uploaded image file
        x (int): X coordinate of user click
        y (int): Y coordinate of user click
        tolerance (int, optional): Segmentation tolerance (default: 30)
    
    Returns:
        JSON: Response containing success status, normalized contour points,
              and algorithm used on success, or error message on failure
    """
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data or 'x' not in data or 'y' not in data:
            return jsonify({'success': False, 'error': 'Incomplete data'})
        
        filename = data['filename']
        x = int(data['x'])
        y = int(data['y'])
        tolerance = int(data.get('tolerance', 30))
        
        # Load image
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Apply optimized single-region detection
        mask = contour_detection.detect_region_at_point(image, x, y, tolerance)
        algorithm_used = 'floodfill'
        
        if mask is not None:
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # Select the best contour (the largest one)
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convert contour to format expected by frontend
                contour_points = []
                for point in best_contour:
                    contour_points.append({
                        'x': int(point[0][0]) / image.shape[1],  # Normalize by width
                        'y': int(point[0][1]) / image.shape[0]   # Normalize by height
                    })
                
                return jsonify({
                    'success': True,
                    'contour_points': contour_points,
                    'algorithm': algorithm_used or 'combined'
                })
        
        return jsonify({
            'success': False,
            'error': 'Could not detect region'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in detect_region: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        })

# Interactive segmentation endpoints

@contour_bp.route('/interactive_segmentation', methods=['POST'])
def interactive_segmentation():
    """
    Perform interactive segmentation using positive and negative seed markers.
    
    Uses user-provided positive and negative seed points to guide segmentation
    algorithm. Converts normalized coordinates to pixel coordinates and applies
    interactive segmentation with seeds to generate region mask.
    
    Expected JSON payload:
        filename (str): Name of uploaded image file
        positive_seeds (list): List of positive seed points with x, y coordinates
        negative_seeds (list): List of negative seed points with x, y coordinates
        tolerance (int, optional): Segmentation tolerance (default: 30)
    
    Returns:
        JSON: Response containing success status, normalized contour points,
              and algorithm identifier on success, or error message on failure
    """
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({'success': False, 'error': 'Incomplete data'})
        
        filename = data['filename']
        positive_seeds = data.get('positive_seeds', [])
        negative_seeds = data.get('negative_seeds', [])
        tolerance = int(data.get('tolerance', 30))
        
        # Load image
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Convert normalized coordinates to pixels
        h, w = image.shape[:2]
        pos_seeds_px = [(int(seed['x'] * w), int(seed['y'] * h)) for seed in positive_seeds]
        neg_seeds_px = [(int(seed['x'] * w), int(seed['y'] * h)) for seed in negative_seeds]
        
        # Apply interactive segmentation
        mask = contour_detection.interactive_segmentation_with_seeds(
            image, pos_seeds_px, neg_seeds_px, tolerance
        )
        
        if mask is not None:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # Select best contour
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convert to normalized format
                contour_points = []
                for point in best_contour:
                    contour_points.append({
                        'x': int(point[0][0]) / w,
                        'y': int(point[0][1]) / h
                    })
                
                return jsonify({
                    'success': True,
                    'contour_points': contour_points,
                    'algorithm': 'interactive_seeds'
                })
        
        return jsonify({'success': False, 'error': 'Could not generate segmentation'})
        
    except Exception as e:
        current_app.logger.error(f"Error in interactive_segmentation: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})


@contour_bp.route('/grabcut_segmentation', methods=['POST'])
def grabcut_segmentation():
    """
    Perform interactive segmentation using GrabCut algorithm.
    
    Applies GrabCut algorithm with optional rectangular initialization region.
    Converts normalized rectangle coordinates to pixel coordinates and runs
    iterative GrabCut segmentation to separate foreground from background.
    
    Expected JSON payload:
        filename (str): Name of uploaded image file
        rect (dict, optional): Rectangle with x, y, width, height (normalized)
        iterations (int, optional): Number of GrabCut iterations (default: 5)
    
    Returns:
        JSON: Response containing success status, normalized contour points,
              and algorithm identifier on success, or error message on failure
    """
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({'success': False, 'error': 'Incomplete data'})
        
        filename = data['filename']
        rect = data.get('rect')  # {x, y, width, height} normalized
        iterations = int(data.get('iterations', 5))
        
        # Load image
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Convert normalized rectangle to pixels
        rect_px = None
        if rect:
            h, w = image.shape[:2]
            rect_px = (
                int(rect['x'] * w),
                int(rect['y'] * h),
                int(rect['width'] * w),
                int(rect['height'] * h)
            )
        
        # Apply GrabCut
        mask = contour_detection.grabcut_interactive(image, rect_px, iterations=iterations)
        
        if mask is not None:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # Select the best contour
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convert to normalized format
                h, w = image.shape[:2]
                contour_points = []
                for point in best_contour:
                    contour_points.append({
                        'x': int(point[0][0]) / w,
                        'y': int(point[0][1]) / h
                    })
                
                return jsonify({
                    'success': True,
                    'contour_points': contour_points,
                    'algorithm': 'grabcut'
                })
        
        return jsonify({'success': False, 'error': 'Could not generate segmentation'})
        
    except Exception as e:
        current_app.logger.error(f"Error in grabcut_segmentation: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})


@contour_bp.route('/refine_with_brush', methods=['POST'])
def refine_with_brush():
    """
    Refine segmentation mask using brush strokes.
    
    Takes current contour and applies user brush strokes to add or remove regions.
    Converts normalized coordinates to pixels, creates mask from current contour,
    applies brush modifications, and returns refined contour.
    
    Expected JSON payload:
        filename (str): Name of uploaded image file
        current_contour (list): Current contour points with x, y coordinates
        brush_strokes (list): List of brush strokes with points, mode, and size
    
    Returns:
        JSON: Response containing success status, refined contour points,
              and algorithm identifier on success, or error message on failure
    """
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data or 'current_contour' not in data:
            return jsonify({'success': False, 'error': 'Incomplete data'})
        
        filename = data['filename']
        current_contour = data['current_contour']
        brush_strokes = data.get('brush_strokes', [])
        
        # Load image
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        h, w = image.shape[:2]
        
        # Create current mask from contour
        current_mask = np.zeros((h, w), dtype=np.uint8)
        if current_contour:
            # Convert normalized contour to pixels
            contour_px = np.array([
                [[int(pt['x'] * w), int(pt['y'] * h)]] for pt in current_contour
            ], dtype=np.int32)
            cv2.fillPoly(current_mask, [contour_px], 255)
        
        # Convert brush strokes to pixels
        brush_strokes_px = []
        for stroke in brush_strokes:
            stroke_px = {
                'points': [(int(pt['x'] * w), int(pt['y'] * h)) for pt in stroke.get('points', [])],
                'mode': stroke.get('mode', 'add'),
                'size': int(stroke.get('size', 10))
            }
            brush_strokes_px.append(stroke_px)
        
        # Apply brush refinement
        refined_mask = contour_detection.refine_mask_with_brush(current_mask, brush_strokes_px)
        
        if refined_mask is not None:
            # Get refined contour
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # Select the best contour
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convert to normalized format
                contour_points = []
                for point in best_contour:
                    contour_points.append({
                        'x': int(point[0][0]) / w,
                        'y': int(point[0][1]) / h
                    })
                
                return jsonify({
                    'success': True,
                    'contour_points': contour_points,
                    'algorithm': 'brush_refined'
                })
        
        return jsonify({'success': False, 'error': 'Could not refine mask'})
        
    except Exception as e:
        current_app.logger.error(f"Error in refine_with_brush: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

# Remaining routes (save_coordinates, export_single_region, etc.)

@contour_bp.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    """
    Save contour coordinates to CSV file format.
    
    Takes coordinate data and saves it as a CSV file with timestamp and optional
    region name. Writes coordinates directly to CSV in output folder.
    
    Expected JSON payload:
        coordinates (list): List of coordinate points with x, y values
        region_name (str, optional): Name for the region (default: timestamped name)
    
    Returns:
        JSON: Response containing success status, filename, and download URL
              on success, or error message on failure
    """
    try:
        data = request.get_json()
        
        if not data or 'coordinates' not in data:
            return jsonify({'success': False, 'error': 'No coordinates provided'})
        
        coordinates = data['coordinates']
        region_name = data.get('region_name', f'region_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{region_name}_{timestamp}.csv"
        filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], filename)
        
        # Save file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for coord in coordinates:
                if isinstance(coord, dict):
                    writer.writerow([coord.get('x'), coord.get('y')])
                elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    writer.writerow(coord[:2])
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/download/{filename}'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in save_coordinates: {e}")
        return jsonify({
            'success': False,
            'error': 'Error saving coordinates'
        })


@contour_bp.route('/export_single_region', methods=['POST'])
def export_single_region():
    """
    Export a single detected region to a CSV file.
    
    This function processes contour points from a single region and exports them
    to a CSV file with proper coordinate transformation (Y-axis inversion from
    image coordinates to Cartesian coordinates).
    
    Adjustments:
    - Scales coordinates to fit within [0,1]x[0,1] while preserving aspect ratio.
    - Clamps values to strictly ensure no coordinate exceeds [0,1] range.
    - Validates that max dimension is positive before scaling.
    - Supports custom scaling configuration via 'scaling_config' parameter.
    
    Expected JSON payload:
        contour_points (list): List of point objects with 'x' and 'y' coordinates
        region_name (str, optional): Name for the region (defaults to timestamped name)
        filename (str, optional): Original filename reference
        scaling_config (dict, optional): Scaling configuration
            - method: 'preserve_aspect_ratio', 'stretch', 'custom'
            - custom_x, custom_y: float values for custom scaling
    
    Returns:
        JSON response with success status, output filename, and download URL,
        or error message if the operation fails.
    """
    try:
        data = request.get_json()
        
        if not data or 'contour_points' not in data:
            return jsonify({'success': False, 'error': 'No contour points provided'})
        
        contour_points = data['contour_points']
        region_name = data.get('region_name', f'region_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        filename = data.get('filename', '')
        scaling_config = data.get('scaling_config', {})
        normalize = data.get('normalize', True)
        
        if not contour_points or len(contour_points) == 0:
            return jsonify({'success': False, 'error': 'No contour points to export'})
        
        # Calculate scaling factors
        scale_x = 1.0
        scale_y = 1.0
        
        if filename:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
                if image is not None:
                    h, w = image.shape[:2]
                    scale_x, scale_y = calculate_export_scale(w, h, scaling_config)
        
        # Convert contour points to coordinates list
        coordinates = []
        # Extract region number from region_name (e.g., "Region 1" -> 1)
        region_number = 1  # Default
        if region_name and 'region' in region_name.lower():
            try:
                # Try to extract number from region name
                match = re.search(r'(\d+)', region_name)
                if match:
                    region_number = int(match.group(1))
            except:
                region_number = 1
        
        for point in contour_points:
            # Transform coordinate with scaling and clamping
            final_x, inverted_y = transform_coordinate(point, scale_x, scale_y)
            coordinates.append([final_x, inverted_y, region_number])
            
        # Normalize all coordinates exactly to [0,1]x[0,1]
        if coordinates and normalize:
            min_x = min(c[0] for c in coordinates)
            max_x = max(c[0] for c in coordinates)
            min_y = min(c[1] for c in coordinates)
            max_y = max(c[1] for c in coordinates)
            
            range_x = max_x - min_x
            range_y = max_y - min_y
            max_range = max(range_x, range_y)
            if max_range == 0:
                max_range = 1.0
            
            for c in coordinates:
                c[0] = (c[0] - min_x) / max_range
                c[1] = (c[1] - min_y) / max_range
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_region_name = secure_filename(region_name) if region_name else 'region'
        output_filename = f"{safe_region_name}_{timestamp}.csv"
        filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'region'])
            writer.writerows(coordinates)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in export_single_region: {e}")
        return jsonify({
            'success': False,
            'error': 'Error exporting region'
        })
 

@contour_bp.route('/save_all_coordinates', methods=['POST'])
def save_all_coordinates():
    """
    Save coordinates from multiple regions to a single CSV file.
    
    This function processes contour points from multiple regions and combines them
    into a single CSV file with proper coordinate transformation (Y-axis inversion
    from image coordinates to Cartesian coordinates) and region identification.
    
    Adjustments:
    - Scales coordinates to fit within [0,1]x[0,1] while preserving aspect ratio.
    - Clamps values to strictly ensure no coordinate exceeds [0,1] range.
    - Validates that max dimension is positive before scaling.
    - Supports custom scaling configuration via 'scaling_config' parameter.
    
    Expected JSON payload:
        regions (list): List of region objects, each containing:
            - contour_points (list): List of point objects with 'x' and 'y' coordinates
            - name (str, optional): Name for the region
        filename (str, optional): Base filename for the output file
        scaling_config (dict, optional): Scaling configuration
            - method: 'preserve_aspect_ratio', 'stretch', 'custom'
            - custom_x, custom_y: float values for custom scaling
    
    Returns:
        JSON response with success status, output filename, and download URL,
        or error message if the operation fails.
    """
    try:
        data = request.get_json()
        
        if not data or 'regions' not in data:
            return jsonify({'success': False, 'error': 'No regions provided'})
        
        regions = data['regions']
        filename = data.get('filename', '')
        scaling_config = data.get('scaling_config', {})
        normalize = data.get('normalize', True)
        
        if not regions or len(regions) == 0:
            return jsonify({'success': False, 'error': 'No regions to save'})
        
        # Calculate scaling factors
        scale_x = 1.0
        scale_y = 1.0
        
        if filename:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
                if image is not None:
                    h, w = image.shape[:2]
                    scale_x, scale_y = calculate_export_scale(w, h, scaling_config)
        
        # Create a single CSV file with all regions
        all_coordinates = []
        
        for i, region in enumerate(regions):
            if 'contour_points' in region and region['contour_points']:
                region_name = region.get('name', f'region_{i+1}')
                
                # Extract region number from region_name (e.g., "Region 1" -> 1)
                region_number = i + 1  # Default to index + 1
                if region_name and 'region' in region_name.lower():
                    try:
                        # Try to extract number from region name
                        match = re.search(r'(\d+)', region_name)
                        if match:
                            region_number = int(match.group(1))
                    except:
                        region_number = i + 1
                
                # Convert contour points to coordinates with region number
                for point in region['contour_points']:
                    # Transform coordinate with scaling and clamping
                    final_x, inverted_y = transform_coordinate(point, scale_x, scale_y)
                    all_coordinates.append([final_x, inverted_y, region_number])
        
        if not all_coordinates:
            return jsonify({'success': False, 'error': 'No coordinates to save'})
            
        # Normalize all coordinates exactly to [0,1]x[0,1]
        if normalize:
            min_x = min(c[0] for c in all_coordinates)
            max_x = max(c[0] for c in all_coordinates)
            min_y = min(c[1] for c in all_coordinates)
            max_y = max(c[1] for c in all_coordinates)
            
            range_x = max_x - min_x
            range_y = max_y - min_y
            max_range = max(range_x, range_y)
            if max_range == 0:
                max_range = 1.0
            
            for c in all_coordinates:
                c[0] = (c[0] - min_x) / max_range
                c[1] = (c[1] - min_y) / max_range
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = secure_filename(filename) if filename else 'regions'
        output_filename = f"{safe_filename}_all_regions_{timestamp}.csv"
        filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'region'])
            writer.writerows(all_coordinates)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in save_all_coordinates: {e}")
        return jsonify({
            'success': False,
            'error': 'Error saving coordinates'
        })


