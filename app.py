"""
mGFD CloudGenerator - Advanced Cloud of Points Generation Tool

This Flask web application provides an advanced tool for generating clouds of points
that can be used with the meshless Generalized Finite Differences (mGFD) method.
The application offers two main functionalities:

1. Contour Creator: Interactive image processing tool that allows users to upload
   images and detect contour regions through multiple segmentation algorithms:
   - Watershed segmentation for automatic region detection
   - GrabCut algorithm for semi-automatic segmentation with rectangular initialization
   - Interactive segmentation with positive/negative seed markers
   - Manual refinement with brush tools for precise contour adjustment
   Users can visualize, manage, and export detected regions as CSV files containing
   normalized coordinate points.

2. Cloud Generator: Cloud of points generation system that processes CSV files
   containing contour coordinates and generates optimized clouds of points using
   two advanced distribution algorithms:
   - Regular Distribution: Uniform grid-based point generation for structured patterns
   - Natural Distribution: Poisson Disk Sampling for organic, randomly spaced patterns
   The system includes intelligent point reduction capabilities and various output 
   formats (CSV, PNG, SVG).

Key Features:
- Interactive web interface with real-time canvas manipulation
- Multiple image segmentation algorithms (Watershed, GrabCut, Interactive)
- Manual contour refinement with brush-based editing tools
- Intelligent point reduction algorithms for contour optimization
- Two cloud generation methods (Regular and Natural distributions)
- Automated file cleanup and management with configurable retention
- Comprehensive logging system with rotation and debug modes
- RESTful API endpoints for all operations with JSON responses
- Support for multiple image formats (PNG, JPG, JPEG, GIF, BMP)
- Asynchronous cloud of points generation with real-time progress tracking
- Multi-region support for complex geometries with interior holes
- Downloadable results in multiple formats with visualization previews

Technical Implementation:
- Flask web framework with Jinja2 templating and secure file handling
- OpenCV for computer vision, image processing, and segmentation algorithms
- NumPy for numerical computations and array operations
- Pandas for data manipulation, CSV handling, and coordinate processing
- Shapely for geometric operations and polygon validation
- Matplotlib for visualization generation and export
- Threading for background tasks, file cleanup, and progress tracking
- Rotating file handlers for production logging with configurable levels
- Scipy for advanced mathematical operations in point generation

Segmentation Algorithms:
- Watershed: Combined with region growing for robust boundary detection
- GrabCut: Iterative foreground/background separation with user guidance
- Interactive: Seed-based segmentation with positive/negative markers
- Brush Refinement: Manual contour adjustment with stroke-based editing

Point Generation Algorithms:
- Regular Distribution: Grid-based uniform point placement with adaptive spacing
- Natural Distribution: Poisson Disk Sampling for organic point patterns
- Point Reduction: Intelligent contour simplification preserving geometric features
- Multi-region Support: Separate cloud generation for main and interior regions

Author: Gerardo Tinoco-Guerrero
Date: May, 2025
Last Modification: September 25th, 2025

Dependencies:
- Flask >= 2.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Werkzeug >= 2.0.0
- Shapely >= 1.8.0
- Matplotlib >= 3.5.0
- Scipy >= 1.8.0
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import tempfile
import logging
import time
import cv2
import os
import re

# Import project-specific modules
from cloud_generation import generate_cloud_regular, generate_cloud_natural
from reduce_points import reduce_points_by_region
import contour_detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'mGFD_CloudGenerator_2025'

# Crear directorios necesarios
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Configurar logging
if not app.debug:
    file_handler = RotatingFileHandler('logs/mGFD_CloudGenerator.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
else:
    file_handler = RotatingFileHandler('logs/mGFD_CloudGenerator_debug.log', maxBytes=10240000, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)

# Configuraciones
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

FILE_CLEANUP_INTERVAL = 600
FILE_MAX_AGE = 3600

def cleanup_old_files():
    """
    Clean old files from uploads and output folders based on file age.
    
    Removes files older than FILE_MAX_AGE seconds from both upload and output
    directories to prevent disk space accumulation. Logs successful deletions
    and errors during the cleanup process.
    """
    try:
        current_time = time.time()
        
        # Clean uploads
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > FILE_MAX_AGE:
                            try:
                                os.remove(file_path)
                                app.logger.info(f"File deleted: {file_path}")
                            except Exception as e:
                                app.logger.error(f"Error deleting file {file_path}: {e}")
    except Exception as e:
        app.logger.error(f"Error in cleanup_old_files: {e}")

def cleanup_scheduler():
    """
    Continuous file cleanup scheduler that runs in an infinite loop.
    
    Sleeps for FILE_CLEANUP_INTERVAL seconds between cleanup cycles.
    This function is designed to run in a separate daemon thread.
    """
    while True:
        time.sleep(FILE_CLEANUP_INTERVAL)
        cleanup_old_files()

def start_cleanup_scheduler():
    """
    Start the file cleanup scheduler in a separate daemon thread.
    
    Creates and starts a background thread that continuously monitors
    and cleans old files from upload and output directories. The thread
    is marked as daemon to ensure it terminates when the main application exits.
    """
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()
    app.logger.info("Cleanup scheduler started")

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed image extension.
    
    Args:
        filename (str): Name of the file to validate
        
    Returns:
        bool: True if file extension is in ALLOWED_EXTENSIONS, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_csv_file(filename):
    """
    Check if the uploaded file has an allowed CSV extension.
    
    Args:
        filename (str): Name of the file to validate
        
    Returns:
        bool: True if file extension is 'csv', False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

# Main application routes
@app.route('/')
def home():
    """
    Render the main homepage of the application.
    
    Returns:
        str: Rendered HTML template for the home page
    """
    return render_template('home.html')

@app.route('/contour_creator')
def contour_creator():
    """
    Render the contour creator page for image processing and region detection.
    
    Returns:
        str: Rendered HTML template for the contour creator interface
    """
    return render_template('contour_creator.html')

@app.route('/cloud_generator')
def cloud_generator():
    """
    Render the cloud generator page for cloud generation from CSV data.
    
    Returns:
        str: Rendered HTML template for the cloud generator interface
    """
    return render_template('cloud_generator.html')

@app.route('/about')
def about():
    """
    Render the about page with application information and documentation.
    
    Returns:
        str: Rendered HTML template for the about page
    """
    return render_template('about.html')

@app.route('/privacy_notice')
def privacy_notice():
    """
    Render the privacy notice page (Aviso de Privacidad) compliant with Mexican law.
    
    Returns:
        str: Rendered HTML template for the privacy notice page
    """
    return render_template('privacy_notice.html')

@app.route('/upload', methods=['POST'])
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
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'url': f'/uploads/{filename}'
            })
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'})
    
    except Exception as e:
        app.logger.error(f"Error in upload_file: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files from the upload directory.
    
    Args:
        filename (str): Name of the file to serve
        
    Returns:
        Response: File response for the requested uploaded file
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/detect_region', methods=['POST'])
def detect_region():
    """
    Detect region in image using combined segmentation algorithms.
    
    Applies watershed segmentation to detect regions based on user click coordinates.
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
        
        # Cargar imagen
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Aplicar segmentación combinada usando el módulo de detección de contornos
        mask = contour_detection.apply_combined_segmentation(image, x, y, tolerance)
        algorithm_used = 'watershed'
        
        if mask is not None:
            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Seleccionar el mejor contorno (el más grande)
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convertir contorno a formato que espera el frontend
                contour_points = []
                for point in best_contour:
                    contour_points.append({
                        'x': int(point[0][0]) / image.shape[1],  # Normalizar por ancho
                        'y': int(point[0][1]) / image.shape[0]   # Normalizar por alto
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
        app.logger.error(f"Error in detect_region: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        })

# Endpoints de segmentación interactiva
@app.route('/interactive_segmentation', methods=['POST'])
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Convertir coordenadas normalizadas a píxeles
        h, w = image.shape[:2]
        pos_seeds_px = [(int(seed['x'] * w), int(seed['y'] * h)) for seed in positive_seeds]
        neg_seeds_px = [(int(seed['x'] * w), int(seed['y'] * h)) for seed in negative_seeds]
        
        # Aplicar segmentación interactiva
        mask = contour_detection.interactive_segmentation_with_seeds(
            image, pos_seeds_px, neg_seeds_px, tolerance
        )
        
        if mask is not None:
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Seleccionar el mejor contorno
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convertir a formato normalizado
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
        app.logger.error(f"Error in interactive_segmentation: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@app.route('/grabcut_segmentation', methods=['POST'])
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
        rect = data.get('rect')  # {x, y, width, height} normalizado
        iterations = int(data.get('iterations', 5))
        
        # Load image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        # Convertir rectángulo normalizado a píxeles
        rect_px = None
        if rect:
            h, w = image.shape[:2]
            rect_px = (
                int(rect['x'] * w),
                int(rect['y'] * h),
                int(rect['width'] * w),
                int(rect['height'] * h)
            )
        
        # Aplicar GrabCut
        mask = contour_detection.grabcut_interactive(image, rect_px, iterations=iterations)
        
        if mask is not None:
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Seleccionar el mejor contorno
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convertir a formato normalizado
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
        app.logger.error(f"Error in grabcut_segmentation: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@app.route('/refine_with_brush', methods=['POST'])
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Error loading image'})
        
        h, w = image.shape[:2]
        
        # Crear máscara actual desde el contorno
        current_mask = np.zeros((h, w), dtype=np.uint8)
        if current_contour:
            # Convertir contorno normalizado a píxeles
            contour_px = np.array([
                [[int(pt['x'] * w), int(pt['y'] * h)]] for pt in current_contour
            ], dtype=np.int32)
            cv2.fillPoly(current_mask, [contour_px], 255)
        
        # Convertir trazos de pincel a píxeles
        brush_strokes_px = []
        for stroke in brush_strokes:
            stroke_px = {
                'points': [(int(pt['x'] * w), int(pt['y'] * h)) for pt in stroke.get('points', [])],
                'mode': stroke.get('mode', 'add'),
                'size': int(stroke.get('size', 10))
            }
            brush_strokes_px.append(stroke_px)
        
        # Aplicar refinamiento con pincel
        refined_mask = contour_detection.refine_mask_with_brush(current_mask, brush_strokes_px)
        
        if refined_mask is not None:
            # Encontrar contornos
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Seleccionar el mejor contorno
                best_contour = max(contours, key=cv2.contourArea)
                
                # Convertir a formato normalizado
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
        app.logger.error(f"Error in refine_with_brush: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

# Resto de rutas (save_coordinates, export_single_region, etc.)
@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    """
    Save contour coordinates to CSV file format.
    
    Takes coordinate data and saves it as a CSV file with timestamp and optional
    region name. Creates pandas DataFrame from coordinates and exports to output folder.
    
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
        
        # Crear DataFrame
        df = pd.DataFrame(coordinates, columns=['x', 'y'])
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{region_name}_{timestamp}.csv"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Guardar archivo
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/download/{filename}'
        })
        
    except Exception as e:
        app.logger.error(f"Error in save_coordinates: {e}")
        return jsonify({
            'success': False,
            'error': 'Error saving coordinates'
        })

@app.route('/export_single_region', methods=['POST'])
def export_single_region():
    """
    Export a single detected region to a CSV file.
    
    This function processes contour points from a single region and exports them
    to a CSV file with proper coordinate transformation (Y-axis inversion from
    image coordinates to Cartesian coordinates).
    
    Expected JSON payload:
        contour_points (list): List of point objects with 'x' and 'y' coordinates
        region_name (str, optional): Name for the region (defaults to timestamped name)
        filename (str, optional): Original filename reference
    
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
        
        if not contour_points or len(contour_points) == 0:
            return jsonify({'success': False, 'error': 'No contour points to export'})
        
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
            # Invert Y coordinate: convert from image coordinates (Y=0 at top) to cartesian (Y=0 at bottom)
            inverted_y = 1.0 - point['y']
            coordinates.append([point['x'], inverted_y, region_number])
        
        # Crear DataFrame with correct column order: x, y, region
        df = pd.DataFrame(coordinates, columns=['x', 'y', 'region'])
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_region_name = secure_filename(region_name) if region_name else 'region'
        output_filename = f"{safe_region_name}_{timestamp}.csv"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Guardar archivo
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        app.logger.error(f"Error in export_single_region: {e}")
        return jsonify({
            'success': False,
            'error': 'Error exporting region'
        })

@app.route('/save_all_coordinates', methods=['POST'])
def save_all_coordinates():
    """
    Save coordinates from multiple regions to a single CSV file.
    
    This function processes contour points from multiple regions and combines them
    into a single CSV file with proper coordinate transformation (Y-axis inversion
    from image coordinates to Cartesian coordinates) and region identification.
    
    Expected JSON payload:
        regions (list): List of region objects, each containing:
            - contour_points (list): List of point objects with 'x' and 'y' coordinates
            - name (str, optional): Name for the region
        filename (str, optional): Base filename for the output file
    
    Returns:
        JSON response with success status, output filename, and download URL,
        or error message if the operation fails.
    """
    try:
        data = request.get_json()
        
        if not data or 'regions' not in data:
            return jsonify({'success': False, 'error': 'No regions provided'})
        
        regions = data['regions']
        filename = data.get('filename', 'regions')
        
        if not regions or len(regions) == 0:
            return jsonify({'success': False, 'error': 'No regions to save'})
        
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
                    # Invert Y coordinate: convert from image coordinates (Y=0 at top) to cartesian (Y=0 at bottom)
                    inverted_y = 1.0 - point['y']
                    all_coordinates.append([point['x'], inverted_y, region_number])
        
        if not all_coordinates:
            return jsonify({'success': False, 'error': 'No coordinates to save'})
        
        # Create DataFrame with correct column order: x, y, region
        df = pd.DataFrame(all_coordinates, columns=['x', 'y', 'region'])
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = secure_filename(filename) if filename else 'regions'
        output_filename = f"{safe_filename}_all_regions_{timestamp}.csv"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save file
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        app.logger.error(f"Error in save_all_coordinates: {e}")
        return jsonify({
            'success': False,
            'error': 'Error saving coordinates'
        })

@app.route('/upload_csv', methods=['POST'])
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
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Leer y validar CSV
            try:
                df = pd.read_csv(filepath)
                
                # Validate required columns
                required_columns = ['x', 'y']
                if not all(col in df.columns for col in required_columns):
                    os.remove(filepath)
                    return jsonify({
                        'success': False,
                        'error': f'CSV file must contain columns: {", ".join(required_columns)}'
                    })
                
                # Validate numeric data
                if not pd.api.types.is_numeric_dtype(df['x']) or not pd.api.types.is_numeric_dtype(df['y']):
                    os.remove(filepath)
                    return jsonify({
                        'success': False,
                        'error': 'Columns x and y must contain numeric values'
                    })
                
                # Analizar regiones si existe la columna 'region'
                regions_info = []
                total_regions = 1
                
                if 'region' in df.columns:
                    unique_regions = df['region'].unique()
                    total_regions = len(unique_regions)
                    regions_info = [f"Region {int(r)}" for r in sorted(unique_regions) if pd.notna(r)]
                else:
                    regions_info = ["Region 1"]
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'total_points': len(df),
                    'regions': total_regions,
                    'region_list': regions_info,
                    'rows': len(df),
                    'preview': df.head(5).to_dict('records')
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
        app.logger.error(f"Error in upload_csv: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@app.route('/generate_cloud', methods=['POST'])
def generate_cloud_api():
    """
    API endpoint to generate a cloud of points from uploaded CSV data.
    
    This endpoint processes CSV files containing coordinate data and generates
    optimized clouds of points using Regular Distribution algorithms.
    
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
            "message": "Cloud of points generated successfully",
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
        - Direct execution using Regular Distribution (scipy.spatial)
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
            "message": "Cloud of points generated successfully",
            "files": ["coordinates_nodes_20250823_143022.csv", "coordinates_elements_20250823_143022.csv"]
        }
    """
    data = request.get_json()
    csv_filename = data.get('csv_filename')
    regiones_inside = data.get('regiones_inside', False)
    reduce_points_flag = data.get('reduce_points', False)
    reduce_points_multiplier = data.get('reduce_points_multiplier', 2)
    
    if not csv_filename:
        return jsonify({'error': 'No CSV file specified'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(csv_filename)[0]
        
        input_file = filepath
        
        # Apply point reduction if requested
        if reduce_points_flag:
            try:
                
                app.logger.info(f"Applying point reduction to {csv_filename}")
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    temp_reduced_file = temp_file.name
                
                # Apply standard reduction without filtering - let cloud_generation_delaunay handle filtering
                result = reduce_points_by_region(filepath, temp_reduced_file, reduce_points_multiplier)
                
                if result is not None:
                    input_file = temp_reduced_file
                    app.logger.info("Point reduction applied successfully")
                else:
                    app.logger.warning("Point reduction failed, using original file")
                    
            except ImportError:
                app.logger.warning("Point reduction module not available, using original file")
            except Exception as e:
                app.logger.error(f"Error in point reduction: {str(e)}, using original file")
        
        # Generate cloud of points using Regular Distribution
        app.logger.info(f"Starting cloud generation for {csv_filename}")
        
        # Construct output file path
        output_filename = f"{base_name}_cloud_{timestamp}.csv"
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        result = generate_cloud_regular(
            csv_file=input_file,
            output_file=output_file,
            regiones_inside=regiones_inside,
            reducir_contorno=reduce_points_flag,  # Use the actual reduce_points flag from frontend
            porcentaje_reduccion=reduce_points_multiplier * 5,  # Convert multiplier to percentage
            cloud_size=None
        )
        
        success = result.get('success', False) if result else False
        
        # Clean up temporary file if it was created
        if reduce_points_flag and input_file != filepath:
            try:
                os.remove(input_file)
            except:
                pass
        
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
            
            app.logger.info(f"Cloud generation completed successfully. Generated files: {generated_files}")
            app.logger.info(f"Total nodes: {result.get('total_nodes', 'unknown')}, Regions: {result.get('regions_generated', 'unknown')}")
            
            return jsonify({
                'success': True,
                'message': 'Cloud generated successfully',
                'files': generated_files
            })
        else:
            app.logger.error("Cloud generation failed")
            return jsonify({'error': 'Error in cloud generation'}), 500
            
    except Exception as e:
        app.logger.error(f"Error in cloud generation: {str(e)}")
        return jsonify({'error': f'Error generating the cloud: {str(e)}'}), 500


@app.route('/generate_cloud_natural', methods=['POST'])
def generate_cloud_natural_api():
    """
    API endpoint to generate a cloud of points using Natural Distribution for more natural distribution.
    
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
            "message": "Natural Distribution cloud of points generated successfully",
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
    
    if not csv_filename:
        return jsonify({'error': 'No CSV file specified'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'CSV file not found'}), 404
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(csv_filename)[0]
        
        input_file = filepath
        
        # Apply point reduction if requested
        if reduce_points_flag:
            try:
                app.logger.info(f"Applying point reduction to {csv_filename} for Natural Distribution")
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    temp_reduced_file = temp_file.name
                
                result = reduce_points_by_region(filepath, temp_reduced_file, reduce_points_multiplier)
                
                if result is not None:
                    input_file = temp_reduced_file
                    app.logger.info("Point reduction applied successfully for Natural Distribution")
                else:
                    app.logger.warning("Point reduction failed, using original file for Natural Distribution")
                    
            except ImportError:
                app.logger.warning("Point reduction module not available, using original file for Natural Distribution")
            except Exception as e:
                app.logger.error(f"Error in point reduction for Natural Distribution: {str(e)}, using original file")
        
        # Generate cloud of points using Natural Distribution
        app.logger.info(f"Starting Natural Distribution cloud generation for {csv_filename}")
        
        # Construct output file path with "natural" identifier
        output_filename = f"{base_name}_cloud_natural_{timestamp}.csv"
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        result = generate_cloud_natural(
            csv_file=input_file,
            output_file=output_file,
            regiones_inside=regiones_inside,
            reducir_contorno=reduce_points_flag,
            porcentaje_reduccion=reduce_points_multiplier * 5,
            cloud_size=None
        )
        
        success = result.get('success', False) if result else False
        
        # Clean up temporary file if it was created
        if reduce_points_flag and input_file != filepath:
            try:
                os.remove(input_file)
            except:
                pass
        
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
            
            app.logger.info(f"Natural Distribution cloud generation completed successfully. Generated files: {generated_files}")
            app.logger.info(f"Total nodes: {result.get('total_nodes', 'unknown')}, Regions: {result.get('regions_generated', 'unknown')}")
            
            return jsonify({
                'success': True,
                'message': 'Cloud with Natural Distribution generated successfully',
                'files': generated_files
            })
        else:
            app.logger.error("Natural cloud generation failed")
            return jsonify({'error': 'Error in Natural Distribution cloud generation'}), 500
            
    except Exception as e:
        app.logger.error(f"Error in Natural cloud generation: {str(e)}")
        return jsonify({'error': f'Error generating the cloud with Natural Distribution: {str(e)}'}), 500


@app.route('/download/<filename>')
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
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error in download_file: {e}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    start_cleanup_scheduler()
    app.run(debug=True, host='0.0.0.0', port=8080)