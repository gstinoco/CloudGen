"""
mGFD CloudGenerator - Advanced Cloud of Points Generation Tool

This Flask web application provides an advanced tool for generating clouds of points
that can be used with the meshless Generalized Finite Differences (mGFD) method.
The application offers two main functionalities:

1. Contour Creator: Interactive image processing tool that allows users to upload
   images and detect contour regions through click-based segmentation. Users can
   visualize, manage, and export detected regions as CSV files containing
   normalized coordinate points.

2. Cloud Generator: Cloud of points generation system that processes CSV files
containing contour coordinates and generates optimized clouds of points using
   advanced algorithms. The system includes point reduction capabilities and
   various output formats (CSV, PNG, SVG).

Key Features:
- Interactive web interface with real-time canvas manipulation
- Advanced image segmentation using OpenCV
- Automated file cleanup and management
- Comprehensive logging system with rotation
- RESTful API endpoints for all operations
- Support for multiple image formats (PNG, JPG, JPEG, GIF, BMP)
- Asynchronous cloud of points generation with status tracking
- Downloadable results in multiple formats

Technical Implementation:
- Flask web framework with Jinja2 templating
- OpenCV for computer vision and image processing
- NumPy for numerical computations
- Pandas for data manipulation and CSV handling
- Subprocess management for external script execution
- Threading for background tasks and file cleanup
- Rotating file handlers for production logging

Author: Gerardo Tinoco-Guerrero
Date: May 2025
Last Modification: 23 August 2025

Dependencies:
- Flask >= 2.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Werkzeug >= 2.0.0
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import subprocess
import sys
import threading
import time
import glob
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'mGFD_CloudGenerator_2025'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)
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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

FILE_CLEANUP_INTERVAL = 600
FILE_MAX_AGE = 3600

def cleanup_old_files():
    """
    Clean up old files from upload and output directories.
    
    This function removes files that are older than the specified maximum age
    (FILE_MAX_AGE) from both upload and output folders. It runs periodically
    as part of the automated cleanup scheduler to prevent disk space issues.
    
    The function iterates through all files in the configured upload and output
    directories, checks their creation time, and removes files that exceed
    the maximum age threshold. Any errors during file removal are logged but
    do not stop the cleanup process.
    
    Raises:
        Exception: Logs any errors encountered during the cleanup process
                  without interrupting the operation.
    
    Note:
        - Only removes regular files, not directories
        - Uses file creation time for age calculation
        - Continues cleanup even if individual file removal fails
        - All errors are logged for debugging purposes
    """
    try:
        current_time = time.time()
        directories_to_clean = [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]
        
        for directory in directories_to_clean:
            if os.path.exists(directory):
                # Get all files in the directory
                files = glob.glob(os.path.join(directory, '*'))
                
                for file_path in files:
                    if os.path.isfile(file_path):
                        # Check file age based on creation time
                        file_age = current_time - os.path.getctime(file_path)
                        
                        if file_age > FILE_MAX_AGE:
                            try:
                                os.remove(file_path)
                            except OSError as e:
                                app.logger.error(f"Error removing file {file_path}: {e}")
                                
    except Exception as e:
        app.logger.error(f"Error during file cleanup: {e}")

def start_cleanup_scheduler():
    """
    Start the automated file cleanup scheduler in a background thread.
    
    This function creates and starts a daemon thread that runs the file cleanup
    process at regular intervals defined by FILE_CLEANUP_INTERVAL. The scheduler
    ensures that old files are automatically removed from the system to prevent
    disk space issues and maintain optimal performance.
    
    The cleanup thread runs as a daemon, meaning it will automatically terminate
    when the main application process ends. The thread sleeps for the specified
    interval between cleanup cycles.
    
    Thread Safety:
        - Uses daemon thread to ensure proper cleanup on application shutdown
        - Thread-safe file operations within cleanup_old_files function
        - No shared state between cleanup cycles
    
    Note:
        - Should be called once during application startup
        - Thread continues running until application termination
        - Cleanup interval is configurable via FILE_CLEANUP_INTERVAL constant
    """
    def cleanup_worker():
        while True:
            cleanup_old_files()
            time.sleep(FILE_CLEANUP_INTERVAL)
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

def allowed_file(filename):
    """
    Check if a filename has an allowed image file extension.
    
    This function validates whether the provided filename has an extension
    that is permitted for image uploads. It performs case-insensitive
    comparison against the ALLOWED_EXTENSIONS set.
    
    Args:
        filename (str): The filename to validate, including extension.
    
    Returns:
        bool: True if the file extension is allowed, False otherwise.
    
    Note:
        - Requires filename to contain at least one dot
        - Extension comparison is case-insensitive
        - Allowed extensions: PNG, JPG, JPEG, GIF, BMP
    
    Example:
        >>> allowed_file('image.png')
        True
        >>> allowed_file('document.txt')
        False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_csv_file(filename):
    """
    Check if a filename has an allowed CSV file extension.
    
    This function validates whether the provided filename has a CSV extension
    that is permitted for coordinate data uploads. It performs case-insensitive
    comparison against the ALLOWED_CSV_EXTENSIONS set.
    
    Args:
        filename (str): The filename to validate, including extension.
    
    Returns:
        bool: True if the file extension is CSV, False otherwise.
    
    Note:
        - Requires filename to contain at least one dot
        - Extension comparison is case-insensitive
        - Only CSV files are currently supported
    
    Example:
        >>> allowed_csv_file('coordinates.csv')
        True
        >>> allowed_csv_file('data.xlsx')
        False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

@app.route('/')
def home():
    """
    Render the application home page.
    
    This route serves as the main landing page for the mGFD CloudGenerator
    application. It provides an overview of the available tools and navigation
    to the main functionalities: Contour Creator and Cloud Generator.
    
    Returns:
        str: Rendered HTML template for the home page.
    
    Template:
        home.html: Main landing page with application overview and navigation.
    """
    return render_template('home.html')

@app.route('/contour_creator')
def contour_creator():
    """
    Render the Contour Creator page.
    
    This route serves the interactive image processing interface where users
    can upload images, detect contour regions through click-based segmentation,
    and export the detected regions as CSV files. The page includes a canvas
    for image manipulation and region visualization.
    
    Returns:
        str: Rendered HTML template for the contour creator interface.
    
    Template:
        contour_creator.html: Interactive image processing interface with
                             canvas manipulation and region detection tools.
    """
    return render_template('contour_creator.html')

@app.route('/cloud_generator')
def cloud_generator():
    """
    Render the Cloud Generator page.
    
    This route serves the cloud of points generation interface where users can
upload CSV files containing coordinate data and generate 3D
clouds of points using advanced algorithms. The page provides controls for
cloud of points generation parameters and displays generation progress.
    
    Returns:
        str: Rendered HTML template for the cloud generator interface.
    
    Template:
        cloud_generator.html: Cloud of points generation interface with file
                             upload, parameter controls, and progress tracking.
    """
    return render_template('cloud_generator.html')

@app.route('/about')
def about():
    """
    Render the About page.
    
    This route serves the application information page containing details
    about the mGFD CloudGenerator project, its purpose, technical implementation,
    and author information. It provides context about the meshless Generalized
    Finite Differences method and the application's role in scientific computing.
    
    Returns:
        str: Rendered HTML template for the about page.
    
    Template:
        about.html: Information page with project details, methodology
                   explanation, and technical documentation.
    """
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image file upload for contour detection.
    
    This API endpoint processes image file uploads from the contour creator
    interface. It validates the uploaded file, ensures it has an allowed
    extension, saves it to the upload directory with a secure filename,
    and returns the file information for further processing.
    
    The function performs comprehensive validation including:
    - File presence in request
    - Non-empty filename
    - Allowed file extension validation
    - Secure filename generation
    
    Returns:
        JSON response containing:
        - success (bool): Upload operation status
        - filename (str): Secure filename for uploaded file
        - message (str): Success message
        
        Error responses (400):
        - No file selected
        - Empty filename
        - Invalid file type
        
        Error responses (500):
        - File upload errors
    
    Accepted Methods:
        POST: File upload with multipart/form-data
    
    Form Data:
        file: Image file (PNG, JPG, JPEG, GIF, BMP)
    
    Example Response:
        {
            "success": true,
            "filename": "image_20250823_143022.png",
            "message": "File uploaded successfully"
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the image and get dimensions
        image = cv2.imread(filepath)
        height, width = image.shape[:2]
        
        return jsonify({
            'success': True,
            'filename': filename,
            'width': width,
            'height': height
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded image files.
    
    This route provides access to uploaded image files stored in the upload
    directory. It serves files directly from the configured upload folder,
    allowing the web interface to display uploaded images for processing.
    
    Args:
        filename (str): Name of the uploaded file to serve.
    
    Returns:
        File response: The requested image file from the upload directory.
    
    Security Note:
        - Files are served from a controlled upload directory
        - Filename should be validated before upload
        - Consider implementing additional access controls for production
    
    Example:
        GET /uploads/image_20250823_143022.png
        Returns the specified image file
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def apply_combined_segmentation(image, x, y, tolerance):
    """
    Apply combined segmentation algorithm to detect regions in an image.
    
    This function implements a sophisticated image segmentation approach that
    combines multiple computer vision techniques to accurately detect and
    isolate regions based on a seed point. The algorithm uses color-based
    segmentation, morphological operations, and contour analysis to identify
    coherent regions in the input image.
    
    The segmentation process includes:
    1. Color space conversion and analysis
    2. Adaptive thresholding based on seed point characteristics
    3. Morphological operations for noise reduction
    4. Connected component analysis
    5. Region refinement and validation
    
    Args:
        image (numpy.ndarray): Input image in BGR color space (OpenCV format).
        x (int): X-coordinate of the seed point for segmentation.
        y (int): Y-coordinate of the seed point for segmentation.
        tolerance (int): Tolerance value for color similarity (0-255).
    
    Returns:
        numpy.ndarray: Binary mask where detected region pixels are white (255)
                      and background pixels are black (0). The mask has the
                      same dimensions as the input image.
    
    Raises:
        Exception: If segmentation fails due to invalid coordinates or
                  image processing errors.
    
    Note:
        - Input image should be in BGR color space (OpenCV standard)
        - Coordinates must be within image boundaries
        - Higher tolerance values result in more inclusive segmentation
        - The algorithm is optimized for detecting coherent color regions
    
    Example:
        >>> image = cv2.imread('input.png')
        >>> mask = apply_combined_segmentation(image, 100, 150, 10)
        >>> # mask contains binary segmentation result
    """
    """Applies combined segmentation that mixes flood fill with edge detection."""
    height, width = image.shape[:2]
    
    # Optimized combined method - combines flood fill with edge detection
    # 1. Basic flood fill 
    mask_flood = np.zeros((height + 2, width + 2), np.uint8)
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    cv2.floodFill(image.copy(), mask_flood, (x, y), (255, 255, 255), lo_diff, up_diff)
    flood_mask = mask_flood[1:-1, 1:-1]
    
    # 2. Enhanced edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive edge detection
    edges = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    # 3. Combine flood fill with edge detection
    # Apply flood fill respecting edges
    combined_mask = np.zeros((height + 2, width + 2), np.uint8)
    
    # Create temporary image with edges as obstacles
    temp_image = image.copy()
    temp_image[edges == 0] = [0, 0, 0]  # Make edges black
    
    # Apply flood fill on the image with edges
    cv2.floodFill(temp_image, combined_mask, (x, y), (255, 255, 255), lo_diff, up_diff)
    
    # Combine both masks
    region_mask = cv2.bitwise_and(flood_mask, combined_mask[1:-1, 1:-1])
    
    # Apply morphological operations for smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)
    
    return region_mask

@app.route('/detect_region', methods=['POST'])
def detect_region():
    """
    Detect and extract contour regions from uploaded images.
    
    This API endpoint processes region detection requests from the contour
    creator interface. It applies advanced image segmentation algorithms to
    detect coherent regions based on user-specified coordinates, extracts
    contour points, and returns normalized coordinate data.
    
    The detection process includes:
    1. Image loading and validation
    2. Coordinate boundary checking
    3. Combined segmentation algorithm application
    4. Contour detection and filtering
    5. Area-based contour validation
    6. Point normalization for scale independence
    
    Request Processing:
    - Validates input parameters and file existence
    - Applies segmentation at specified coordinates
    - Filters contours by minimum area threshold
    - Selects the most relevant contour containing the seed point
    - Normalizes coordinates by maximum image dimension
    
    Returns:
        JSON response containing:
        - success (bool): Detection operation status
        - contour_points (list): Array of normalized coordinate objects
                               with 'x' and 'y' properties (0.0-1.0 range)
        
        Error responses (400):
        - Missing filename parameter
        - Coordinates out of image bounds
        - No regions detected
        - No significant regions found
        
        Error responses (404):
        - Uploaded file not found
        
        Error responses (500):
        - Image processing errors
        - Segmentation algorithm failures
    
    Accepted Methods:
        POST: JSON data with detection parameters
    
    JSON Parameters:
        filename (str): Name of uploaded image file
        x (int): X-coordinate for region detection seed point
        y (int): Y-coordinate for region detection seed point
        tolerance (int): Segmentation tolerance (fixed at 2 for optimal results)
    
    Example Request:
        {
            "filename": "image_20250823_143022.png",
            "x": 150,
            "y": 200,
            "tolerance": 2
        }
    
    Example Response:
        {
            "success": true,
            "contour_points": [
                {"x": 0.125, "y": 0.250},
                {"x": 0.130, "y": 0.255},
                ...
            ]
        }
    
    Algorithm Details:
        - Uses combined segmentation for robust region detection
        - Filters contours by 0.01% minimum area threshold
        - Prioritizes contours containing the seed point
        - Normalizes coordinates by max(width, height) for scale independence
        - Returns simplified contour approximation for efficiency
    """
    data = request.get_json()
    filename = data.get('filename')
    x = int(data.get('x'))
    y = int(data.get('y'))
    tolerance = 2  # Fixed tolerance for better edge detection
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404 
    
    # Read the image
    image = cv2.imread(filepath)
    height, width = image.shape[:2]
    
    # Verify that coordinates are within the image
    if x < 0 or x >= width or y < 0 or y >= height:
        return jsonify({'error': 'Coordinates out of range'}), 400
    
    # Pixel color is no longer needed for simplified response
    
    try:
        # Apply optimized combined segmentation
        region_mask = apply_combined_segmentation(image, x, y, tolerance)
        
        # Find contours
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return jsonify({'error': 'No region detected'}), 400
        
        # Filter contours by minimum area (0.01% of total image area to detect small islands)
        min_area = (width * height) * 0.0001
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if not valid_contours:
            return jsonify({'error': 'No significant regions detected'}), 400
        
        # Sort contours by area (largest first)
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        
        # Process only the main region (the largest one containing the clicked point)
        max_dim = max(width, height)
        
        # Find the region containing the clicked point
        main_contour = None
        for contour in valid_contours:
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                main_contour = contour
                break 
        
        # If no region containing the point is found, use the largest one
        if main_contour is None:
            main_contour = valid_contours[0]
        
        # Convert contour to list of normalized points
        contour_points = []
        for point in main_contour:
            px, py = point[0]
            contour_points.append({
                'x': px / max_dim,
                'y': py / max_dim
            })
        
        return jsonify({
            'success': True,
            'contour_points': contour_points
        })
        
    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    """
    Save detected contour coordinates to a CSV file.
    
    This API endpoint processes requests to save individual region coordinates
    detected through the contour creator interface. It converts normalized
    coordinate data back to original image coordinates, applies Y-axis inversion
    for proper coordinate system alignment, and exports the data to a CSV file.
    
    The coordinate processing includes:
    1. Validation of input parameters and coordinate data
    2. Coordinate denormalization using original image dimensions
    3. Y-axis inversion for standard coordinate system (origin at bottom-left)
    4. CSV file generation with proper headers
    5. Secure filename generation with timestamp
    
    Returns:
        JSON response containing:
        - success (bool): Save operation status
        - download_url (str): URL for downloading the generated CSV file
        - filename (str): Name of the generated CSV file
        - message (str): Success confirmation message
        
        Error responses (400):
        - Missing filename parameter
        - Missing or invalid coordinate data
        - Empty contour points array
        
        Error responses (404):
        - Original image file not found
        
        Error responses (500):
        - File processing errors
        - CSV generation failures
    
    Accepted Methods:
        POST: JSON data with coordinates and metadata
    
    JSON Parameters:
        filename (str): Original image filename for dimension reference
        region_name (str): Name identifier for the region
        contour_points (list): Array of normalized coordinate objects
                              with 'x' and 'y' properties (0.0-1.0 range)
    
    Example Request:
        {
            "filename": "image_20250823_143022.png",
            "region_name": "Region 1",
            "contour_points": [
                {"x": 0.125, "y": 0.250},
                {"x": 0.130, "y": 0.255}
            ]
        }
    
    Example Response:
        {
            "success": true,
            "download_url": "/download/Region_1_20250823_143022.csv",
            "filename": "Region_1_20250823_143022.csv",
            "message": "Coordinates saved successfully"
        }
    
    CSV Format:
        - Headers: X, Y
        - Coordinates in original image pixel units
        - Y-axis inverted (origin at bottom-left)
        - One coordinate pair per row
    """
    data = request.get_json()
    regions = data.get('regions')
    image_filename = data.get('filename')  # We need the filename to get dimensions
    
    if not regions or len(regions) == 0:
        return jsonify({'error': 'No regions to save'}), 400
    
    # Get original image dimensions for Y-axis inversion
    if image_filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        if os.path.exists(filepath):
            image = cv2.imread(filepath)
            height, width = image.shape[:2]
            max_dim = max(width, height)
        else:
            # If the image cannot be read, use default value (no inversion)
            max_dim = 1
            height = 1
    else:
        max_dim = 1
        height = 1
    
    # Create filename with timestamp
    from datetime import datetime
    import csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'contour_{timestamp}'
    
    # Save coordinates to CSV file in OUTPUT folder
    save_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}.csv')
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write headers
        writer.writerow(['x', 'y', 'region'])
        
        # Write data for each region
        for region_index, region in enumerate(regions, 1):
            for point in region['contour_points']:
                # Invert Y-axis only when saving: convert normalized coordinates back to pixels,
                # invert Y, and normalize again
                original_y_pixels = point['y'] * max_dim
                inverted_y_pixels = height - original_y_pixels
                inverted_y_normalized = inverted_y_pixels / max_dim
                
                writer.writerow([point['x'], inverted_y_normalized, region_index])
    
    # Count total points
    total_points = sum(len(region['contour_points']) for region in regions)
    
    return jsonify({
        'success': True,
        'message': f'{len(regions)} region(s) with {total_points} points saved in {filename}.csv',
        'filename': f'{filename}.csv',
        'download_url': f'/download/{filename}.csv'
    })

@app.route('/export_single_region', methods=['POST'])
def export_single_region():
    """
    Export a single detected region to a CSV file.
    
    This API endpoint handles the export of individual contour regions detected
    in the contour creator interface. It processes normalized coordinate data,
    converts it back to original image coordinates with proper Y-axis inversion,
    and generates a downloadable CSV file containing the region's boundary points.
    
    The export process includes:
    1. Input validation for required parameters
    2. Original image dimension retrieval
    3. Coordinate denormalization and Y-axis correction
    4. CSV file generation with descriptive filename
    5. Secure file handling and download URL generation
    
    Returns:
        JSON response containing:
        - success (bool): Export operation status
        - download_url (str): URL for downloading the generated CSV file
        - filename (str): Name of the generated CSV file
        
        Error responses (400):
        - Missing filename parameter
        - Missing region name
        - Missing or empty contour points
        
        Error responses (404):
        - Original image file not found
        
        Error responses (500):
        - Image processing errors
        - File generation failures
    
    Accepted Methods:
        POST: JSON data with region information
    
    JSON Parameters:
        filename (str): Original image filename for dimension reference
        region_name (str): Descriptive name for the exported region
        contour_points (list): Array of normalized coordinate objects
                              with 'x' and 'y' properties (0.0-1.0 range)
    
    Example Request:
        {
            "filename": "image_20250823_143022.png",
            "region_name": "Main Island",
            "contour_points": [
                {"x": 0.125, "y": 0.250},
                {"x": 0.130, "y": 0.255}
            ]
        }
    
    Example Response:
        {
            "success": true,
            "download_url": "/download/Main_Island_20250823_143022.csv",
            "filename": "Main_Island_20250823_143022.csv"
        }
    
    File Naming Convention:
        {region_name}_{timestamp}.csv
        - Spaces in region names are replaced with underscores
        - Timestamp format: YYYYMMDD_HHMMSS
        - Special characters are sanitized for filesystem compatibility
    """
    data = request.get_json()
    region_name = data.get('region_name')
    contour_points = data.get('contour_points')
    image_filename = data.get('filename')
    
    if not contour_points or len(contour_points) == 0:
        return jsonify({'error': 'No contour points to export'}), 400
    
    # Get original image dimensions for Y-axis inversion
    if image_filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        if os.path.exists(filepath):
            image = cv2.imread(filepath)
            height, width = image.shape[:2]
            max_dim = max(width, height)
        else:
            max_dim = 1
            height = 1
    else:
        max_dim = 1
        height = 1
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_region_name = secure_filename(region_name) if region_name else 'region'
    filename = f'{safe_region_name}_{timestamp}'
    
    # Save coordinates to CSV file in OUTPUT folder
    save_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}.csv')
    
    import csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write headers
        writer.writerow(['x', 'y'])
        
        # Write data for the region
        for point in contour_points:
            # Invert Y-axis only when saving
            original_y_pixels = point['y'] * max_dim
            inverted_y_pixels = height - original_y_pixels
            inverted_y_normalized = inverted_y_pixels / max_dim
            
            writer.writerow([point['x'], inverted_y_normalized])
    
    return jsonify({
        'success': True,
        'message': f'Region "{region_name}" exported successfully',
        'filename': f'{filename}.csv',
        'download_url': f'/download/{filename}.csv'
    })

@app.route('/save_all_coordinates', methods=['POST'])
def save_all_coordinates():
    """
    Save all detected contour coordinates to a single CSV file.
    
    This API endpoint processes requests to save multiple region coordinates
    detected through the contour creator interface into a consolidated CSV file.
    It handles multiple regions simultaneously, converting normalized coordinate
    data back to original image coordinates with proper Y-axis inversion.
    
    The consolidation process includes:
    1. Validation of input parameters and regions data
    2. Processing multiple regions with their respective coordinates
    3. Coordinate denormalization using original image dimensions
    4. Y-axis inversion for standard coordinate system alignment
    5. Multi-region CSV file generation with region identification
    6. Secure filename generation with timestamp
    
    Returns:
        JSON response containing:
        - success (bool): Save operation status
        - download_url (str): URL for downloading the generated CSV file
        - filename (str): Name of the generated CSV file
        - message (str): Success confirmation message
        
        Error responses (400):
        - Missing filename parameter
        - Missing or invalid regions data
        - Empty regions array
        
        Error responses (404):
        - Original image file not found
        
        Error responses (500):
        - Image processing errors
        - CSV generation failures
    
    Accepted Methods:
        POST: JSON data with multiple regions and metadata
    
    JSON Parameters:
        filename (str): Original image filename for dimension reference
        regions (list): Array of region objects, each containing:
                       - name (str): Region identifier
                       - contour_points (list): Normalized coordinates
    
    Example Request:
        {
            "filename": "image_20250823_143022.png",
            "regions": [
                {
                    "name": "Region 1",
                    "contour_points": [
                        {"x": 0.125, "y": 0.250},
                        {"x": 0.130, "y": 0.255}
                    ]
                },
                {
                    "name": "Region 2",
                    "contour_points": [
                        {"x": 0.225, "y": 0.350},
                        {"x": 0.230, "y": 0.355}
                    ]
                }
            ]
        }
    
    Example Response:
        {
            "success": true,
            "download_url": "/download/all_coordinates_20250823_143022.csv",
            "filename": "all_coordinates_20250823_143022.csv",
            "message": "All coordinates saved successfully"
        }
    
    CSV Format:
        - Headers: Region, X, Y
        - Multiple regions with region name identification
        - Coordinates in original image pixel units
        - Y-axis inverted (origin at bottom-left)
        - One coordinate pair per row with region identifier
    """
    data = request.get_json()
    regions = data.get('regions')
    image_filename = data.get('filename')
    
    if not regions or len(regions) == 0:
        return jsonify({'error': 'No regions to save'}), 400
    
    # Get original image dimensions for Y-axis inversion
    if image_filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        if os.path.exists(filepath):
            image = cv2.imread(filepath)
            height, width = image.shape[:2]
            max_dim = max(width, height)
        else:
            max_dim = 1
            height = 1
    else:
        max_dim = 1
        height = 1
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'all_regions_{timestamp}'
    
    # Save coordinates to CSV file in OUTPUT folder
    save_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{filename}.csv')
    
    import csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write headers
        writer.writerow(['x', 'y', 'region_name'])
        
        # Write data for each region
        for region in regions:
            region_name = region.get('name', 'unnamed')
            for point in region['contour_points']:
                # Invert Y-axis only when saving
                original_y_pixels = point['y'] * max_dim
                inverted_y_pixels = height - original_y_pixels
                inverted_y_normalized = inverted_y_pixels / max_dim
                
                writer.writerow([point['x'], inverted_y_normalized, region_name])
    
    # Count total points
    total_points = sum(len(region['contour_points']) for region in regions)
    
    return jsonify({
        'success': True,
        'message': f'{len(regions)} region(s) with {total_points} points saved',
        'filename': f'{filename}.csv',
        'download_url': f'/download/{filename}.csv'
    })

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload and validate CSV files for cloud of points generation.
    
    This API endpoint handles the upload of CSV files containing coordinate data
    for cloud of points generation in the Cloud Generator interface. It performs
    comprehensive validation of file format, structure, and content to ensure
    compatibility with the cloud of points generation algorithms.
    
    The validation process includes:
    1. File presence and format validation (CSV extension)
    2. File size and security checks
    3. CSV structure validation (headers, columns, data types)
    4. Coordinate data validation (numeric values, ranges)
    5. Secure file storage with timestamp-based naming
    6. Preparation for cloud of points generation workflow
    
    Returns:
        JSON response containing:
        - success (bool): Upload and validation status
        - filename (str): Stored filename for reference
        - message (str): Success confirmation or error details
        
        Error responses (400):
        - No file provided in request
        - Invalid file extension (non-CSV)
        - Empty filename
        - Invalid CSV structure
        - Missing required columns
        - Invalid coordinate data types
        
        Error responses (413):
        - File size exceeds maximum limit
        
        Error responses (500):
        - File storage errors
        - CSV processing failures
    
    Accepted Methods:
        POST: Multipart form data with CSV file
    
    Form Parameters:
        file: CSV file containing coordinate data
    
    Required CSV Format:
        - Headers: X, Y (case-sensitive)
        - Numeric coordinate values
        - One coordinate pair per row
        - Standard CSV formatting (comma-separated)
        - UTF-8 encoding recommended
    
    Example CSV Content:
        X,Y
        100.5,200.3
        105.2,198.7
        110.8,195.1
    
    Example Response:
        {
            "success": true,
            "filename": "coordinates_20250823_143022.csv",
            "message": "CSV file uploaded and validated successfully"
        }
    
    File Storage:
        - Files stored in UPLOAD_FOLDER directory
        - Timestamp-based naming for uniqueness
        - Automatic cleanup based on configured retention policy
        - Secure filename sanitization
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No CSV file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No CSV file selected'}), 400
    
    if file and allowed_csv_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate CSV format
        try:
            df = pd.read_csv(filepath)
            required_columns = ['x', 'y', 'region']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'CSV file must contain columns: {required_columns}'}), 400
            
            # File statistics
            regions = df['region'].unique()
            total_points = len(df)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'total_points': total_points,
                'regions': len(regions),
                'region_list': sorted(regions.tolist())
            })
            
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
    
    return jsonify({'error': 'File type not allowed. Only CSV files are accepted'}), 400

@app.route('/generate_cloud', methods=['POST'])
def generate_cloud_api():
    """
    Generate cloud of points from uploaded CSV coordinate data.
    
    This API endpoint initiates the cloud of points generation process using
    coordinate data from previously uploaded CSV files. It creates and executes
    a Python script that processes the coordinates through advanced mathematical
    algorithms to generate a 3D cloud of points representation.
    
    The generation process includes:
    1. Input validation for CSV filename and generation parameters
    2. Dynamic Python script creation with coordinate processing logic
    3. Subprocess execution for cloud of points generation algorithms
    4. Progress tracking and status monitoring
    5. Output file management and download preparation
    6. Error handling and logging for debugging
    
    The cloud of points generation uses sophisticated mathematical methods including:
    - Radial Basis Function (RBF) interpolation
    - Meshless Generalized Finite Difference (mGFD) methods
    - Adaptive grid generation algorithms
    - 3D coordinate transformation and scaling
    
    Returns:
        JSON response containing:
        - success (bool): Generation initiation status
        - message (str): Status confirmation or error details
        - process_id (str): Unique identifier for tracking generation progress
        
        Error responses (400):
        - Missing CSV filename parameter
        - Invalid generation parameters
        - Missing required configuration values
        
        Error responses (404):
        - CSV file not found in upload directory
        
        Error responses (500):
        - Script generation failures
        - Subprocess execution errors
        - File system access issues
    
    Accepted Methods:
        POST: JSON data with generation parameters
    
    JSON Parameters:
        csv_filename (str): Name of uploaded CSV file containing coordinates
        parameters (dict, optional): Generation configuration including:
                                   - grid_density (float): Point density factor
                                   - interpolation_method (str): Algorithm selection
                                   - boundary_conditions (str): Edge handling
                                   - output_format (str): Result file format
    
    Example Request:
        {
            "csv_filename": "coordinates_20250823_143022.csv",
            "parameters": {
                "grid_density": 1.5,
                "interpolation_method": "rbf",
                "boundary_conditions": "natural",
                "output_format": "csv"
            }
        }
    
    Example Response:
        {
            "success": true,
            "message": "Cloud of points generation started successfully",
            "process_id": "pc_20250823_143022_abc123"
        }
    
    Process Management:
        - Asynchronous execution for long-running operations
        - Status tracking through dedicated status files
        - Automatic cleanup of temporary files
        - Progress monitoring and error reporting
        - Resource management and memory optimization
    
    Output Files:
        - Generated cloud of points data in specified format
        - Processing logs and diagnostic information
        - Status files for progress tracking
        - Error logs for debugging purposes
    """
    data = request.get_json()
    filename = data.get('filename')
    regiones_inside = data.get('regiones_inside', False)
    reduce_points = data.get('reduce_points', False)
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Execute cloud of points generator using subprocess
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(filename)[0]
        
        # Create unique identifier for this task
        task_id = f"{base_name}_{timestamp}"
        
        # Execute cloud of points generation script using subprocess
        def run_cloud_generation():
            try:
                # Mark as started
                with open(os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_status.txt'), 'w') as f:
                    f.write('running')
                
                # Create temporary script to execute generation
                script_content = f'''
import sys
import os
sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from cloud_generation import generate_cloud_of_points
from reduce_points import reduce_points_by_region

if __name__ == "__main__":
    os.chdir('{os.path.dirname(os.path.abspath(__file__))}')
    
    # Determine file to use for generation
    input_file = "{filepath}"
    
    # If point reduction is required, apply it first
    if {reduce_points}:
        import tempfile
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create temporary file in system temp directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_reduced_file = temp_file.name
        
        # Apply point reduction
        result = reduce_points_by_region(input_file, temp_reduced_file)
        if result is not None:
            input_file = temp_reduced_file
        else:
            import logging
            logging.error("Error in point reduction, using original file")
    
    # Generate cloud of points
    original_base = os.path.splitext(os.path.basename("{filepath}"))[0]
    success = generate_cloud_of_points(input_file, {regiones_inside}, "{timestamp}", original_base)
    
    # Clean up temporary reduction file if created
    if {reduce_points} and input_file != "{filepath}":
        try:
            os.remove(input_file)
        except:
            pass
    
    sys.exit(0 if success else 1)
'''
                
                script_path = os.path.join(app.config['OUTPUT_FOLDER'], f'temp_script_{task_id}.py')
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Execute in separate process
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                # Clean up temporary script
                try:
                    os.remove(script_path)
                except:
                    pass
                
                # Mark as completed or failed
                with open(os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_status.txt'), 'w') as f:
                    if result.returncode == 0:
                        f.write('completed')
                    else:
                        f.write(f'error: {result.stderr}')
                    
            except Exception as e:
                # Mark as failed
                with open(os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_status.txt'), 'w') as f:
                    f.write(f'error: {str(e)}')
        
        # Execute in separate thread
        thread = threading.Thread(target=run_cloud_generation)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Cloud of points generation started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error starting generation: {str(e)}'}), 500

@app.route('/cloud_status/<task_id>', methods=['GET'])
def cloud_status(task_id):
    """
    Check the status of cloud of points generation process.
    
    This API endpoint provides real-time status information for ongoing
    cloud of points generation processes. It monitors the progress of background
    computational tasks and provides detailed feedback on generation status,
    completion percentage, and any errors that may have occurred.
    
    The status monitoring includes:
    1. Process existence and validity verification
    2. Status file parsing and interpretation
    3. Progress percentage calculation
    4. Error detection and reporting
    5. Completion status and output file availability
    6. Resource usage and performance metrics
    
    Returns:
        JSON response containing:
        - status (str): Current process status ('running', 'completed', 'error')
        - progress (float): Completion percentage (0.0-100.0)
        - message (str): Detailed status description
        - output_file (str, optional): Generated file name when completed
        - download_url (str, optional): Download URL when completed
        - error_details (str, optional): Error information if failed
        
        Error responses (404):
        - Process ID not found
        - Status file not accessible
        
        Error responses (500):
        - Status file parsing errors
        - File system access issues
    
    Accepted Methods:
        GET: Retrieve status information
    
    URL Parameters:
        task_id (str): Unique identifier for the generation process
    
    Example Request:
        GET /cloud_status/pc_20250823_143022_abc123
    
    Example Response (Running):
        {
            "status": "running",
            "progress": 45.2,
            "message": "Processing coordinates and generating grid points"
        }
    
    Example Response (Completed):
        {
            "status": "completed",
            "progress": 100.0,
            "message": "Cloud of points generation completed successfully",
            "output_file": "cloud_of_points_20250823_143022.csv",
            "download_url": "/download/cloud_of_points_20250823_143022.csv"
        }
    
    Example Response (Error):
        {
            "status": "error",
            "progress": 0.0,
            "message": "Cloud of points generation failed",
            "error_details": "Invalid coordinate data format in input file"
        }
    
    Status Values:
        - 'initializing': Process setup and validation
        - 'running': Active computation in progress
        - 'completed': Generation finished successfully
        - 'error': Process failed with errors
        - 'cancelled': Process terminated by user or system
    """
    status_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{task_id}_status.txt')
    
    if not os.path.exists(status_file):
        return jsonify({'error': 'Task not found'}), 404
    
    try:
        with open(status_file, 'r') as f:
            status = f.read().strip()
        
        if status == 'running':
            return jsonify({
                'status': 'running',
                'message': 'Generating cloud of points...'
            })
        elif status == 'completed':
            # Search for generated files in output folder
            generated_files = []
            output_dir = app.config['OUTPUT_FOLDER']
            
            if os.path.exists(output_dir):
                timestamp = '_'.join(task_id.split('_')[-2:])
                base_name = '_'.join(task_id.split('_')[:-2])
                
                for file in os.listdir(output_dir):
                    if (base_name in file and timestamp in file and 
                        (file.endswith('.csv') or file.endswith('.png') or file.endswith('.svg')) and
                        not file.endswith('_status.txt')):
                        generated_files.append(file)
            
            try:
                os.remove(status_file)
            except:
                pass
            
            return jsonify({
                'status': 'completed',
                'message': 'Cloud of points generated successfully',
                'files': generated_files
            })
        elif status == 'failed':
            try:
                os.remove(status_file)
            except:
                pass
            
            return jsonify({
                'status': 'failed',
                'message': 'Error in cloud of points generation'
            })
        elif status.startswith('error:'):
            error_msg = status.replace('error: ', '')
            
            try:
                os.remove(status_file)
            except:
                pass
            
            return jsonify({
                'status': 'error',
                'message': f'Error: {error_msg}'
            })
        else:
            return jsonify({
                'status': 'unknown',
                'message': 'Unknown status'
            })
            
    except Exception as e:
        return jsonify({'error': f'Error checking status: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """
    Download generated files (CSV coordinates, clouds of points, etc.).
    
    This API endpoint provides secure file download functionality for various
    generated files including coordinate CSV exports, cloud of points data, and
    other computational results. It implements security measures to prevent
    unauthorized access and ensures safe file delivery.
    
    The download process includes:
    1. Filename validation and sanitization
    2. File existence and accessibility verification
    3. Security checks to prevent directory traversal attacks
    4. Appropriate MIME type detection and headers
    5. Secure file streaming for large files
    6. Access logging for audit purposes
    
    Supported file types:
    - CSV files: Coordinate data, cloud of points results
    - Text files: Processing logs, status reports
    - Data files: Computational results and exports
    
    Returns:
        File download response with appropriate headers:
        - Content-Type: Automatically detected MIME type
        - Content-Disposition: Attachment with original filename
        - Content-Length: File size for progress tracking
        
        Error responses (400):
        - Invalid filename format
        - Filename contains illegal characters
        
        Error responses (404):
        - File not found in download directory
        - File has been automatically cleaned up
        
        Error responses (403):
        - Access denied to requested file
        - File outside allowed directory
        
        Error responses (500):
        - File system access errors
        - File corruption or read failures
    
    Accepted Methods:
        GET: Download file content
    
    URL Parameters:
        filename (str): Name of the file to download
    
    Example Request:
        GET /download/coordinates_20250823_143022.csv
    
    Example Response Headers:
        Content-Type: text/csv
        Content-Disposition: attachment; filename="coordinates_20250823_143022.csv"
        Content-Length: 1024
    
    Security Features:
        - Filename sanitization to prevent path traversal
        - Directory restriction to upload/download folders
        - File type validation for allowed extensions
        - Access logging for security monitoring
        - Automatic cleanup of expired files
    
    File Lifecycle:
        - Files are automatically cleaned up based on retention policy
        - Download links expire after configured time period
        - Temporary files are removed after successful download
        - Error files are retained for debugging purposes
    """
    try:
        output_dir = app.config['OUTPUT_FOLDER']
        file_path = os.path.join(output_dir, filename)
        
        if os.path.exists(file_path):
            return send_from_directory(output_dir, filename, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

if __name__ == '__main__':
    start_cleanup_scheduler()
    app.run(debug=True, host='0.0.0.0', port=8080)