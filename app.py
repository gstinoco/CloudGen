"""
mGFD CloudGenerator - Advanced Point Cloud Generation Tool

This Flask web application provides an advanced tool for generating point clouds
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

2. Cloud Generator: Point cloud generation system that processes CSV files
   containing contour coordinates and generates optimized point clouds using
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
- Asynchronous point cloud generation with real-time progress tracking
- Multi-region support for complex geometries with interior holes
- Downloadable results in multiple formats with visualization previews

Technical Implementation:
- Flask web framework with Jinja2 templating and secure file handling
- OpenCV for computer vision, image processing, and segmentation algorithms
- NumPy for numerical computations and array operations
- Standard libraries for data manipulation, CSV handling, and coordinate processing
- Shapely for geometric operations and polygon validation
- Threading for background tasks, file cleanup, and progress tracking
- Rotating file handlers for production logging with configurable levels

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
Last Modification: March, 2026

Dependencies:
- Flask >= 2.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Werkzeug >= 2.0.0
- Shapely >= 1.8.0
- Matplotlib >= 3.10.0
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import threading
import tempfile
import logging
import time
import csv
import cv2
import os
import re

# Import project-specific modules
from cloud_modules.generator import generate_cloud_regular, generate_cloud_natural
from cloud_modules.reduction import reduce_points_by_region
from contour_modules import detection as contour_detection
from cloud_modules.data_processing import load_cloud_data
from cloud_modules.visualization import create_visualization, render_neighbors_graph
from analysis_modules.neighbors import compute_neighbors_from_file

from flask_babel import Babel

app = Flask(__name__, static_url_path='/static')

def get_locale():
    return request.cookies.get('lang', 'en')

babel = Babel(app, locale_selector=get_locale)

# Absolute routes for uploads and outputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('VERCEL'):
    # Vercel environment: Use /tmp for writable directories
    LOG_DIR = os.path.join(tempfile.gettempdir(), 'logs')
    app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(tempfile.gettempdir(), 'output')
else:
    # Local or Custom Server environment
    # Use environment variables if provided (useful for production deployments)
    LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
    app.config['OUTPUT_FOLDER'] = os.environ.get('OUTPUT_FOLDER', os.path.join(BASE_DIR, 'output'))

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'mGFD_CloudGenerator_2025'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
if not app.debug:
    if os.environ.get('VERCEL'):
        # Stream logging for Vercel
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.INFO)
    else:
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, 'mGFD_CloudGenerator.log'),
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
else:
    if os.environ.get('VERCEL'):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        stream_handler.setLevel(logging.DEBUG)
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.DEBUG)
    else:
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, 'mGFD_CloudGenerator_debug.log'),
            maxBytes=10240000,
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.DEBUG)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.DEBUG)

# Log startup info
app.logger.info(f"App started. BASE_DIR: {BASE_DIR}")
app.logger.info(f"LOG_DIR: {LOG_DIR}")
app.logger.info(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")

# Configuration
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

def calculate_export_scale(w, h, config=None):
    """
    Calculate scaling factors based on image dimensions and configuration.
    
    Determines the appropriate scaling factors for X and Y axes based on the
    provided configuration method (preserve aspect ratio, stretch, or custom).
    
    Args:
        w (int): Image width
        h (int): Image height
        config (dict, optional): Scaling configuration. Defaults to None.
            - method: 'preserve_aspect_ratio' (default), 'stretch', 'custom'
            - custom_x: float (for custom method)
            - custom_y: float (for custom method)
            
    Returns:
        tuple: (scale_x, scale_y) floats
    """
    if config is None:
        config = {}
        
    method = config.get('method', 'preserve_aspect_ratio')
    
    if method == 'custom':
        # Use custom scaling factors provided in config
        return float(config.get('custom_x', 1.0)), float(config.get('custom_y', 1.0))
        
    if method == 'stretch':
        # Map [0,1] input directly to [0,1] output (stretch to fill square)
        return 1.0, 1.0
        
    # Default: preserve_aspect_ratio
    # Scales the largest dimension to 1.0 and the other proportionally
    max_dim = max(w, h)
    if max_dim > 0:
        return w / max_dim, h / max_dim
        
    return 1.0, 1.0

def transform_coordinate(point, scale_x, scale_y):
    """
    Transform a single coordinate point with scaling and safety clamping.
    
    Applies scaling factors, inverts Y-axis (image to Cartesian), and strictly
    clamps result to [0,1] range to prevent out-of-bounds errors.
    
    Args:
        point (dict): Point object with 'x' and 'y' keys (normalized [0,1])
        scale_x (float): Scaling factor for X axis
        scale_y (float): Scaling factor for Y axis
        
    Returns:
        tuple: (final_x, inverted_y) floats in range [0,1]
    """
    # Apply scaling
    final_x = point['x'] * scale_x
    # Invert Y coordinate (image Y=0 is top, Cartesian Y=0 is bottom)
    inverted_y = (1.0 - point['y']) * scale_y
    
    # Strictly clamp to [0,1] range
    final_x = max(0.0, min(1.0, final_x))
    inverted_y = max(0.0, min(1.0, inverted_y))
    
    return final_x, inverted_y

# Main application routes

from routes import main_bp, contour_bp, cloud_bp, viewer_bp, neighbors_bp

app.register_blueprint(main_bp)
app.register_blueprint(contour_bp)
app.register_blueprint(cloud_bp)
app.register_blueprint(viewer_bp)
app.register_blueprint(neighbors_bp)

# Background file cleanup task
def cleanup_old_files():
    """
    Background task to clean up files older than 24 hours in upload and output directories.
    Runs periodically to prevent disk space exhaustion.
    """
    import time, os, logging
    while True:
        try:
            current_time = time.time()
            # 24 hours in seconds
            max_age = 24 * 3600
            
            # Use current_app might not be available here, but we can use app.config directly since it's global in app.py
            directories = [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]
            
            for directory in directories:
                if not os.path.exists(directory):
                    continue
                    
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > max_age:
                            os.remove(filepath)
                            app.logger.info(f"Cleaned up old file: {filepath}")
        except Exception as e:
            app.logger.error(f"Error in cleanup task: {str(e)}")
            
        # Run every 6 hours
        time.sleep(6 * 3600)

if __name__ == '__main__':
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    app.logger.info("Started background file cleanup task")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
