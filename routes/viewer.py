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

from cloud_modules.visualization import create_visualization
from cloud_modules.data_processing import load_cloud_data

from . import viewer_bp


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS
    
@viewer_bp.route('/viewer')
def viewer():
    """
    Render the cloud/contour viewer page.
    
    Returns:
        str: Rendered HTML template for the viewer interface
    """
    return render_template('viewer.html')


@viewer_bp.route('/upload_viewer', methods=['POST'])
def upload_viewer():
    """
    Handle CSV file upload and generate visualization for the viewer.
    
    Returns:
        JSON response with image URLs or error message
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
            
        if file and allowed_csv_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"viewer_{timestamp}_{filename}"
            input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            output_base = os.path.splitext(input_path)[0]
            
            file.save(input_path)
            
            # Load cloud data
            points, regions, classifications = load_cloud_data(input_path)
            
            if not points:
                return jsonify({'success': False, 'error': 'Failed to load points from CSV or file is empty'})
                
            # Retrieve points data for interactive editing
            return jsonify({
                'success': True,
                'points': points,
                'regions': regions,
                'classifications': classifications,
                'total_points': len(points)
            })
            
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})
            
    except Exception as e:
        current_app.logger.error(f"Error in upload_viewer: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


