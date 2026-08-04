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

from cloud_modules.visualization import render_neighbors_graph
from analysis_modules.neighbors import compute_neighbors_from_file
from cloud_modules.data_processing import load_cloud_data

from . import neighbors_bp


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS
    
@neighbors_bp.route('/neighbors')
def neighbors():
    """
    Render the neighbors calculator page.
    
    Returns:
        str: Rendered HTML template for the neighbors interface
    """
    return render_template('neighbors.html')


@neighbors_bp.route('/upload_neighbors', methods=['POST'])
def upload_neighbors():
    """
    Handle CSV file upload and calculate neighbors.
    
    Returns:
        JSON response with neighbors CSV URL and visualization or error message
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
            unique_filename = f"neighbors_{timestamp}_{filename}"
            input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            output_base = os.path.splitext(input_path)[0]
            
            nvec_str = request.form.get('nvec', '9')
            try:
                nvec = int(nvec_str)
            except ValueError:
                nvec = 9
            
            file.save(input_path)
            
            # Calculate neighbors
            neighbors_indices = compute_neighbors_from_file(input_path, nvec=nvec)
            
            if neighbors_indices is None:
                return jsonify({'success': False, 'error': 'Failed to compute neighbors'})
            
            # Save neighbors to CSV
            neighbors_csv_path = f"{output_base}_neighbors.csv"
            with open(neighbors_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header: point_idx, neighbor_1, neighbor_2, ...
                header = ['point_idx'] + [f'neighbor_{i+1}' for i in range(neighbors_indices.shape[1])]
                writer.writerow(header)
                for i, row in enumerate(neighbors_indices):
                    writer.writerow([i] + row.tolist())
            
            # Load cloud data for statistics
            points, regions, classifications = load_cloud_data(input_path)
            
            # Calculate statistics
            total_points = len(points)
            unique_regions = len(np.unique(regions))
            
            # Calculate neighbors stats
            neighbors_found = np.sum(neighbors_indices != -1)
            total_possible_neighbors = neighbors_indices.size
            avg_neighbors = neighbors_found / total_points
            
            filename_base = os.path.basename(output_base)
            neighbors_csv_url = url_for('contour.uploaded_file', filename=f"{filename_base}_neighbors.csv")
            
            # Generate Graph Visualization
            graph_generated = render_neighbors_graph(points, neighbors_indices, regions, output_base)
            
            response_data = {
                'success': True,
                'neighbors_csv_url': neighbors_csv_url,
                'points': points.tolist() if isinstance(points, np.ndarray) else points,
                'regions': regions.tolist() if isinstance(regions, np.ndarray) else regions,
                'classifications': classifications.tolist() if isinstance(classifications, np.ndarray) else classifications,
                'neighbors_indices': neighbors_indices.tolist() if isinstance(neighbors_indices, np.ndarray) else neighbors_indices,
                'stats': {
                    'total_points': total_points,
                    'total_regions': unique_regions,
                    'avg_neighbors': round(avg_neighbors, 2),
                    'max_neighbors': neighbors_indices.shape[1]
                }
            }
            
            if graph_generated:
                response_data['png_url'] = url_for('contour.uploaded_file', filename=f"{filename_base}.png")
                response_data['svg_url'] = url_for('contour.uploaded_file', filename=f"{filename_base}.svg")
            
            return jsonify(response_data)
                
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'})
            
    except Exception as e:
        current_app.logger.error(f"Error in upload_neighbors: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


