"""
app.py
-------

This script contains the backend logic for mGFD: CloudGenerator designed to:
- Handle file uploads and processing.
- Generate clouds of points and visualize them.
- Serve results as downloadable files.

Features:
1. Processes CSV files to create structured clouds of points with regional and boundary flags.
2. Provides an interactive web interface for file uploads, configuration, and visualization.
3. Automatically cleans up temporary and result files after a configurable delay.

Author: Gerardo Tinoco-Guerrero
Created: May 7th, 2024.
Last Modified: April 1st, 2025.

Dependencies:
- Flask: Web framework for routing and rendering.
- pandas: Data manipulation and CSV handling.
- numpy: Numerical computations.
- matplotlib: Plotting and visualization.
- shapely: Geometric calculations (e.g., boundaries and holes).
- dmsh: Adaptive mesh generation.
- werkzeug: Secure file handling.
- threading: Asynchronous file deletion.

Configuration:
- UPLOAD_FOLDER: Directory for storing uploaded files.
- OUTPUT_FOLDER: Directory for storing generated results.
- ALLOWED_EXTENSIONS: Allowed file types for uploads (images).
- ALLOWED_EXTENSIONS_D: Allowed file types for uploads (CSV for boundaries).
"""

# --- Imports ---
# Standard library imports
from datetime import datetime
from threading import Timer
import os

# Third-party library imports
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, flash
from shapely.geometry import Point, Polygon
from werkzeug.utils import secure_filename
from shapely.ops import unary_union
import pandas as pd
import csv

# Core importation
from core import (
    process_csv,
    generate_polygons,
    test_all_region_containments,
    remove_duplicate_containments,
    generate_clouds_for_all_regions,
    GraphCloud
)

# Logging imports
from utils import app_logger, log_error

# Plot imports
import matplotlib
matplotlib.use('Agg')                                                                                   # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

# --- Flask Configuration ---
app = Flask(__name__)
app.secret_key = os.urandom(24)                                                                         # Generate a random secret key
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                                                   # Absolute route for the script.
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'tmp', 'uploads')                                  # Directory for uploaded files
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'tmp', 'results')                                  # Directory for output files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}                                               # Allowed extensions for image uploads
app.config['ALLOWED_EXTENSIONS_D'] = {'csv'}                                                            # Allowed extensions for CSV uploads

# --- Environment Variables ---
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, 'tmp', 'matplotlib')                                # Ensure matplotlib uses a temporary directory

# --- Ensure Directories Exist ---
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok = True)                                               # Ensure the output folder exists.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)                                               # Ensure the upload folder exists.

'''
======================================
1. CONFIGURATION AND UTILITIES
======================================
'''

def allowed_file(filename):
    """
    Check if a file has an allowed extension for images.

    Parameters:
        filename (str):                                 Name of the file to check.

    Returns:
        bool:                                           True if the file extension is allowed, False otherwise.
    """

    # Check if the filename contains a dot and if its extension is in the allowed list.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']   # Check if the file has an allowed extension.

def allowed_file_D(filename):
    """
    Check if a file has an allowed extension for a specific category of files.

    Parameters:
        filename (str):                                 Name of the file to check.

    Returns:
        bool:                                           True if the file extension is allowed, False otherwise.
    """

    # Check if the filename contains a dot and if its extension is in the allowed list for category D.
    return '.' in filename and filename.rsplit('.', 1)[1].lower()\
            in app.config['ALLOWED_EXTENSIONS_D']                                                       # Check if the file has an allowed extension.

def delete_file(path, delay):
    """
    Schedules the deletion of a file after a specified delay.

    Parameters:
        path (str):                                     The file path to be deleted.
        delay (int or float):                           The delay in seconds before deleting the file.

    Returns:
        None                                            The function schedules a deletion but does not return any value.
    """

    def delayed_delete():
        """
            Deletes the file from the specified path after the delay.
            Logs the deletion status or any errors encountered.
        """

        try:
            if os.path.exists(path):                                                                    # Check if the file exists.
                os.remove(path)                                                                         # Remove the file from the system.
                app_logger.info(f"File deleted: {path}")                                                # Log successful deletion.
            else:                                                                                       # If the file does not exist.
                app_logger.warning(f"The file doesn't exists: {path}")                                  # Log a warning about missing file.
        
        except Exception as e:                                                                          # Handle any deletion errors.
            app_logger.error(f"Error while deleting the file {path}: {str(e)}")                         # Log the error details.

    timer = Timer(delay, delayed_delete)                                                                # Create a timer to delay file deletion.
    timer.start()                                                                                       # Start the timer.

@app.route('/download_csv', methods = ['POST'])
def download_csv():
    """
    Handles the creation and download of a CSV file containing point data.

    Methods:
        POST:                                           Receives JSON data, generates a CSV file, and returns its filename.

    Returns:
        dict:                                           A dictionary containing the generated CSV filename.
    """

    # Retrieve JSON data from the request.
    data = request.get_json()                                                                           # Get JSON data from the request body.
    points = data['points']                                                                             # Extract the list of points.
    
    # Generate the name of the file using a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                                                # Generate a timestamp with the current date and time.
    csv_filename = f'points_{timestamp}.csv'                                                            # Create a unique filename for the CSV.

    # Define the full path for the generated CSV file.
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)                                  # Construct the complete path for the output file.

    # Save the points data into a CSV file.
    with open(csv_path, mode = 'w', newline = '') as file:                                              # Open the file in write mode.
        cw = csv.writer(file)                                                                           # Create a CSV writer object.
        cw.writerow(['x', 'y', 'flag'])                                                                 # Write the header row.
        cw.writerows([[p['x'], p['y'], p['flag']] for p in points])                                     # Write the points data.

    # Schedule file deletion after 1 hour to free up space.
    delete_file(csv_path, 3600)                                                                         # Schedule automatic file deletion after 3600 seconds.

    # Return the generated filename as a response.
    return {'filename': csv_filename}                                                                   # Send the filename as a JSON response.

'''
======================================
2. FILE HANDLING AND SERVER ROUTES
======================================
'''

@app.route('/uploadI', methods = ['GET', 'POST'])
def upload_image():
    """
    Handles the upload of image files, saves them securely, and schedules deletion.

    Methods:
        GET:                                            Renders the file upload page.
        POST:                                           Processes the uploaded file, saves it, and redirects to the contour page.

    Returns:
        Response:                                       Renders different templates based on the request method and file status.
    """

    if request.method == 'POST':                                                                        # If the request method is POST.
        if 'file' not in request.files:                                                                 # If no file was uploaded.
            return redirect(request.url)                                                                # Redirect to the upload page.
        
        file = request.files['file']                                                                    # Retrieve the uploaded file.

        if file.filename == '':                                                                         # If no file was selected.
            return redirect(request.url)                                                                # Redirect to the upload page.
        
        if file and allowed_file(file.filename):                                                        # If the file exists and has a valid extension.
            filename = secure_filename(file.filename)                                                   # Secure the filename.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)                              # Construct the file path.
            file.save(filepath)                                                                         # Save the file to the specified location.
            delete_file(filepath, 3600)                                                                 # Schedule deletion of the file after 3600 seconds.

            return render_template('contour.html', filename = filename)                                 # Render the contour page with the uploaded filename.

    return render_template('uploadI.html')                                                              # Render the upload page for GET requests or failed uploads.

@app.route('/uploadC', methods = ['GET', 'POST'])
def upload_files():
    """
    Handles the upload of CSV files, processes the data, generates clouds of points, and provides visualization.

    Methods:
        GET:                                            Renders the file upload page.
        POST:                                           Validates and processes the uploaded CSV file, generates clouds of points, and renders results.

    Returns:
        Response:                                       Renders different templates based on request method and file status.
    """

    if request.method == 'POST':                                                                        # Check if the request is POST
        # Check if a file has been uploaded
        if 'points_file' not in request.files:                                                          # Check if a file exists in the request
            app_logger.warning("No file found in the request")                                          # Log a warning if no file is found
            flash("Please select a file to upload", "error")                                            # Show error message to user
            return redirect(request.url)                                                                # Redirect user to upload page

        file = request.files['points_file']                                                             # Get file from request

        if file.filename == '':                                                                         # Check if a file was selected
            app_logger.warning("Empty filename")                                                        # Log warning if filename is empty
            flash("No file has been selected", "error")                                                 # Show error message to user
            return redirect(request.url)                                                                # Redirect user to upload page

        if file and allowed_file_D(file.filename):                                                      # Check if file exists and has allowed extension
            filename = secure_filename(file.filename)                                                   # Ensure filename is secure
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                             # Build complete path to save file
            
            try:
                file.save(file_path)                                                                    # Try to save file on server
                app_logger.info(f"File saved successfully: {filename}")                                 # Log successful operation
            except Exception as e:
                error_msg = log_error(e, "Error saving the file")                                       # Log any errors during save
                flash("Error saving the file. Please try again.", "error")                              # Inform user of error
                return redirect(request.url)                                                            # Redirect user to upload page

            # Read and validate CSV file
            try:
                app_logger.info(f"Starting file processing: {filename}")                                # Log start of processing
                regions = process_csv(file_path)                                                        # Process CSV file to get regions
                polygons = generate_polygons(regions)                                                   # Generate polygons from regions
                containment_results = test_all_region_containments(polygons)                            # Test containment relationships between regions
                depurated_results = remove_duplicate_containments(containment_results)                  # Remove duplicate containment results
                app_logger.info("Data processing completed successfully")                               # Log successful completion of processing

                try:
                    # Retrieve form parameters
                    num = int(request.form.get('num', 3))                                               # Get number of points per region
                    rand = int(request.form.get('rand', 1))                                             # Get randomness factor
                    mod = int(request.form.get('mod', 1))                                               # Get density modifier
                    gen = int(request.form.get('gen', 0))                                               # Get generation type
                except ValueError as e:
                    error_msg = log_error(e, "Error in input parameters")                               # Log errors in parameters
                    flash("Parameter values are invalid. Please verify the entered data.", "error")     # Inform user of error
                    return redirect(request.url)                                                        # Redirect user to upload page

                try:
                    app_logger.info("Starting cloud of points generation")                              # Log start of cloud generation
                    all_clouds = generate_clouds_for_all_regions(polygons, depurated_results, num, rand, mod, gen)
                                                                                                        # Generate point clouds
                    app_logger.info("Cloud of points generated successfully")                           # Log successful generation
                except Exception as e:
                    error_msg = log_error(e, "Error in cloud generation")                               # Log errors in generation
                    flash("Error generating the cloud of points. Please verify the data and try again.", "error")
                    return redirect(request.url)                                                        # Redirect user to upload page

                try:
                    # Generate unique names for output files
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                                # Create unique timestamp
                    image_name = f'plot_{timestamp}.png'                                                # Name for PNG image
                    eps_name = f'plot_{timestamp}.eps'                                                  # Name for EPS image
                    p_csv_name = f'cloud_{timestamp}.csv'                                               # Name for CSV file

                    # Generate visualizations and save data
                    app_logger.info("Generating visualizations and saving data")                        # Log start of visualization
                    GraphCloud(all_clouds, folder=app.config['OUTPUT_FOLDER'], image_name=image_name, eps_name=eps_name)
                                                                                                        # Create graphs

                    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name)                    # Build path for CSV file
                    pd.DataFrame(all_clouds, columns=["x", "y", "region", "boundary_flag"]).to_csv(csv_path, index=False)
                                                                                                        # Save data

                    # Schedule file deletion
                    delete_file(file_path, 3600)                                                        # Delete original file after 1 hour
                    delete_file(os.path.join(app.config['OUTPUT_FOLDER'], image_name), 3600)            # Delete PNG image
                    delete_file(os.path.join(app.config['OUTPUT_FOLDER'], eps_name), 3600)              # Delete EPS image
                    delete_file(csv_path, 3600)                                                         # Delete CSV file

                    app_logger.info("Process completed successfully")                                   # Log successful completion
                    flash("Cloud of points has been generated successfully", "success")                 # Inform user of success

                    return render_template(                                                             # Render template with results
                        'cloud.html',
                        image_name=image_name,
                        eps_name=eps_name,
                        p_csv_name=p_csv_name,
                        tables=[pd.DataFrame(all_clouds, columns=["x", "y", "region", "boundary_flag"]).to_html(classes='data')],
                        titles=['na', 'Cloud Data']
                    )

                except Exception as e:
                    error_msg = log_error(e, "Error generating output files")                           # Log errors in file generation
                    flash("Error generating output files. Please try again.", "error")                  # Inform user of error
                    return redirect(request.url)                                                        # Redirect user to upload page

            except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
                error_msg = log_error(e, "Error processing the file")                                   # Log errors in file processing
                flash(str(e), "error")                                                                  # Show error message to user
                return redirect(request.url)                                                            # Redirect user to upload page
            except Exception as e:
                error_msg = log_error(e, "Unexpected error in processing")                              # Log unexpected errors
                flash("An unexpected error has occurred. Please try again.", "error")                   # Inform user of error
                return redirect(request.url)                                                            # Redirect user to upload page
        else:
            app_logger.warning(f"File type not allowed: {file.filename}")                               # Log warning for not allowed file type
            flash("File type not allowed. Please use a CSV file.", "error")                             # Inform user of error
            return redirect(request.url)                                                                # Redirect user to upload page

    return render_template('uploadC.html')

@app.route('/modify', methods = ['GET', 'POST'])
def modify():
    """
    Handles file uploads for modification purposes and schedules file deletion.

    Methods:
        GET:                                            Renders the file upload page.
        POST:                                           Processes the uploaded file, saves it, and redirects to the modification page.

    Returns:
        Response:                                       Renders different templates based on request method and file status.
    """

    if request.method == 'POST':                                                                        # If the request method is POST.
        if 'file' not in request.files:                                                                 # If no file was uploaded.
            return redirect(request.url)                                                                # Redirect to the upload page.

        file = request.files['file']                                                                    # Retrieve the uploaded file.

        if file.filename == '':                                                                         # If no file was selected.
            return redirect(request.url)                                                                # Redirect to the upload page.
        
        if file and allowed_file_D(file.filename):                                                      # If the file exists and has a valid extension.
            filename = secure_filename(file.filename)                                                   # Secure the filename.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)                              # Construct the file path.
            file.save(filepath)                                                                         # Save the file to the specified location.
            delete_file(filepath, 3600)                                                                 # Schedule deletion of the file after 3600 seconds.

            return render_template('modify.html', filename = filename)                                  # Render the modification page with the uploaded filename.

    return render_template('uploadM.html')                                                              # Render the upload page for GET requests or failed uploads.

@app.route('/CloudGen/results/<path:filename>')
def static_files(filename):
    """
    Serves static files from the output directory.

    Parameters:
        filename (str):                                 The name of the file to be served.

    Returns:
        Response:                                       The requested file as an HTTP response.
    """

    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)                                   # Serve the requested file from the output directory.

@app.route('/CloudGen/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serves uploaded files from the upload directory.

    Parameters:
        filename (str):                                 The name of the uploaded file to be served.

    Returns:
        Response:                                       The requested file as an HTTP response.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)                                   # Serve the requested file from the upload directory.

'''
======================================
7. SERVER ROUTES AND WEB PAGES
======================================
'''

@app.route('/')
def index():
    """
    Renders the main index page of the web application.

    Returns:
        Response:                                       An HTTP response that renders the 'index.html' template.
    """

    return render_template('index.html')                                                                # Render and return the index page.

@app.route('/about')
def about():
    """
    Renders the 'About' page of the web application.

    Returns:
        Response:                                       An HTTP response that renders the 'about.html' template.
    """

    return render_template('about.html')                                                                # Render and return the 'About' page.

@app.route('/howto')
def howto():
    """
    Renders the 'How to' page of the web application.

    Returns:
        Response:                                       An HTTP response that renders the 'howto.html' template.
    """

    return render_template('howto.html')                                                                # Render and return the 'How to' page.

if __name__ == '__main__':
    app.run(debug = True)