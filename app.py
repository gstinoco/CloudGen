"""
app.py
-------

This script contains the backend logic for mGFD: CloudGenerator designed to:
- Handle file uploads and processing.
- Generate point clouds and visualize them.
- Serve results as downloadable files.

Features:
1. Processes CSV files to create structured point clouds with regional and boundary flags.
2. Provides an interactive web interface for file uploads, configuration, and visualization.
3. Automatically cleans up temporary and result files after a configurable delay.

Author: Gerardo Tinoco-Guerrero
Created: May 7th, 2024.
Last Modified: December 4th, 2024.

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
import tempfile
import getpass
import logging
import os

# Third-party library imports
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from shapely.geometry import Point, Polygon
from werkzeug.utils import secure_filename
from shapely.ops import unary_union
import pandas as pd
import numpy as np
import dmsh
import csv

import matplotlib
matplotlib.use('Agg')                                                                                   # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt

# --- Flask Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './tmp/uploads/'                                                          # Directory for uploaded files
app.config['OUTPUT_FOLDER'] = './tmp/results/'                                                          # Directory for output files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}                                               # Allowed extensions for image uploads
app.config['ALLOWED_EXTENSIONS_D'] = {'csv'}                                                            # Allowed extensions for CSV uploads

# --- Logging ---
logging.basicConfig(level=logging.INFO)                                                                 # Set logging level

# --- Environment Variables ---
os.environ['MPLCONFIGDIR'] = "./tmp/" + getpass.getuser()                                               # Ensure matplotlib uses a temporary directory

# --- Ensure Directories Exist ---
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok = True)                                               # Ensure the output folder exists.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)                                               # Ensure the upload folder exists.

"""
Functions Overview:
1. Utility Functions:
   - distance: Calculates mean distance between consecutive points.
   - generate_grid: Generates a grid of points within a bounding box.
   - allowed_file, allowed_file_D: Validates file extensions for uploads.
   - delete_file: Schedules deletion of temporary files.
2. Cloud Generation:
   - CreateCloud: Generates a point cloud with regional and boundary flags.
   - CreateCloud_v2: Adaptive version for point cloud generation.
   - load_and_create_cloud: Integrates file processing and cloud creation logic.
3. Visualization:
   - GraphCloud: Creates a graphical representation of the point cloud.
4. Flask Routes:
   - Handles requests for file uploads, visualization, and downloads.
"""


# --- General Utilities ---
def distance(x, y):
    """
    Calculate the mean distance between consecutive points.

    Parameters:
        x (array):              x-coordinates of the points.
        y (array):              y-coordinates of the points.

    Returns:
        float:                  Mean distance between consecutive points.
    """
    coords = np.column_stack((x, y))                                                                    # Combine x and y into a single array.
    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))                                         # Calculate Euclidean distances.
    return np.mean(dists)                                                                               # Return the mean distance.

def generate_grid(min_x, max_x, min_y, max_y, spacing):
    """
    Generate a grid of points within the bounding box with a given spacing.

    Parameters:
        min_x (float):          Minimum x-coordinate of the bounding box.
        max_x (float):          Maximum x-coordinate of the bounding box.
        min_y (float):          Minimum y-coordinate of the bounding box.
        max_y (float):          Maximum y-coordinate of the bounding box.
        spacing (float):        Distance between points in the grid.

    Returns:
        np.array:               Array of grid points.
    """
    x_coords = np.arange(min_x, max_x, spacing)                                                         # Generate x-coordinates.
    y_coords = np.arange(min_y, max_y, spacing)                                                         # Generate y-coordinates.
    return np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)                                   # Create a grid of points.

def allowed_file(filename):
    """
    Check if a file has an allowed extension for images.

    Parameters:
        filename (str):         Name of the file to check.

    Returns:
        bool:                   True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']   # Check if the file has an allowed extension.

def allowed_file_D(filename):
    """
    Check if a file has an allowed extension for boundaries.

    Parameters:
        filename (str):         Name of the file to check.

    Returns:
        bool:                   True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower()\
            in app.config['ALLOWED_EXTENSIONS_D']                                                       # Check if the file has an allowed extension.

# --- Data Processing ---
def CreateCloud(xb, yb, h_coor_sets, num, rand, region_flag):
    """
    Create a cloud of points with potential holes inside a polygon.

    Parameters:
        xb (array):             x-coordinates of the boundary.
        yb (array):             y-coordinates of the boundary.
        h_coor_sets (list):     List of tuples containing x and y coordinates of the holes.
        num (int):              To determine the density of the cloud.
        rand (int):             Randomization.

    Returns:
        np.array:               Array of generated points with boundary and hole flags.
    """
    
    # Polygons creation
    boundary_polygon = Polygon(np.column_stack((xb, yb)))                                               # Create the main polygon.
    holes            = [Polygon(np.column_stack((hx, hy))) for hx, hy in h_coor_sets]                   # Create a polygon for each of the holes.
    holes_union      = unary_union(holes) if holes else Polygon()                                       # Create the union of the holes.

    # Calculate the minimum distance between consecutive boundary points
    dist = distance(xb, yb)/num                                                                         # Distance for the cloud generation.

    # Cloud Generation
    min_x, min_y, max_x, max_y = boundary_polygon.bounds                                                # Bounding box coordinates for the cloud generation.
    grid_points                = generate_grid(min_x, max_x, min_y, max_y, dist)                        # Generate a grid of points with the calculated minimum distance.

    # Add a flag for each of the nodes.
    points = [[x, y, region_flag, 1] for x, y in zip(xb, yb)]                                           # Boundary points with flag 1.
    
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        points.extend([[x, y, region_flag, 2] for x, y in zip(hx, hy)])                                 # Hole points with flag 2.
    
    generated_points = [
        [x, y, region_flag, 0] for x, y in grid_points\
        if boundary_polygon.contains(Point(x, y)) and not holes_union.contains(Point(x, y))]            # Add grid points if they are inside the main polygon and outside holes
    
    generated_points = np.array(generated_points)                                                       # Convert list to numpy array

    # Randomization
    if rand != 0:                                                                                       # If random is selected.
        perturbation              = 0.5 * dist * (np.random.rand(generated_points.shape[0], 2) - 0.5)   # Define a perturbation for each internal node.
        generated_points[:, 0:2] += perturbation                                                        # Apply the perturbation.

    # Combine all points
    p = np.vstack((points, generated_points))                                                           # Combine all the points in one array.

    return p

def CreateCloud_v2(xb, yb, h_coor_sets, num, rand, region_flag):
    """
    Create an adaptive cloud of points with potential holes inside a polygon.

    Parameters:
        xb (array):             x-coordinates of the boundary.
        yb (array):             y-coordinates of the boundary.
        h_coor_sets (list):     List of tuples containing x and y coordinates of the holes.
        num (int):              How dense the cloud should be.

    Returns:
        np.array:               Array of generated points with boundary and hole flags.
    """

    # Variable initialization
    dist = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))/num                                     # Calculate the distance between points in the boundary.
    pb   = np.column_stack((xb, yb))                                                                    # Stack x and y boundary coordinates.
    geo  = dmsh.Polygon(pb)                                                                             # Create a polygon from boundary points
    
    ## Holes initialization
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        hole_pb  = np.column_stack((hx, hy))                                                            # Stack x and y coordinates of the hole.
        hole_geo = dmsh.Polygon(hole_pb)                                                                # Create a polygon for the hole.
        geo     -= hole_geo                                                                             # Subtract hole geometry from the main polygon.
    
    # Cloud Generation
    X, _       = dmsh.generate(geo, dist)                                                               # Generate points within the polygon.
    poly       = Polygon(pb).buffer(-dist/4)                                                            # Create a buffer around the polygon boundary.
    hole_polys = [Polygon(np.column_stack((hx, hy))).buffer(dist/4) for hx, hy in h_coor_sets]          # Buffer for holes.
    points     = [Point(point[0], point[1]) for point in X]                                             # Convert points to shapely Point objects.
    A          = np.zeros([len(points), 1])                                                             # Initialize an array to store boundary/hole flags.
    
    # Flag the nodes at the boundaries
    for i, point in enumerate(points):                                                                  # For each of the points.
        if not point.within(poly):                                                                      # If the node is not inside the external boundary.
            A[i] = 1                                                                                    # Mark as a node within the external boundary.
        else:
            for hole_poly in hole_polys:                                                                # For each of the nodes inside the external boundary.
                if point.within(hole_poly):                                                             # If the node is in the internal boundary.
                    A[i] = 2                                                                            # Mark as a node within a internal boundary.
                    break
    p = np.column_stack((X, np.full((len(X), 1), region_flag), A))                                      # Combine points and flags into a single array

    # Randomization
    if rand != 0:                                                                                       # If random is selected.
        mask = p[:, 3] == 0                                                                             # Create a mask for points with boundary_flag == 0.
        perturbation = 0.5 * dist * (np.random.rand(np.sum(mask), 2) - 0.5)                             # Define a perturbation for each internal node.
        p[mask, 0:2] += perturbation                                                                    # Apply the perturbation to internal nodes.

    return p

# Function to load CSV files and create the point cloud
def load_and_create_cloud(exterior_file, interior_files, num, rand, mod, gen):
    """
    Load CSV files and create the point cloud with possible randomization.

    Parameters:
        exterior_file (str):    Path to the exterior boundary CSV file.
        interior_files (list):  List of paths to interior hole CSV files.
        num (int):              To determine the density of the cloud.
        rand (int):             Randomization.

    Returns:
        np.array:               Array of generated and possibly randomized points.
    """
    # Data loading.
    pat_out = pd.read_csv(exterior_file)                                                                # Load external boundary CSV.
    xb      = pat_out['x'].values                                                                       # Get x-coordinates of the boundary.
    yb      = pat_out['y'].values                                                                       # Get y-coordinates of the boundary.

    h_coor_sets = []                                                                                    # Create an empty list to store the internal boundaries.
    for interior_file in interior_files:                                                                # For each received file for internal boundaries.
        pat_in = pd.read_csv(interior_file)                                                             # Load the internal boundary CSV.
        hx     = pat_in['x'].values                                                                     # Get x-coordinates of the hole.
        hy     = pat_in['y'].values                                                                     # Get y-coordinates of the hole.
        h_coor_sets.append((hx, hy))                                                                    # Append hole coordinates.

    # Select the cloud creation method based on the value of "mod".
    if mod == 0:
        p = CreateCloud(xb, yb, h_coor_sets, num, rand, region_flag = 1)                                # Use a simple implementation.
    else:
        p = CreateCloud_v2(xb, yb, h_coor_sets, num, rand, region_flag = 1)                             # Use the original implementation
    
    if gen == 1:
        for i, (hx, hy) in enumerate(h_coor_sets, start = 2):                                           # Loop through each interior boundary.
            if mod == 0:
                interior_cloud = CreateCloud(hx, hy, [], num, rand, region_flag = i)                    # Create cloud with the interior as exterior.
            else:
                interior_cloud = CreateCloud_v2(hx, hy, [], num, rand, region_flag = i)                 # Use alternative method.
            
            # Add the points from the individual cloud to the main cloud
            p = np.vstack((p, interior_cloud))

    return p, xb, yb, h_coor_sets

# --- Visualization ---
def GraphCloud(p, xb, yb, h_coor_sets, folder, image_name):
    """
    Graph the generated point cloud and save the plot.

    Parameters:
        p (np.array):           Array of points with boundary flags.
        xb (array):             x-coordinates of the boundary.
        yb (array):             y-coordinates of the boundary.
        h_coor_sets (list):     List of tuples containing x and y coordinates of the holes.
        folder (str):           Directory to save the plot.
        image_name (str):       Name to correctly save the file.
    """
    nomp  = os.path.join(folder, image_name)                                                            # Define the plot filename.
    
    # Unique flags for regions
    unique_flags = np.unique(p[:, 2])
    colors = plt.cm.get_cmap('tab10', len(unique_flags))                                                # Generate a colormap for the regions.

    plt.rcParams["figure.figsize"] = (16, 12)                                                           # Set figure size.
    
    # Complete the polygon for graphics.
    xb          = np.append(xb, xb[0])                                                                  # Copy the first x-coordinate in the end.
    yb          = np.append(yb, yb[0])                                                                  # Copy the first y-coordinate in the end.
    h_coor_sets = [(np.append(hx, hx[0]), np.append(hy, hy[0])) for hx, hy in h_coor_sets]              # Copy the first coordinates in the end.

    # Plot the boundary and holes
    plt.plot(xb, yb, 'r-', label="External Boundary")                                                   # Plot the boundary
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        plt.plot(hx, hy, 'b-', label="Hole Boundary")                                                   # Plot the holes
    
    # Scatter plot for each region
    for i, flag in enumerate(unique_flags):
        region_points = p[p[:, 2] == flag]                                                              # Points belonging to the current region.
        boundary_points = region_points[region_points[:, 3] == 1]                                       # Boundary points.
        interior_points = region_points[region_points[:, 3] == 0]                                       # Interior points.

        plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c=[colors(i)], s=50, label=f"Region {int(flag)} (Boundary)")
        plt.scatter(interior_points[:, 0], interior_points[:, 1], c=[colors(i)], s=20, alpha=0.5, label=f"Region {int(flag)} (Interior)")

    plt.title('Generated Cloud')                                                                        # Set plot title.
    plt.axis('equal')                                                                                   # Set equal axes.
    #plt.legend(loc='upper right')                                                                       # Add a legend for regions and boundaries.
    plt.savefig(nomp)                                                                                   # Save the plot.
    plt.close()                                                                                         # Close the plot.

# --- File Handling ---
def delete_file(path, delay):
    """
    Delete a file after a specified delay.

    Parameters:
        path (str):             Path to the file to delete.
        delay (int):            Delay in seconds before deleting the file.
    """
    def delayed_delete():
        try:
            if os.path.exists(path):                                                                    # If the file exists.
                os.remove(path)                                                                         # Remove the file.
                logging.info(f"File deleted: {path}")                                                   # Save the deletion in the log.
            else:                                                                                       # If the file doesn't exist.
                logging.warning(f"The file doesn't exists: {path}")                                     # Save the error in the log.
        except Exception as e:                                                                          # If there is an exception.
            logging.error(f"Error while deleting the file {path}: {str(e)}")                            # Save the error in the log.

    timer = Timer(delay, delayed_delete)                                                                # Create the timer.
    timer.start()                                                                                       # Start the timer.

# --- General Routes ---
@app.route('/')
def index():
    """
    Render the index page.
    """
    return render_template('index.html')                                                                # Render the index page.

@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('about.html')                                                                # Render the about page.

@app.route('/howto')
def howto():
    """
    Render the info page.
    """
    return render_template('howto.html')                                                                 # Render the info page.

# --- File Upload Routes ---
@app.route('/uploadI', methods = ['GET', 'POST'])
def upload_image():
    """
    Handle image file uploads.
    """
    if request.method == 'POST':                                                                        # If a POST was performed.
        if 'file' not in request.files:                                                                 # If there is no file.
            return redirect(request.url)                                                                # Render the UploadI page.
        file = request.files['file']                                                                    # Get the loaded file.
        if file.filename == '':                                                                         # If no file was selected.
            return redirect(request.url)                                                                # Render the UploadI page.
        
        if file and allowed_file(file.filename):                                                        # If the file exists and has a valid file extension.
            filename = secure_filename(file.filename)                                                   # Secure the filename.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)                              # Save the path for the file.
            file.save(filepath)                                                                         # Save the file.
            delete_file(filepath, 3600)                                                                 # Schedule file deletion.

            return render_template('contour.html', filename = filename)                                 # Render the contour page.
    return render_template('uploadI.html')                                                              # Render the UploadI page.

@app.route('/uploadC', methods=['GET', 'POST'])
def upload_files():
    """
    Handle single CSV file upload, process points, and create a cloud of points.
    """
    if request.method == 'POST':
        # Verificar si el archivo está presente
        if 'points_file' not in request.files:
            print("No encuentro un archivo.")
            return redirect(request.url)

        file = request.files['points_file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file_D(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Leer el CSV y validar columnas
            try:
                df = pd.read_csv(file_path)
                required_columns = {'x', 'y', 'flag'}
                if not required_columns.issubset(df.columns):
                    return "CSV file must contain columns: 'x', 'y', 'flag'", 400
            except Exception as e:
                return f"Error reading the file: {e}", 500

            # Separar puntos
            exterior_points = df[df['flag'] == 1][['x', 'y']].values.tolist()
            interior_points = [
                df[df['flag'] == flag_value][['x', 'y']].values.tolist()
                for flag_value in sorted(df['flag'].unique()) if flag_value != 1
            ]

            # Crear archivos temporales
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as ext_file:
                pd.DataFrame(exterior_points, columns=['x', 'y']).to_csv(ext_file.name, index=False)
                exterior_file_path = ext_file.name

            interior_file_paths = []
            try:
                for points in interior_points:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as int_file:
                        pd.DataFrame(points, columns=['x', 'y']).to_csv(int_file.name, index=False)
                        interior_file_paths.append(int_file.name)

                # Obtener parámetros
                num = int(request.form.get('num', 3))
                rand = int(request.form.get('rand', 1))
                mod = int(request.form.get('mod', 1))
                gen = int(request.form.get('gen', 1))

                # Crear la nube de puntos
                try:
                    p, xb, yb, h_coor_sets = load_and_create_cloud(exterior_file_path, interior_file_paths, num, rand, mod, gen)
                except Exception as e:
                    return f"Error generating the cloud: {e}", 500

                # Generar nombres de archivos de resultados
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f'plot_{timestamp}.png'
                p_csv_name = f'cloud_{timestamp}.csv'

                # Crear la gráfica
                GraphCloud(p, xb, yb, h_coor_sets, folder=app.config['OUTPUT_FOLDER'], image_name=image_name)

                # Guardar la nube en CSV
                csv_path = os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name)
                pd.DataFrame(p, columns=["x", "y", "region", "boundary_flag"]).to_csv(csv_path, index=False)

                # Programar eliminación de archivos
                delete_file(file_path, 3600)
                delete_file(exterior_file_path, 3600)
                for path in interior_file_paths:
                    delete_file(path, 3600)
                delete_file(os.path.join(app.config['OUTPUT_FOLDER'], image_name), 3600)
                delete_file(csv_path, 3600)

                # Renderizar resultados
                return render_template(
                    'cloud.html',
                    image_name=image_name,
                    p_csv_name=p_csv_name,
                    tables=[pd.DataFrame(p, columns=["x", "y", "region", "boundary_flag"]).to_html(classes='data')],
                    titles=['na', 'Cloud Data']
                )

            finally:
                # Asegurarse de eliminar archivos temporales en caso de error
                for path in interior_file_paths:
                    if os.path.exists(path):
                        os.unlink(path)
                if os.path.exists(exterior_file_path):
                    os.unlink(exterior_file_path)

    # Renderizar la página de carga
    return render_template('uploadC.html')

# --- File Download Routes ---
@app.route('/download_csv', methods=['POST'])
def download_csv():
    """
    Handle CSV file download requests.
    """
    data = request.get_json()                                                                           # Get JSON data from the request.
    points = data['points']                                                                             # Extract points data.
    
    # Generate the name of the file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                                                # Generate a timestamp with the date and time.
    csv_filename = f'points_{timestamp}.csv'                                                            # Generate the CSV filename

    # Save CSV to a file
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)                                  # Complete path for the generated file.
    with open(csv_path, mode='w', newline='') as file:                                                  # Write the information into the file.
        cw = csv.writer(file)                                                                           # Create the file.
        cw.writerow(['x', 'y', 'flag'])                                                                 # Write the header.
        cw.writerows([[p['x'], p['y'], p['flag']] for p in points])                                     # Write the points data.

    # Schedule file deletion after 1 hour
    delete_file(csv_path, 3600)                                                                         # Delete the created file.

    # Send the file as a download response
    return {'filename': csv_filename}

# --- Processing Routes ---
@app.route('/modify', methods = ['GET', 'POST'])
def modify():
    """
    Render the info page.
    """
    if request.method == 'POST':                                                                        # If a POST was performed.
        if 'file' not in request.files:                                                                 # If there is no file.
            return redirect(request.url)                                                                # Render the UploadI page.
        file = request.files['file']                                                                    # Get the loaded file.
        if file.filename == '':                                                                         # If no file was selected.
            return redirect(request.url)                                                                # Render the UploadI page.
        
        if file and allowed_file_D(file.filename):                                                        # If the file exists and has a valid file extension.
            filename = secure_filename(file.filename)                                                   # Secure the filename.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)                              # Save the path for the file.
            file.save(filepath)                                                                         # Save the file.
            delete_file(filepath, 3600)                                                                 # Schedule file deletion.

            return render_template('modify.html', filename = filename)                                 # Render the contour page.
    return render_template('uploadM.html')                                                                # Render the modify page.

# --- File Serving Routes ---
@app.route('/tmp/results/<path:filename>')
def static_files(filename):
    """ 
    Serve result files.
    """
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/tmp/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug = True)