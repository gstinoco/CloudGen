# Import necessary libraries
import pandas as pd
import numpy as np
import getpass
import logging
import csv
import io
import os
os.environ['MPLCONFIGDIR'] = "/tmp/" + getpass.getuser()

# Import parts of some libraries needed for the code later
from flask import Flask, render_template, request, redirect, Response, send_from_directory
from shapely.geometry import Point, Polygon
from werkzeug.utils import secure_filename
from shapely.ops import unary_union
from threading import Timer

# Import matplotlib for web plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define system variables
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'                                                            # Directory for uploading files.
app.config['OUTPUT_FOLDER'] = '/tmp/results/'                                                            # Directory for output files.
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}                                               # Allowed file extensions for images.
app.config['ALLOWED_EXTENSIONS_D'] = {'csv'}                                                            # Allowed file extensions for boundaries.

# Set up logging
logging.basicConfig(level=logging.INFO)                                                                 # Configure logging.

# Ensure upload directories exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok = True)                                               # Ensure the output folder exists.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)                                               # Ensure the upload folder exists.

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

def CreateCloud(xb, yb, h_coor_sets, num, rand):
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
    points = [[x, y, 1] for x, y in zip(xb, yb)]                                                        # Boundary points with flag 1.
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        points.extend([[x, y, 2] for x, y in zip(hx, hy)])                                              # Hole points with flag 2.
    generated_points = [[x, y, 0] for x, y in grid_points\
        if boundary_polygon.contains(Point(x, y)) and not holes_union.contains(Point(x, y))]            # Add grid points if they are inside the main polygon and outside holes
    generated_points = np.array(generated_points)                                                       # Convert list to numpy array

    # Randomization
    if rand != 0:                                                                                       # If random is selected.
        perturbation              = 0.5 * dist * (np.random.rand(generated_points.shape[0], 2) - 0.5)   # Define a perturbation for each internal node.
        generated_points[:, 0:2] += perturbation                                                        # Apply the perturbation.

    # Combine all points
    p = np.vstack((points, generated_points))                                                           # Combine all the points in one array.

    # Check and flag boundary points (if needed)
    boundary_buffer = boundary_polygon.buffer(-dist/(4*num))                                            # Create a buffer to check for new boundary nodes.
    for i, (x, y, flag) in enumerate(p):                                                                # For each of the nodes.
        if flag == 0 and not boundary_buffer.contains(Point(x, y)):                                     # If it was an interior node and is in the boundary.
            p[i, 2] = 1                                                                                 # 1 means on the boundary.

    # Check and flag hole points (if needed)
    for hole in holes:                                                                                  # For each of the holes.
        hole_buffer = hole.buffer(dist/(4*num))                                                         # Create a buffer to check for new boundary nodes.
        for i, (x, y, flag) in enumerate(p):                                                            # For each of the nodes.
            if flag == 0 and hole_buffer.contains(Point(x, y)):                                         # If it was an interior node and is in a hole.
                p[i, 2] = 2                                                                             # 2 means inside a hole

    return p

# Function to load CSV files and create the point cloud
def load_and_create_cloud(exterior_file, interior_files, num, rand):
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

    # Cloud creation.
    p = CreateCloud(xb, yb, h_coor_sets, num, rand)                                                     # Create the cloud of points.

    return p, xb, yb, h_coor_sets

# Function to graph the point cloud for display
def GraphCloud(p, xb, yb, h_coor_sets, folder):
    """
    Graph the generated point cloud and save the plot.

    Parameters:
        p (np.array):           Array of points with boundary flags.
        xb (array):             x-coordinates of the boundary.
        yb (array):             y-coordinates of the boundary.
        h_coor_sets (list):     List of tuples containing x and y coordinates of the holes.
        folder (str):           Directory to save the plot.
    """
    nomp  = folder + 'result_plot.png'                                                                  # Define the plot filename.
    color = ['blue' if x == 0 else 'red' for x in p[:, 2]]                                              # Set colors based on flags.
    plt.rcParams["figure.figsize"] = (16, 12)                                                           # Set figure size.
    
    # Complete the polygon for graphics.
    xb          = np.append(xb, xb[0])                                                                  # Copy the first x-coordinate in the end.
    yb          = np.append(yb, yb[0])                                                                  # Copy the first y-coordinate in the end.
    h_coor_sets = [(np.append(hx, hx[0]), np.append(hy, hy[0])) for hx, hy in h_coor_sets]              # Copy the first coordinates in the end.

    # Plot the boundary and holes
    plt.plot(xb, yb, 'r-')                                                                              # Plot the boundary
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        plt.plot(hx, hy, 'r-')                                                                          # Plot the holes

    plt.scatter(p[:, 0], p[:, 1], c = color, s = 20)                                                    # Create scatter plot.
    plt.title('Generated Cloud')                                                                        # Set plot title.
    plt.axis('equal')                                                                                   # Set equal axes.
    plt.savefig(nomp)                                                                                   # Save the plot.
    plt.close()                                                                                         # Close the plot.

# Function to define allowed file types for images.
def allowed_file(filename):
    """
    Check if a file has an allowed extension for images.

    Parameters:
        filename (str):         Name of the file to check.

    Returns:
        bool:                   True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']   # Check if the file has an allowed extension.

# Function to define allowed file types for boundaries.
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

# Function to delete files after a delay
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

# Flask routes
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

@app.route('/uploadC', methods = ['GET', 'POST'])
def upload_files():
    """
    Handle CSV file uploads for boundary and holes.
    """
    # Check for files.
    if request.method == 'POST':                                                                        # If a POST was performed.
        ## Check for external boundary files.
        if 'exterior' not in request.files:                                                             # If there is no file.
            return redirect(request.url)                                                                # Render the UploadC page.
        exterior_file = request.files['exterior']                                                       # Get the loaded file.
        if exterior_file.filename == '':                                                                # If no file was selected.
            return redirect(request.url)                                                                # Render the UploadC page.
        if exterior_file and allowed_file_D(exterior_file.filename):                                    # If the file exists and has a valid file extension.
            filename = secure_filename(exterior_file.filename)                                          # Secure the filename.
            exterior_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                         # Save the path for the file.
            exterior_file.save(exterior_path)                                                           # Save the file.
            delete_file(exterior_path, 3600)                                                            # Schedule external boundary file deletion.

        ## Check for internal boundary files.
        interior_files = request.files.getlist('interiors')                                             # Get all the files for external boundaries.
        interior_paths = []                                                                             # Create a list of external boundaries files.
        for interior_file in interior_files:                                                            # For each of the loaded files.
            if interior_file.filename != '' and allowed_file_D(interior_file.filename):                 # If a files was selected.
                filename = secure_filename(interior_file.filename)                                      # Secure the filename.
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                              # Save the path for the file.
                interior_file.save(path)                                                                # Save the file.
                interior_paths.append(path)                                                             # Save the file path.

        # Get parameters from HTML.
        num = int(request.form.get('num', 100))                                                         # Get num from HTML form.
        rand = int(request.form.get('rand', 100))                                                       # Get rand from HTML form.
        
        # Cloud creation.
        p, xb, yb, h_coor_sets = load_and_create_cloud(exterior_path, interior_paths, num, rand)        # Create the cloud of points.
        
        # Delete Files
        for path in interior_paths:                                                                     # For each path in external boundaries.
            delete_file(path, 3600)                                                                     # Schedule external boundaries files deletion.

        image_name = 'result_plot.png'                                                                  # Name for the saved plot.
        GraphCloud(p, xb, yb, h_coor_sets, folder=app.config['OUTPUT_FOLDER'])                          # Graph the generated cloud of points.

        # Save results to CSV files
        p_csv_name  = 'p.csv'                                                                           # Name of the resulting file.
        pd.DataFrame(p, columns = ["x", "y", "boundary_flag"]).to_csv(os.path.join\
                     (app.config['OUTPUT_FOLDER'], p_csv_name), index = False)

        return render_template('cloud.html', image_name = image_name, p_csv_name = p_csv_name, \
                               tables = [pd.DataFrame(p, columns = ["x", "y", "boundary_flag"]).\
                               to_html(classes = 'data')], titles = ['na', 'Cloud Data'])
    return render_template('uploadC.html')                                                              # Render the UploadC page.

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

@app.route('/download_csv', methods = ['POST'])
def download_csv():
    """
    Handle CSV file download requests.
    """
    data = request.get_json()                                                                           # Get JSON data from the request
    points = data['points']                                                                             # Extract points data
    si = io.StringIO()                                                                                  # Create an in-memory string buffer
    cw = csv.writer(si)                                                                                 # Create a CSV writer object
    cw.writerow(['x', 'y'])                                                                             # Write header
    cw.writerows([[p['x'], p['y']] for p in points])                                                    # Write points data
    output = si.getvalue()                                                                              # Get the CSV content as a string
    return Response(output, mimetype = "text/csv",\
                    headers = {"Content-disposition": "attachment; filename = points.csv"})

if __name__ == '__main__':
    app.run(debug = True)