# Import necessary libraries
import pandas as pd
import numpy as np
import logging
import random
import dmsh
import csv
import os
import io

# Import parts of some libraries needed for the code later
from flask import Flask, render_template, request, redirect, Response, send_from_directory
from shapely.geometry import Point, Polygon, MultiPoint
from werkzeug.utils import secure_filename
from threading import Timer

# Import matplotlib for web plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define system variables
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'                                                         # Directory for uploading files.
app.config['OUTPUT_FOLDER'] = 'static/results/'                                                         # Directory for output files.
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}                                               # Allowed file extensions for images.
app.config['ALLOWED_EXTENSIONS_D'] = {'csv'}                                                            # Allowed file extensions for boundaries.

# Set up logging
logging.basicConfig(level=logging.INFO)                                                                 # Configure logging.

# Ensure upload directories exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok = True)                                               # Ensure the output folder exists.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)                                               # Ensure the upload folder exists.

# Function to create clouds of points
def CreateCloud(xb, yb, h_coor_sets, num):
    """
    Create a cloud of points with potential holes inside a polygon.

    Parameters:
        xb (array):             x-coordinates of the boundary.
        yb (array):             y-coordinates of the boundary.
        h_coor_sets (list):     List of tuples containing x and y coordinates of the holes.
        num (int):              How dense the cloud should be.

    Returns:
        np.array:               Array of generated points with boundary and hole flags.
    """
    # Variable initialization.
    dist = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))/num                                     # Calculate the distance between points in the boundary.
    pb   = np.column_stack((xb, yb))                                                                    # Stack x and y boundary coordinates.
    geo  = dmsh.Polygon(pb)                                                                             # Create a polygon from boundary points
    
    ## Holes initialization.
    for hx, hy in h_coor_sets:                                                                          # For each of the holes.
        hole_pb  = np.column_stack((hx, hy))                                                            # Stack x and y coordinates of the hole.
        hole_geo = dmsh.Polygon(hole_pb)                                                                # Create a polygon for the hole.
        geo     -= hole_geo                                                                             # Subtract hole geometry from the main polygon.
    
    # Cloud Generation.
    X, _       = dmsh.generate(geo, dist)                                                               # Generate points within the polygon.
    poly       = Polygon(pb).buffer(-dist/4)                                                            # Create a buffer around the polygon boundary.
    hole_polys = [Polygon(np.column_stack((hx, hy))).buffer(dist/4) for hx, hy in h_coor_sets]          # Buffer for holes.
    points     = [Point(point[0], point[1]) for point in X]                                             # Convert points to shapely Point objects.
    A          = np.zeros([len(points), 1])                                                             # Initialize an array to store boundary/hole flags.
    
    # Flag the nodes at the boundaries.
    for i, point in enumerate(points):                                                                  # For each of the points.
        if not point.within(poly):                                                                      # If the node is not inside the external boundary.
            A[i] = 1                                                                                    # Mark as a node within the external boundary.
        else:
            for hole_poly in hole_polys:                                                                # For each of the nodes inside the external boundary.
                if point.within(hole_poly):                                                             # If the node is in the internal boundary.
                    A[i] = 2                                                                            # Mark as a node within a internal boundary.
                    break
    X = np.column_stack((X, A))                                                                         # Combine points and flags into a single array

    return X

def Randomize(p, rand):
    """
    Randomly move points within a polygon boundary.

    Parameters:
        p (np.array):           Array of points with boundary flags.
        me (int):               Magnitude of the movement (1, 2, or 3).

    Returns:
        np.array:               Array of randomized points.
    """
    # Variable Initialization.
    m        = len(p[:, 0])                                                                             # Get the number of boundary nodes.
    boundary = MultiPoint(p[p[:, 2] == 1][:, :2])                                                       # Get the boundary nodes.
    pol      = boundary.convex_hull                                                                     # Create a polygon.
    pol      = pol.buffer(-0.01)                                                                        # Create a buffer for the polygon.
    r        = rand/150                                                                                 # Define the random movement magnitude.
    print(r)
    
    # Randomly move the nodes.
    for i in range(m):                                                                                  # For each of the nodes.
        if p[i, 2] == 0:                                                                                # If the node is not in one of the boundaries.
            inside_poly = False                                                                         # inside_poly is False to perform at least one iteration.
            while not inside_poly:                                                                      # While the node is outside the boundaries.
                move_x    = random.uniform(-r, r)                                                       # Random x movement.
                move_y    = random.uniform(-r, r)                                                       # Random y movement.
                new_point = Point(p[i, 0] + move_x, p[i, 1] + move_y)                                   # Create new point with random movement.
                in_poly   = pol.contains(new_point)                                                     # Check if the new point is inside the polygon.
                on_poly   = pol.touches(new_point)                                                      # Check if the new point is on the polygon.
                if in_poly and not on_poly:                                                             # If the new point is inside of the boundaries and not on the boundaries.
                    p[i, 0] += move_x                                                                   # Apply x movement.
                    p[i, 1] += move_y                                                                   # Apply y movement.
                    inside_poly = True                                                                  # inside_poly changes to True.
    return p

# Function to load CSV files and create the point cloud
def load_and_create_cloud(exterior_file, interior_files, num, rand):
    """
    Load CSV files and create the point cloud with possible randomization.

    Parameters:
        exterior_file (str):    Path to the exterior boundary CSV file.
        interior_files (list):  List of paths to interior hole CSV files.
        num (int):              Number of divisions for the boundary.
        rand (int):             Randomization magnitude.

    Returns:
        np.array:               Array of generated and possibly randomized points.
    """
    # Data loading.
    pat_out     = pd.read_csv(exterior_file)                                                            # Load external boundary CSV.
    xb          = pat_out['x'].values                                                                   # Get x-coordinates of the boundary.
    yb          = pat_out['y'].values                                                                   # Get y-coordinates of the boundary.

    h_coor_sets = []                                                                                    # Create an empty list to store the internal boundaries.
    for interior_file in interior_files:                                                                # For each received file for internal boundaries.
        pat_in = pd.read_csv(interior_file)                                                             # Load the internal boundary CSV.
        hx     = pat_in['x'].values                                                                     # Get x-coordinates of the hole.
        hy     = pat_in['y'].values                                                                     # Get y-coordinates of the hole.
        h_coor_sets.append((hx, hy))                                                                    # Append hole coordinates.

    #Cloud creation.
    X = CreateCloud(xb, yb, h_coor_sets, num = num)                                                     # Create the cloud of points.

    # Randomization process.
    if rand != 0:                                                                                       # If randomized factor is greater than 0%.
        X = Randomize(X, rand)                                                                          # Randomize points.

    return X

# Function to graph the point cloud for display
def GraphCloud(p, folder):
    """
    Graph the generated point cloud and save the plot.

    Parameters:
        p (np.array):           Array of points with boundary flags.
        folder (str):           Directory to save the plot.
    """
    nomp  = folder + 'result_plot.png'                                                                  # Define the plot filename.
    color = ['blue' if x == 0 else 'red' for x in p[:, 2]]                                              # Set colors based on flags.
    plt.rcParams["figure.figsize"] = (16, 12)                                                           # Set figure size.
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS_D'] # Check if the file has an allowed extension.

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
        X = load_and_create_cloud(exterior_path, interior_paths, num, rand)                             # Create the cloud of points.
        
        # Delete Files
        for path in interior_paths:                                                                     # For each path in external boundaries.
            delete_file(path, 3600)                                                                     # Schedule external boundaries files deletion.

        image_name = 'result_plot.png'                                                                  # Name for the saved plot.
        GraphCloud(X, folder = app.config['OUTPUT_FOLDER'])                                             # Graph the generated cloud of points.

        # Save results to CSV files
        p_csv_name  = 'p.csv'                                                                           # Name of the resulting file.
        pd.DataFrame(X, columns = ["x", "y", "boundary_flag"]).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name), index = False)

        return render_template('cloud.html', image_name = image_name, p_csv_name = p_csv_name, tables = [pd.DataFrame(X, columns = ["x", "y", "boundary_flag"]).to_html(classes = 'data')], titles = ['na', 'Cloud Data'])
    return render_template('uploadC.html')                                                              # Render the UploadC page.

@app.route('/static/results/<path:filename>')
def static_files(filename):
    """
    Serve static result files.
    """
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

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
    return Response(output, mimetype = "text/csv", headers = {"Content-disposition": "attachment; filename = points.csv"})

if __name__ == '__main__':
    app.run(debug = True)