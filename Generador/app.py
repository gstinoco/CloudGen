# Import necessary libraries
import pandas as pd
import numpy as np
import dmsh
import csv
import os
import io

# Import parts of some libraries needed for the code later
from flask import Flask, render_template, request, redirect, Response, send_from_directory
from threading import Timer
from werkzeug.utils import secure_filename
from shapely.geometry import Point, Polygon

# Import matplotlib for web plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define system variables
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/results/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directories exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok = True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

# Function to create point clouds with holes
def CreateCloud_Holes(xb, yb, h_coor_sets, num):
    dist = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))/num
    pb   = np.column_stack((xb, yb))
    geo  = dmsh.Polygon(pb)
    
    for hx, hy in h_coor_sets:
        hole_pb  = np.column_stack((hx, hy))
        hole_geo = dmsh.Polygon(hole_pb)
        geo     -= hole_geo
    
    X, cells   = dmsh.generate(geo, dist)
    poly       = Polygon(pb).buffer(-dist / 4)
    hole_polys = [Polygon(np.column_stack((hx, hy))).buffer(dist / 4) for hx, hy in h_coor_sets]
    points     = [Point(point[0], point[1]) for point in X]
    A          = np.zeros([len(points), 1])
    
    for i, point in enumerate(points):
        if not point.within(poly):
            A[i] = 1
        else:
            for hole_poly in hole_polys:
                if point.within(hole_poly):
                    A[i] = 2
                    break

    X = np.column_stack((X, A))
    return X

# Function to load CSV files and create the point cloud
def load_and_create_cloud(exterior_file, interior_files, num):
    pat_out = pd.read_csv(exterior_file)
    xb      = pat_out['x'].values
    yb      = pat_out['y'].values

    h_coor_sets = []
    for interior_file in interior_files:
        pat_in = pd.read_csv(interior_file)
        hx     = pat_in['x'].values
        hy     = pat_in['y'].values
        h_coor_sets.append((hx, hy))

    X = CreateCloud_Holes(xb, yb, h_coor_sets, num = num)
    return X

# Function to graph the point cloud for display
def GraphCloud(p, folder):
    nomp  = folder + 'result_plot.png'
    color = ['blue' if x == 0 else 'red' for x in p[:, 2]]
    plt.rcParams["figure.figsize"] = (16, 12)
    plt.scatter(p[:, 0], p[:, 1], c = color, s = 20)
    plt.title('Generated Cloud')
    plt.axis('equal')
    plt.savefig(nomp)
    plt.close()

# Function to define allowed file types
def allowed_file(filename):
    '''
    Define allowed file types for upload.
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to delete files after a delay
def delete_file(path, delay):
    '''
    Schedule the deletion of temporary files
    '''
    def delayed_delete():
        if os.path.exists(path):
            os.remove(path)

    timer = Timer(delay, delayed_delete)
    timer.start()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/uploadI', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            delete_file(filepath, 3600)
            return render_template('contour.html', filename = filename)
    return render_template('uploadI.html')

@app.route('/uploadC', methods = ['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'exterior' not in request.files:
            return redirect(request.url)
        
        num = int(request.form.get('num', 100))
        print(num)

        exterior_file = request.files['exterior']
        if exterior_file.filename == '':
            return redirect(request.url)
        
        exterior_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(exterior_file.filename))
        exterior_file.save(exterior_path)

        interior_files = request.files.getlist('interiors')
        interior_paths = []
        for interior_file in interior_files:
            if interior_file.filename != '':
                path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(interior_file.filename))
                interior_file.save(path)
                interior_paths.append(path)
        
        X = load_and_create_cloud(exterior_path, interior_paths, num)

        image_name = 'result_plot.png'
        GraphCloud(X, folder = app.config['OUTPUT_FOLDER'])

        # Save results to CSV files
        p_csv_name  = 'p.csv'
        pd.DataFrame(X, columns = ["x", "y", "boundary_flag"]).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name), index = False)

        return render_template('cloud.html', image_name = image_name, p_csv_name = p_csv_name, tables = [pd.DataFrame(X, columns = ["x", "y", "boundary_flag"]).to_html(classes = 'data')], titles = ['na', 'Cloud Data'])
    return render_template('uploadC.html')

@app.route('/static/results/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/download_csv', methods = ['POST'])
def download_csv():
    data = request.get_json()
    points = data['points']
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['x', 'y'])
    cw.writerows([[p['x'], p['y']] for p in points])
    output = si.getvalue()
    return Response(output, mimetype = "text/csv", headers = {"Content-disposition": "attachment; filename = points.csv"})

if __name__ == '__main__':
    app.run(debug = True)