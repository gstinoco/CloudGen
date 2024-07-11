from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import dmsh
from shapely.geometry import Point, Polygon
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'

# Asegúrate de que los directorios de carga y salida existan
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def CreateCloud_Holes(xb, yb, num, h_coor_sets):
    dist = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2)) / num
    pb  = np.column_stack((xb, yb))
    geo = dmsh.Polygon(pb)
    
    for hx, hy in h_coor_sets:
        hole_pb = np.column_stack((hx, hy))
        hole_geo = dmsh.Polygon(hole_pb)
        geo -= hole_geo

    X, cells = dmsh.generate(geo, dist)
    poly = Polygon(pb).buffer(-dist / 4)
    hole_polys = [Polygon(np.column_stack((hx, hy))).buffer(dist / 4) for hx, hy in h_coor_sets]

    points = [Point(point[0], point[1]) for point in X]
    A = np.zeros([len(points), 1])
    
    for i, point in enumerate(points):
        if not point.within(poly):
            A[i] = 1
        else:
            for hole_poly in hole_polys:
                if point.within(hole_poly):
                    A[i] = 2
                    break

    X = np.column_stack((X, A))
    return X, cells

def load_and_create_cloud(exterior_file, interior_files, num):
    pat_out = pd.read_csv(exterior_file)
    xb = pat_out['x'].values
    yb = pat_out['y'].values

    h_coor_sets = []
    for interior_file in interior_files:
        pat_in = pd.read_csv(interior_file)
        hx = pat_in['x'].values
        hy = pat_in['y'].values
        h_coor_sets.append((hx, hy))

    X, cells = CreateCloud_Holes(xb, yb, num, h_coor_sets)
    return X, cells

def GraphCloud(p, folder, nom):
    nomp = folder + nom + '.png'

    color = ['blue' if x == 0 else 'red' for x in p[:, 2]]

    plt.rcParams["figure.figsize"] = (16, 12)
    plt.scatter(p[:, 0], p[:, 1], c=color, s=20)
    plt.title(nom)
    plt.axis('equal')
    plt.savefig(nomp)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'exterior' not in request.files:
        return redirect(request.url)

    exterior_file = request.files['exterior']
    if exterior_file.filename == '':
        return redirect(request.url)
    
    num = int(request.form.get('num', 100))

    exterior_path = os.path.join(app.config['UPLOAD_FOLDER'], exterior_file.filename)
    exterior_file.save(exterior_path)

    interior_files = request.files.getlist('interiors')
    interior_paths = []
    for interior_file in interior_files:
        if interior_file.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], interior_file.filename)
            interior_file.save(path)
            interior_paths.append(path)
    
    X, cells = load_and_create_cloud(exterior_path, interior_paths, num)

    image_name = 'result_plot.png'
    image_path = os.path.join(app.config['OUTPUT_FOLDER'], image_name)
    GraphCloud(X, folder=app.config['OUTPUT_FOLDER'] + '/', nom='result_plot')

    # Guardar los resultados en archivos CSV
    p_csv_name = 'p.csv'
    tt_csv_name = 'tt.csv'
    pd.DataFrame(X, columns=["x", "y", "boundary_flag"]).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name), index=False)
    pd.DataFrame(cells).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], tt_csv_name), index=False)

    return render_template('result.html', image_name=image_name, p_csv_name=p_csv_name, tt_csv_name=tt_csv_name, tables=[pd.DataFrame(X, columns=["x", "y", "boundary_flag"]).to_html(classes='data')], titles=['na', 'Point Cloud Data'])

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
