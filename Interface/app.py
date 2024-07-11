import os
import csv
import io
from threading import Timer
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import pandas as pd
import numpy as np
import dmsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'csv'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def allowed_file(filename):
    '''
    Definir el tipo de archivos que se permite subir.
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def delete_file(path, delay):
    '''
    Programar el borrado de archivos temporales
    '''
    def delayed_delete():
        if os.path.exists(path):
            os.remove(path)

    timer = Timer(delay, delayed_delete)
    timer.start()

@app.route('/upload', methods=['GET', 'POST'])
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
            return render_template('contour.html', filename=filename)
    return render_template('upload.html')

@app.route('/creator', methods=['GET', 'POST'])
def upload_image2():
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
            return render_template('creator.html', filename=filename)
    return render_template('upload.html')

@app.route('/create_cloud', methods=['POST'])
def create_cloud():
    data = request.get_json()
    points = data['points']
    filename = 'temp_cloud.csv'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])
        writer.writerows([[p['x'], p['y']] for p in points])
    
    return jsonify({'url': url_for('view_cloud', filename=filename)})

@app.route('/view_cloud/<filename>')
def view_cloud(filename):
    return render_template('view_cloud.html', filename=filename)


@app.route('/capture_point', methods=['POST'])
def capture_point():
    data = request.get_json()
    print(data)  # Aquí puedes procesar o guardar los datos como prefieras
    return jsonify({'status': 'success', 'x': data['x'], 'y': data['y']})

@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = request.get_json()
    points = data['points']
    si = io.StringIO()  # Usar StringIO para manejar el archivo CSV en memoria
    cw = csv.writer(si)
    cw.writerow(['x', 'y'])  # Escribir la cabecera del CSV
    cw.writerows([[p['x'], p['y']] for p in points])  # Escribir los puntos

    output = si.getvalue()
    return Response(output, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=points.csv"})

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

if __name__ == '__main__':
    app.run(debug=True)
