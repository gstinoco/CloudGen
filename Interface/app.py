import os
import csv
import io
from threading import Timer
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response

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

if __name__ == '__main__':
    app.run(debug=True)
