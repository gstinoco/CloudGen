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
Last Modified: January 9th, 2024.

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

# Plot imports
import matplotlib
matplotlib.use('Agg')                                                                                   # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

# --- Flask Configuration ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                                                   # Absolute route for the script.
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'tmp', 'uploads')                                  # Directory for uploaded files
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'tmp', 'results')                                  # Directory for output files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}                                               # Allowed extensions for image uploads
app.config['ALLOWED_EXTENSIONS_D'] = {'csv'}                                                            # Allowed extensions for CSV uploads

# --- Logging ---
logging.basicConfig(level=logging.INFO)                                                                 # Set logging level

# --- Environment Variables ---
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, 'tmp', 'matplotlib')                                # Ensure matplotlib uses a temporary directory

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
def distance(x, y, hole_coordinates=None):
    """
    Calcula la distancia promedio entre puntos consecutivos en el contorno principal y los huecos.

    Args:
        xb (list): Coordenadas X del polígono principal.
        yb (list): Coordenadas Y del polígono principal.
        hole_coordinates (list): Lista de pares de coordenadas ([hx], [hy]) para los huecos.

    Returns:
        float: Distancia promedio considerando el polígono principal y los huecos.
    """
    # Combinar coordenadas del contorno principal
    coords = np.column_stack((x, y))                                                                    # Crear un array de las coordenadas principales
    
    # Agregar coordenadas de los huecos, si existen
    if hole_coordinates:
        for hx, hy in hole_coordinates:
            hole_coords = np.column_stack((hx, hy))                                                     # Crear array para cada hueco
            coords = np.vstack((coords, hole_coords))                                                   # Combinar coordenadas de huecos con las principales
    
    # Calcular distancias entre puntos consecutivos
    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    
    return np.mean(dists)                                                                               # Retornar la distancia promedio

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
def CreateCloud(xb, yb, h_coor_sets, dist, rand, region_flag, method = 1):
    """
    Genera una nube de puntos dentro de un polígono delimitado, permitiendo agujeros internos y opciones de aleatoriedad.

    Args:
        xb (list): Coordenadas X del polígono principal.
        yb (list): Coordenadas Y del polígono principal.
        h_coor_sets (list): Lista de huecos, donde cada hueco es un par de listas ([hx], [hy]).
        dist (float): Distancia mínima entre puntos generados.
        rand (int): Si es diferente de 0, aplica aleatoriedad a los puntos.
        region_flag (int): Bandera que identifica la región.
        method (str): Método de generación ("v1" o "v2").
    Returns:
        np.ndarray: Nube de puntos generada con columnas [x, y, region_flag, tipo].
    """
    boundary_polygon = Polygon(np.column_stack((xb, yb)))
    holes            = [Polygon(np.column_stack((hx, hy))) for hx, hy in h_coor_sets]
    holes_union      = unary_union(holes) if holes else Polygon()
    
    if method == 0:
        # Usar dmsh para generar puntos
        geo = dmsh.Polygon(np.column_stack((xb, yb)))
        for hx, hy in h_coor_sets:
            geo -= dmsh.Polygon(np.column_stack((hx, hy)))
        X, _ = dmsh.generate(geo, dist)
        
        # Clasificar los puntos generados
        points = [Point(p[0], p[1]) for p in X]
        A = np.zeros([len(points), 1])
        cloud = np.column_stack((X, np.full((len(X), 1), region_flag), A))
    
    elif method == 1:
        # Generar una cuadrícula de puntos y filtrar
        min_x, min_y, max_x, max_y = boundary_polygon.bounds
        grid_points = generate_grid(min_x, max_x, min_y, max_y, dist)
        
        # Filtrar los puntos que están dentro del polígono y fuera de los huecos
        generated_points = [
            [x, y, region_flag, 0] for x, y in grid_points
            if boundary_polygon.contains(Point(x, y)) and not holes_union.contains(Point(x, y))
        ]
        cloud = np.array(generated_points)
    else:
        raise ValueError("Método no soportado. Use 'v1' o 'v2'.")
    
    # Agregar puntos de borde y huecos
    points = [[x, y, region_flag, 1] for x, y in zip(xb, yb)]
    for hx, hy in h_coor_sets:
        points.extend([[x, y, region_flag, 2] for x, y in zip(hx, hy)])
    cloud = np.vstack((points, cloud)) if cloud.size else np.array(points)
    
    # Aplicar aleatoriedad
    if rand != 0 and cloud.size:
        mask = cloud[:, 3] == 0
        perturbation = 0.5 * dist * (np.random.rand(np.sum(mask), 2) - 0.5)
        cloud[mask, 0:2] += perturbation
    
    return cloud

# Function to load CSV files
def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = {'x', 'y', 'flag'}
        if not required_columns.issubset(df.columns):
            raise ValueError("El archivo CSV debe contener las columnas: 'x', 'y', 'flag'")
        df = df[['x', 'y', 'flag']]
        points_by_region = (
            df.groupby('flag', group_keys=False)
            .apply(lambda group: group[['x', 'y']].values.tolist(), include_groups = False)
            .to_dict()
        )
        return points_by_region
    except Exception as e:
        raise RuntimeError(f"Error al procesar el archivo: {e}")

def generate_polygons(points_by_region):
    polygons_by_region = {}
    
    for region, points in points_by_region.items():
        # Eliminar duplicados y validar entrada
        points = pd.DataFrame(points, columns=['x', 'y']).drop_duplicates()
        
        try:
            if len(points) < 3:
                raise ValueError(f"La región {region} tiene menos de 3 puntos únicos, no se puede formar un polígono.")
            polygon = Polygon(points.values)
            if not polygon.is_valid:
                print(f"El polígono de la región {region} no es válido. Intentando crear un Convex Hull.")
                polygon = Polygon(points.values).convex_hull
            polygons_by_region[region] = polygon
        except Exception as e:
            print(f"Error al procesar la región {region}: {e}")
    return polygons_by_region

def test_region_containment(polygon_1, polygon_2):
    if polygon_1 is None or polygon_2 is None:
        raise ValueError("Los polígonos proporcionados no son válidos.")
    return polygon_2.within(polygon_1) or (polygon_2.intersects(polygon_1) and polygon_1.area > polygon_2.area)

def test_all_region_containments(polygons_by_region):
    containment_results = {region: [] for region in polygons_by_region}
    for region_1, polygon_1 in polygons_by_region.items():
        for region_2, polygon_2 in polygons_by_region.items():
            if region_1 != region_2 and test_region_containment(polygon_1, polygon_2):
                if region_2 not in containment_results[region_1]:
                    containment_results[region_1].append(region_2)
    return containment_results

def remove_duplicate_containments(containment_results):
    for region in containment_results:
        direct_contained = set(containment_results[region])
        for subregion in list(direct_contained):
            nested = set(containment_results.get(subregion, []))
            direct_contained -= nested
        containment_results[region] = list(direct_contained)
    return containment_results

def generate_clouds_for_all_regions(polygons_by_region, containment_results, num, rand, mod, gen):
    all_points = []
    
    for region, polygon in polygons_by_region.items():
        try:
            # Validar que el polígono sea válido y tenga un contorno exterior
            if not polygon.is_valid or not polygon.exterior:
                raise ValueError(f"El polígono de la región {region} no es válido.")

            # Obtener coordenadas de los huecos para la región actual
            holes = []
            for contained_region in containment_results.get(region, []):
                # Validar que los huecos existan y sean válidos
                if contained_region not in polygons_by_region:
                    print(f"Advertencia: La región {contained_region} no está en polygons_by_region.")
                    continue
                contained_polygon = polygons_by_region[contained_region]
                if not contained_polygon.is_valid or not contained_polygon.exterior:
                    print(f"Advertencia: El polígono interior {contained_region} no es válido.")
                    continue
                holes.append(np.array(contained_polygon.exterior.xy)[:, :-1])
            
            # Formatear los huecos como una lista de pares (x, y)
            hole_coordinates = [(hx, hy) for hx, hy in holes]

            # Extraer coordenadas del contorno del polígono, eliminando el último punto repetido
            xb, yb = np.array(polygon.exterior.xy)[:, :-1]
            dist = distance(xb, yb, hole_coordinates)/num
            while max(max(xb),max(yb))/dist < 21:
                dist = dist/2

            # Generar la nube de puntos para la región
            cloud = CreateCloud(xb, yb, hole_coordinates, dist, rand, region, mod)
            
            # Almacenar la nube generada
            all_points.append(cloud)

            if gen == 0:
                break

        except Exception as e:
            print(f"Error al generar nube para la región {region}: {e}")
            continue

    # Combinar todas las nubes en un único arreglo
    if all_points:
        return np.vstack(all_points)
    else:
        print("No se generaron nubes de puntos para ninguna región.")
        return np.array([])  # Retorna un arreglo vacío en caso de error

# --- Visualization ---
def GraphCloud(all_clouds, folder, image_name, eps_name):
    """
    Graph the generated point cloud and save the plot.

    Parameters:
        p (np.array): Array of points with boundary flags.
        xb (array): x-coordinates of the boundary.
        yb (array): y-coordinates of the boundary.
        h_coor_sets (list): List of tuples containing x and y coordinates of the holes.
        folder (str): Directory to save the plot.
        image_name (str): Name to save the PNG file.
        eps_name (str): Name to save the EPS file.
    """
    plt.figure(figsize=(12, 8))
    unique_flags = np.unique(all_clouds[:, 2])
    colors = plt.cm.get_cmap('tab20', len(unique_flags))

    # Scatter points by region
    for i, flag in enumerate(unique_flags):
        cloud = all_clouds[all_clouds[:, 2] == flag]
        plt.scatter(cloud[:, 0], cloud[:, 1], s=5, c=[colors(i)], label=f'Región {int(flag)}')
        bound = cloud[cloud[:,3] == 1]
        bound = np.vstack([bound, bound[0]])
        plt.plot(bound[:, 0], bound[:, 1], color=colors(i), linestyle='solid')

    plt.legend()
    plt.grid(True)
    plt.title("Generated Cloud of Points")
    plt.savefig(os.path.join(folder, image_name), format='png')
    plt.savefig(os.path.join(folder, eps_name), format='eps')
    plt.close()

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
                regions = process_csv(file_path)
                polygons = generate_polygons(regions)
                containment_results = test_all_region_containments(polygons)
                depurated_results = remove_duplicate_containments(containment_results)

                # Obtener parámetros
                num  = int(request.form.get('num', 3))
                rand = int(request.form.get('rand', 1))
                mod  = int(request.form.get('mod', 1))
                gen  = int(request.form.get('gen', 0))

                try:
                    all_clouds = generate_clouds_for_all_regions(polygons, depurated_results, num, rand, mod, gen)
                except Exception as e:
                    return f"Error generating the cloud: {e}", 500

                # Generar nombres de archivos de resultados
                timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f'plot_{timestamp}.png'
                eps_name   = f'plot_{timestamp}.eps'
                p_csv_name = f'cloud_{timestamp}.csv'

                # Crear la gráfica
                GraphCloud(all_clouds, folder=app.config['OUTPUT_FOLDER'], image_name = image_name, eps_name = eps_name)

                # Guardar la nube en CSV
                csv_path = os.path.join(app.config['OUTPUT_FOLDER'], p_csv_name)
                pd.DataFrame(all_clouds, columns=["x", "y", "region", "boundary_flag"]).to_csv(csv_path, index=False)

                # Programar eliminación de archivos
                delete_file(file_path, 3600)
                delete_file(os.path.join(app.config['OUTPUT_FOLDER'], image_name), 3600)
                delete_file(os.path.join(app.config['OUTPUT_FOLDER'], eps_name), 3600)
                delete_file(csv_path, 3600)

                # Renderizar resultados
                return render_template(
                    'cloud.html',
                    image_name = image_name,
                    eps_name   = eps_name,
                    p_csv_name = p_csv_name,
                    tables = [pd.DataFrame(all_clouds, columns=["x", "y", "region", "boundary_flag"]).to_html(classes='data')],
                    titles = ['na', 'Cloud Data']
                )


            except Exception as e:
                return f"Error reading the file: {e}", 500
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
@app.route('/CloudGen/results/<path:filename>')
def static_files(filename):
    """ 
    Serve result files.
    """
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/CloudGen/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug = True)